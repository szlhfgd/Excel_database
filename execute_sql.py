"""Read-only SQL guard for query paths.

Hand-written SQL (app "sql" mode) and LLM-generated NL2SQL are executed
against the database with no prior validation, so a stray DROP / DELETE /
UPDATE / ALTER (or a prompt-injected destructive statement) could wipe
user data. Every query path goes through :func:`assert_readonly_sql` before
execution, which enforces a strict read-only allow-list.

Forwards-compatible with SQLite specifics: single-statement-only (no ``;``
multi-statement), and only read prefixes (SELECT / WITH / VALUES / PRAGMA
read) are accepted.
"""

import re

# Statements that mutate state and must never be allowed.
_BLOCKED_KEYWORDS = (
    "DROP", "DELETE", "UPDATE", "INSERT", "REPLACE", "ALTER",
    "CREATE", "ATTACH", "DETACH", "VACUUM", "REINDEX",
    "ANALYZE", "TRUNCATE", "BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT",
)

# Function calls that can read/write files or load extensions.
_BLOCKED_FUNCTIONS = ("load_extension", "writefile", "readfile")

# Allowed first tokens for a read-only statement.
_READ_PREFIX_RE = re.compile(
    r"^\s*(SELECT|WITH|VALUES|PRAGMA|EXPLAIN)\b", re.IGNORECASE | re.DOTALL
)


class ReadOnlySQLViolation(ValueError):
    """Raised when *sql* is not a safe, single, read-only statement."""


def _strip_comments(sql: str) -> str:
    """Remove line (``--``) and block (``/* ... */``) comments.

    Naive but sufficient for guarding: comments could otherwise smuggle a
    blocked keyword past the prefix check (e.g. ``SELECT * FROM t; -- DROP``).
    Keeping the text outside comments intact, we reject multi-statement and
    blocked bodies regardless.
    """
    out: list[str] = []
    i, n = 0, len(sql)
    while i < n:
        if sql[i : i + 2] == "--":
            nl = sql.find("\n", i)
            i = n if nl == -1 else nl + 1
        elif sql[i : i + 2] == "/*":
            end = sql.find("*/", i + 2)
            i = n if end == -1 else end + 2
        else:
            out.append(sql[i])
            i += 1
    return "".join(out)


def _single_statement(raw: str) -> bool:
    """Reject multi-statement SQL.

    A ``;`` inside a comment or string would otherwise smuggle a second
    destructive statement past the allow-list, so the check runs on the raw
    input (not the comment-stripped body). Only an optional single trailing
    ``;`` is allowed.
    """
    body = raw.rstrip()
    if body.endswith(";"):
        body = body[:-1].rstrip()
    return ";" not in body


def assert_readonly_sql(sql: str, allowed_tables: set[str] | None = None) -> str:
    """Validate that *sql* is a single, read-only statement.

    Returns the original *sql* on success; raises :class:`ReadOnlySQLViolation`
    otherwise. When *allowed_tables* is given, referenced table names must be
    restricted to that set (defence in depth; optional).

    Raises:
        ReadOnlySQLViolation: if the statement is multi-statement, starts with
            a blocked keyword, calls a blocked function, or references a
            table outside *allowed_tables*.
    """
    if not sql or not sql.strip():
        raise ReadOnlySQLViolation("SQL 为空。")

    if not _single_statement(sql):
        raise ReadOnlySQLViolation("仅允许单条只读 SQL，禁止多语句（;）。")

    body = _strip_comments(sql)
    if not body.strip():
        raise ReadOnlySQLViolation("SQL 仅含注释。")

    # Block any statement not starting with a read prefix.
    if not _READ_PREFIX_RE.match(body):
        # Distinguish destructive statements for a clearer message.
        first = re.match(r"^\s*([A-Za-z]+)", body)
        head = (first.group(1).upper() if first else "") or "无"
        if head in _BLOCKED_KEYWORDS:
            raise ReadOnlySQLViolation(f"禁止写操作 SQL：{head}。只允许 SELECT/WITH/VALUES/PRAGMA 只读查询。")
        raise ReadOnlySQLViolation("仅允许只读 SQL（SELECT/WITH/VALUES/PRAGMA）。")

    # Block dangerous function calls anywhere in the statement.
    upper = body.upper()
    for fn in _BLOCKED_FUNCTIONS:
        if fn.upper() in upper:
            raise ReadOnlySQLViolation(f"禁止调用危险函数：{fn}。")

    # Optional: restrict referenced table names to the allowed set.
    if allowed_tables is not None:
        refs = set(re.findall(r"\bFROM\s+([A-Za-z_][A-Za-z0-9_]*)\b", body, re.IGNORECASE))
        refs |= set(
            re.findall(r"\bJOIN\s+([A-Za-z_][A-Za-z0-9_]*)\b", body, re.IGNORECASE)
        )
        unknown = refs - set(allowed_tables)
        if unknown:
            raise ReadOnlySQLViolation(f"引用了未授权表：{', '.join(sorted(unknown))}。")

    return sql
