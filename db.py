import sqlite3
import sqlite_vec
import re
import os
import math
import pandas as pd

DB_PATH = os.environ.get("SPREADSHEET_DB", "spreadsheet.db")
EMBED_DIM = 1024
VEC_PREFIX = "vec_"
_CJK = r"0-9a-zA-Z_\u4e00-\u9fff"


def _sanitize(name: str, prefix: str, default: str) -> str:
    cleaned = re.sub(r"[^" + _CJK + r"]", "_", name) or default
    if cleaned[0].isdigit():
        cleaned = prefix + cleaned
    return cleaned


def _table_name_from_path(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    if not base:
        return "table"
    # Sanitize to a safe SQLite identifier: keep alphanumerics, underscores and
    # CJK; replace everything else (notably ".", spaces, "-") with "_". This is
    # idempotent, so calling it repeatedly on an already-derived name is safe and
    # avoids table-name mismatches when a name contains a dot (e.g. "V1.0").
    cleaned = re.sub(r"[^" + _CJK + r"]", "_", base)
    return cleaned or "table"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.execute("PRAGMA foreign_keys = OFF")
    return conn


def list_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name NOT LIKE ?",
        (VEC_PREFIX + "%",),
    ).fetchall()
    return [r["name"] for r in rows]


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return name in list_tables(conn)


def _sql_type(dtype: str) -> str:
    dt = dtype.lower()
    if "int" in dt:
        return "INTEGER"
    if "float" in dt or "double" in dt or "real" in dt:
        return "REAL"
    if "datetime" in dt or "timestamp" in dt:
        return "TEXT"
    return "TEXT"


def create_table_from_df(conn: sqlite3.Connection, name: str, df, row_texts: list[str], sheet: str | None = None) -> str:
    df = df.copy()
    if sheet is not None and "sheet" not in df.columns:
        df["sheet"] = sheet
    safe = _table_name_from_path(name)
    cols = []
    for col in df.columns:
        cname = _sanitize(str(col), "c_", "col")
        cols.append((col, cname, _sql_type(str(df[col].dtype))))
    col_defs = ", ".join(f'"{cname}" {t}' for _, cname, t in cols)
    conn.execute(f'DROP TABLE IF EXISTS "{safe}"')
    conn.execute(f'CREATE TABLE "{safe}" (row_id INTEGER PRIMARY KEY AUTOINCREMENT, __row_text TEXT, {col_defs})')
    for i, (_, row) in enumerate(df.iterrows()):
        row_values = []
        for orig, _, _ in cols:
            v = row[orig]
            if pd.isna(v):
                row_values.append(None)
            else:
                # pandas/numpy scalars must become native Python types or
                # sqlite3 stores them as BLOBs instead of INTEGER/REAL.
                row_values.append(v.item() if hasattr(v, "item") else v)
        values = [row_texts[i]] + row_values
        placeholders = ", ".join(["?"] * (len(cols) + 1))
        quoted = ", ".join(f'"{cname}"' for _, cname, _ in cols)
        conn.execute(f'INSERT INTO "{safe}" (__row_text, {quoted}) VALUES ({placeholders})', values)
    conn.commit()
    return safe


def get_rows(conn: sqlite3.Connection, name: str, limit: int | None = None) -> list[dict]:
    sql = f'SELECT * FROM "{name}"'
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


def get_preview(conn: sqlite3.Connection, name: str, n: int = 5) -> tuple[list[str], list[dict]]:
    rows = get_rows(conn, name, limit=n)
    if not rows:
        return [], []
    columns = [k for k in rows[0].keys() if k not in ("row_id", "__row_text")]
    clean = [{k: r[k] for k in columns} for r in rows]
    return columns, clean


def col_letter(n: int) -> str:
    """1-based column index → Excel-style letter (1→A, 26→Z, 27→AA)."""
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def get_row_coords(conn: sqlite3.Connection, name: str, row_id: int) -> list[tuple[str, str, object]]:
    """Return [(coord, col_name, value), ...] for one row.

    coord is spreadsheet-style, e.g. 'Sheet1!B3' (sheet from the row's
    `sheet` column, column letter from 1-based data-column position, row
    from the row's `src_row` column). Returns [] if row missing or has no
    src_row.
    """
    rows = get_rows(conn, name)
    row = next((r for r in rows if r.get("row_id") == row_id), None)
    if row is None:
        return []
    src_row = row.get("src_row")
    if src_row is None:
        return []
    sheet = row.get("sheet")
    internal = ("row_id", "__row_text", "sheet", "src_row")
    cols = [k for k in row.keys() if k not in internal]
    out = []
    for idx, col in enumerate(cols, start=1):
        coord = f"{sheet}!{col_letter(idx)}{src_row}" if sheet else f"{col_letter(idx)}{src_row}"
        out.append((coord, col, row[col]))
    return out


def delete_table(conn: sqlite3.Connection, name: str) -> None:
    conn.execute(f'DROP TABLE IF EXISTS "{name}"')
    conn.execute(f'DROP TABLE IF EXISTS "{VEC_PREFIX}{name}"')
    conn.commit()


def get_schema(conn: sqlite3.Connection, name: str) -> dict:
    cols = conn.execute(f'PRAGMA table_info("{name}")').fetchall()
    columns = [(r["name"], r["type"]) for r in cols if r["name"] not in ("row_id", "__row_text")]
    sample = conn.execute(f'SELECT * FROM "{name}" LIMIT 3').fetchall()
    sample_rows = [{k: v for k, v in dict(r).items() if k != "__row_text"} for r in sample]
    return {"table": name, "columns": columns, "sample_rows": sample_rows}


def summarize(conn: sqlite3.Connection, name: str) -> dict:
    """Return row count + per-column aggregates for *name*.

    Numeric columns (INTEGER/REAL) get 非空数/求和/平均/最小/最大; text
    columns get 非空数/去重数. Internal columns (row_id, __row_text) are skipped.
    """
    schema = get_schema(conn, name)
    row_count = conn.execute(f'SELECT COUNT(*) AS c FROM "{name}"').fetchone()["c"]
    result_columns = []
    for col_name, col_type in schema["columns"]:
        if col_type in ("INTEGER", "REAL"):
            row = conn.execute(
                f'SELECT COUNT("{col_name}") AS nn, SUM("{col_name}") AS s, '
                f'AVG("{col_name}") AS a, MIN("{col_name}") AS mn, MAX("{col_name}") AS mx '
                f'FROM "{name}"'
            ).fetchone()
            result_columns.append({
                "列名": col_name,
                "类型": col_type,
                "非空数": row["nn"],
                "求和": row["s"],
                "平均": row["a"],
                "最小": row["mn"],
                "最大": row["mx"],
                "去重数": None,
            })
        else:
            row = conn.execute(
                f'SELECT COUNT("{col_name}") AS nn, COUNT(DISTINCT "{col_name}") AS d '
                f'FROM "{name}"'
            ).fetchone()
            result_columns.append({
                "列名": col_name,
                "类型": col_type,
                "非空数": row["nn"],
                "求和": None,
                "平均": None,
                "最小": None,
                "最大": None,
                "去重数": row["d"],
            })
    return {"row_count": row_count, "columns": result_columns}


def _normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm else v


def create_vec_table(conn: sqlite3.Connection, name: str) -> None:
    vname = VEC_PREFIX + name
    conn.execute(f'DROP TABLE IF EXISTS "{vname}"')
    conn.execute(f'CREATE VIRTUAL TABLE "{vname}" USING vec0(embedding float[{EMBED_DIM}])')
    conn.commit()


def upsert_embeddings(conn: sqlite3.Connection, name: str, row_ids: list[int], vectors: list[list[float]]) -> None:
    vname = VEC_PREFIX + name
    for rid, vec in zip(row_ids, vectors):
        blob = sqlite_vec.serialize_float32(_normalize(vec))
        conn.execute(f'INSERT INTO "{vname}" (rowid, embedding) VALUES (?, ?)', (rid, blob))
    conn.commit()


def vec_search(conn: sqlite3.Connection, name: str, query_vec: list[float], k: int | None = None) -> list[tuple[int, float]]:
    vname = VEC_PREFIX + name
    blob = sqlite_vec.serialize_float32(_normalize(query_vec))
    if k is None:
        k = conn.execute(f'SELECT COUNT(*) AS c FROM "{name}"').fetchone()["c"] or 1
    sql = f'SELECT rowid, distance FROM "{vname}" WHERE embedding MATCH ? ORDER BY distance LIMIT {int(k)}'
    rows = conn.execute(sql, (blob,)).fetchall()
    return [(r["rowid"], r["distance"]) for r in rows]


def replace_vec(conn: sqlite3.Connection, name: str, row_id: int, vec: list[float]) -> None:
    """Replace the vector for a single row (delete + insert)."""
    vname = VEC_PREFIX + name
    blob = sqlite_vec.serialize_float32(_normalize(vec))
    conn.execute(f'DELETE FROM "{vname}" WHERE rowid=?', (row_id,))
    conn.execute(f'INSERT INTO "{vname}" (rowid, embedding) VALUES (?, ?)', (row_id, blob))
    conn.commit()


def delete_vec_rows(conn: sqlite3.Connection, name: str, row_ids: list[int]) -> None:
    vname = VEC_PREFIX + name
    for rid in row_ids:
        conn.execute(f'DELETE FROM "{vname}" WHERE rowid=?', (rid,))
    conn.commit()


def upsert_rows(conn: sqlite3.Connection, name: str, df, row_texts: list[str], key_col: str, mode: str, sheet: str | None = None) -> tuple[list[tuple[int, dict]], list[int]]:
    df = df.copy()
    if sheet is not None and "sheet" not in df.columns:
        df["sheet"] = sheet
    """Upsert *df* into table *name* keyed by *key_col*.

    Returns (changed, deleted):
      - changed: list of (row_id, row_dict) for inserted/updated rows
        (row_dict keyed by the table's sanitized column names).
      - deleted: list of row_ids removed (only when mode == "update").

    *mode* "update" deletes DB rows whose key is absent from the new file
    (true sync); "merge" keeps them.
    """
    safe = _table_name_from_path(name)
    table_cols = {r[1] for r in conn.execute(f'PRAGMA table_info("{safe}")').fetchall()}
    orig_cols = [c for c in df.columns if c in table_cols]
    sanitized = [_sanitize(str(c), "c_", "col") for c in orig_cols]
    san_to_orig = dict(zip(sanitized, orig_cols))

    existing = get_rows(conn, safe)
    existing_by_key: dict[str, dict] = {}
    for r in existing:
        k = r.get(key_col)
        if k is not None:
            existing_by_key[str(k)] = r

    changed: list[tuple[int, dict]] = []
    new_keys: set[str] = set()
    for i, (_, row) in enumerate(df.iterrows()):
        row_dict = {}
        for san in sanitized:
            v = row[san_to_orig[san]]
            row_dict[san] = None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)
        k = str(row_dict.get(key_col))
        new_keys.add(k)
        if k in existing_by_key:
            rid = existing_by_key[k]["row_id"]
            set_parts = ", ".join(f'"{san}"=?' for san in sanitized)
            params = [row_dict[san] for san in sanitized] + [row_texts[i]] + [rid]
            conn.execute(
                f'UPDATE "{safe}" SET {set_parts}, "__row_text"=? WHERE row_id=?',
                params,
            )
            changed.append((rid, row_dict))
        else:
            col_list = ", ".join(f'"{san}"' for san in sanitized)
            placeholders = ", ".join(["?"] * (len(sanitized) + 1))
            params = [row_dict[san] for san in sanitized] + [row_texts[i]]
            cur = conn.execute(
                f'INSERT INTO "{safe}" ({col_list}, "__row_text") VALUES ({placeholders})',
                params,
            )
            changed.append((cur.lastrowid, row_dict))

    deleted: list[int] = []
    if mode == "update":
        for r in existing:
            if str(r.get(key_col)) not in new_keys:
                rid = r["row_id"]
                deleted.append(rid)
                conn.execute(f'DELETE FROM "{safe}" WHERE row_id=?', (rid,))

    conn.commit()
    return changed, deleted
