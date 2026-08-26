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


def _sanitize_table_name(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    return _sanitize(base, "t_", "table")


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


def create_table_from_df(conn: sqlite3.Connection, name: str, df, row_texts: list[str]) -> str:
    safe = _sanitize_table_name(name)
    cols = []
    for col in df.columns:
        cname = _sanitize(str(col), "c_", "col")
        cols.append((col, cname, _sql_type(str(df[col].dtype))))
    col_defs = ", ".join(f'"{cname}" {t}' for _, cname, t in cols)
    conn.execute(f'DROP TABLE IF EXISTS "{safe}"')
    conn.execute(f'CREATE TABLE "{safe}" (row_id INTEGER PRIMARY KEY AUTOINCREMENT, __row_text TEXT, {col_defs})')
    for i, (_, row) in enumerate(df.iterrows()):
        values = [row_texts[i]] + [None if pd.isna(row[orig]) else row[orig] for orig, _, _ in cols]
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
