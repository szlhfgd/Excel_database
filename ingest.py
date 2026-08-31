import os
import math
import pandas as pd
import db
import llm

SEARCH_COLS: list[str] = []          # 参与向量化的列；空 = 全部数据列
SEARCH_COL_MIN_CHARS = 10            # 单个列值字数超过该值才纳入拼接
_INTERNAL_COLS = ("row_id", "__row_text")


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _search_text_for_row(row: dict, search_cols: list[str] | None = None) -> str:
    """拼接用于向量化的文本：仅取 search_cols 中、单个列值字数 > SEARCH_COL_MIN_CHARS 的列。

    过滤后为空时回退到全部数据列（不过滤长度），保证每行都有可向量化内容。
    """
    data_cols = [k for k in row if k not in _INTERNAL_COLS]
    cols = [c for c in search_cols if c in data_cols] if search_cols else data_cols
    parts = []
    for c in cols:
        v = row.get(c)
        if _is_missing(v):
            continue
        s = str(v)
        if len(s) > SEARCH_COL_MIN_CHARS:
            parts.append(f"{c}: {s}")
    text = " ".join(parts)
    if not text.strip():
        text = " ".join(f"{c}: {row[c]}" for c in data_cols if not _is_missing(row.get(c)))
    return text


def read_file(path: str, header_row: int = 1) -> tuple[pd.DataFrame, str]:
    """Read *path* into a DataFrame. Returns (df, sheet_name).

    *header_row* is 1-based: row 1 means the first row holds the column
    names (the previous default). Rows above the header are skipped.
    The returned df carries a `src_row` column with the 1-based
    spreadsheet row number for each data row (used for cell coordinates).
    """
    ext = os.path.splitext(path)[1].lower()
    pandas_header = header_row - 1
    if ext in (".xlsx", ".xls"):
        with pd.ExcelFile(path) as xl:
            sheet_name = xl.sheet_names[0]
            df = xl.parse(sheet_name, header=pandas_header)
    elif ext == ".csv":
        sheet_name = os.path.splitext(os.path.basename(path))[0]
        df = None
        for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
            try:
                df = pd.read_csv(path, header=pandas_header, encoding=enc)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        if df is None:
            df = pd.read_csv(path, header=pandas_header)
    else:
        raise ValueError(f"不支持的文件类型: {ext}")
    df = df.copy()
    df["src_row"] = header_row + 1 + df.index
    return df, sheet_name


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    data_cols = [c for c in df.columns if c != "src_row"]
    df = df.dropna(how="all", subset=data_cols)
    df.columns = [str(c).strip().replace("\n", " ") for c in df.columns]
    df = df.reset_index(drop=True)
    return df


def row_texts_for_df(df: pd.DataFrame) -> list[str]:
    skip = {"src_row", "sheet"}
    out = []
    for _, row in df.iterrows():
        parts = [f"{col}: {row[col]}" for col in df.columns if col not in skip and not pd.isna(row[col])]
        out.append(" | ".join(parts))
    return out


def ingest_file(conn: db.sqlite3.Connection, path: str, on_progress=None, name: str | None = None, header_row: int = 1, key_col=None, mode="replace") -> tuple[str, bool]:
    def prog(frac: float, msg: str):
        if on_progress:
            on_progress(frac, msg)

    prog(0.02, "正在读取文件…")
    name = db._table_name_from_path(name) if name else db._table_name_from_path(os.path.basename(path))
    updated = db.table_exists(conn, name)
    if mode == "create" and updated:
        raise ValueError(
            f"表 '{name}' 已存在，无法以'新建表'模式导入。"
            f"请改用'替换'/'更新'/'合并'，或更换文件名。"
        )
    raw_df, sheet_name = read_file(path, header_row=header_row)
    df = clean_df(raw_df)
    df = df.copy()
    df["sheet"] = sheet_name
    prog(0.10, "文件读取完成")
    texts = row_texts_for_df(df)

    # The data-table writes and the companion vector writes must be atomic:
    # any failure partway leaves no "table without vectors" (or stale) state.
    # Python's sqlite3 legacy mode auto-begins a transaction only before DML,
    # so CREATE/DROP TABLE would otherwise commit immediately and defeat the
    # rollback. Begin an explicit transaction and roll back on any error so the
    # whole import is all-or-nothing.
    conn.execute("BEGIN")
    try:
        if updated and key_col and mode in ("update", "merge"):
            changed, deleted = db.upsert_rows(conn, name, df, texts, key_col, mode, sheet=sheet_name, commit=False)
            if changed:
                embed_texts = [_search_text_for_row(r, SEARCH_COLS) for _, r in changed]
                vectors = llm.embed(embed_texts)
                for (rid, _), vec in zip(changed, vectors):
                    db.replace_vec(conn, name, rid, vec, commit=False)
            if deleted:
                db.delete_vec_rows(conn, name, deleted, commit=False)
            conn.commit()
            prog(1.0, "导入完成")
            return name, True
        if updated:
            db.delete_table(conn, name, commit=False)
        db.create_table_from_df(conn, name, df, texts, sheet=sheet_name, commit=False)
        prog(0.30, "已建表，开始生成向量…")
        build_embeddings(conn, name, on_progress=prog, commit=False)
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    prog(1.0, "导入完成")
    return name, updated


def build_embeddings(conn: db.sqlite3.Connection, name: str, on_progress=None, commit: bool = True) -> None:
    rows = db.get_rows(conn, name)
    if not rows:
        return
    texts = [_search_text_for_row(r, SEARCH_COLS) for r in rows]
    vectors: list[list[float]] = []
    n = len(texts)
    batch = 32
    for i in range(0, n, batch):
        chunk = texts[i : i + batch]
        vecs = llm.embed(chunk)
        vectors.extend(vecs)
        done = min(i + len(chunk), n)
        if on_progress:
            frac = 0.30 + (done / n) * 0.60
            on_progress(frac, f"生成向量 {done}/{n}")
    db.create_vec_table(conn, name, commit=commit)
    db.upsert_embeddings(conn, name, [r["row_id"] for r in rows], vectors, commit=commit)
