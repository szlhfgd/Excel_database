import os
import pandas as pd
import db
import llm


def read_file(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".csv":
        for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
            try:
                return pd.read_csv(path, encoding=enc)
            except (UnicodeDecodeError, LookupError):
                continue
        return pd.read_csv(path)
    raise ValueError(f"不支持的文件类型: {ext}")


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all")
    df.columns = [str(c).strip().replace("\n", " ") for c in df.columns]
    df = df.reset_index(drop=True)
    return df


def row_texts_for_df(df: pd.DataFrame) -> list[str]:
    out = []
    for _, row in df.iterrows():
        parts = [f"{col}: {row[col]}" for col in df.columns if not pd.isna(row[col])]
        out.append(" | ".join(parts))
    return out


def ingest_file(conn: db.sqlite3.Connection, path: str) -> str:
    df = clean_df(read_file(path))
    texts = row_texts_for_df(df)
    name = db.create_table_from_df(conn, os.path.basename(path), df, texts)
    build_embeddings(conn, name)
    return name


def build_embeddings(conn: db.sqlite3.Connection, name: str) -> None:
    rows = db.get_rows(conn, name)
    if not rows:
        return
    texts = [r["__row_text"] for r in rows]
    vectors = llm.embed(texts)
    db.create_vec_table(conn, name)
    db.upsert_embeddings(conn, name, [r["row_id"] for r in rows], vectors)
