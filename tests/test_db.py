import os
import sqlite3
import pytest

os.environ["SPREADSHEET_DB"] = ":memory:"
import db


def test_create_and_read_roundtrip():
    conn = db.get_conn()
    df = __make_df()
    texts = ["a 1", "b 2", "c 3"]
    name = db.create_table_from_df(conn, "测试表.xlsx", df, texts)
    assert db.table_exists(conn, name)
    assert name == "测试表"
    rows = db.get_rows(conn, name)
    assert len(rows) == 3
    assert rows[0]["__row_text"] == "a 1"
    assert rows[0]["name"] == "a"


def test_delete_table_cleans_data_and_vec():
    conn = db.get_conn()
    name = db.create_table_from_df(conn, "t.xlsx", __make_df(), ["x", "y", "z"])
    db.create_vec_table(conn, name)
    db.upsert_embeddings(conn, name, [1, 2, 3], [[0.0] * db.EMBED_DIM] * 3)
    assert db.table_exists(conn, name)
    db.delete_table(conn, name)
    assert not db.table_exists(conn, name)
    assert "vec_t" not in db.list_tables(conn)


def test_vec_search_returns_nearest():
    conn = db.get_conn()
    name = db.create_table_from_df(conn, "v.xlsx", __make_df(), ["x", "y", "z"])
    db.create_vec_table(conn, name)
    db.upsert_embeddings(conn, name, [1, 2, 3], [
        [1.0] + [0.0] * (db.EMBED_DIM - 1),
        [0.0] * db.EMBED_DIM,
        [0.0] * db.EMBED_DIM,
    ])
    res = db.vec_search(conn, name, [1.0] + [0.0] * (db.EMBED_DIM - 1), k=2)
    assert res[0][0] == 1
    assert res[0][1] < 1e-6


def __make_df():
    import pandas as pd
    return pd.DataFrame({"name": ["a", "b", "c"], "val": [1, 2, 3]})
