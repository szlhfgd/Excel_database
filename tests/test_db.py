import os
import sqlite3
import uuid
import pytest
import pandas as pd

os.environ["SPREADSHEET_DB"] = ":memory:"
from src.data import db


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


def test_vec_search_unbounded_returns_all_rows():
    conn = db.get_conn()
    name = db.create_table_from_df(conn, "v.xlsx", __make_df(), ["x", "y", "z"])
    db.create_vec_table(conn, name)
    db.upsert_embeddings(conn, name, [1, 2, 3], [
        [1.0] + [0.0] * (db.EMBED_DIM - 1),
        [0.0] * db.EMBED_DIM,
        [0.0] * db.EMBED_DIM,
    ])
    res = db.vec_search(conn, name, [1.0] + [0.0] * (db.EMBED_DIM - 1), k=None)
    assert len(res) == 3


def test_ingest_sanitizes_dirty_column_names():
    conn = db.get_conn()
    import pandas as pd
    df = pd.DataFrame({"姓名, 姓名": ["alice", "bob"], "Unnamed: 1": [10, 20]})
    name = db.create_table_from_df(conn, "脏列名.xlsx", df, ["r1", "r2"])
    rows = db.get_rows(conn, name)
    assert len(rows) == 2
    cols = [c for c, _ in db.get_schema(conn, name)["columns"]]
    assert "姓名__姓名" in cols
    assert "Unnamed__1" in cols
    assert rows[0]["姓名__姓名"] == "alice"


def __make_df():
    import pandas as pd
    return pd.DataFrame({"name": ["a", "b", "c"], "val": [1, 2, 3]})


def test_sanitize_leading_digit_gets_prefix():
    assert db._sanitize("123abc", "t_", "table") == "t_123abc"


def test_sanitize_special_chars_to_underscore():
    assert db._sanitize("foo bar!", "c_", "col") == "foo_bar_"


def test_sanitize_empty_uses_default():
    assert db._sanitize("", "c_", "col") == "col"


def test_table_name_sanitizes_special_chars():
    # Dots, spaces and parens are normalized to "_" so the result is a safe
    # SQLite identifier (and idempotent, so "V1.0" -> "V1_0" instead of being
    # truncated by a second os.path.splitext pass).
    assert db._table_name_from_path("My File (2).xlsx") == "My_File__2_"


def test_table_name_keeps_leading_digit():
    assert db._table_name_from_path("123.xlsx") == "123"


def test_table_name_empty_uses_default():
    assert db._table_name_from_path("") == "table"


def test_table_name_sanitizes_double_quote():
    # Double quotes (and any other non-identifier char) become "_" rather than
    # being escaped inside a quoted identifier, which keeps vec-table names valid.
    assert db._table_name_from_path('a"b.xlsx') == "a_b"


def test_sql_type_mapping():
    assert db._sql_type("int64") == "INTEGER"
    assert db._sql_type("float64") == "REAL"
    assert db._sql_type("datetime64[ns]") == "TEXT"
    assert db._sql_type("object") == "TEXT"
    assert db._sql_type("bool") == "TEXT"


def test_normalize_l2():
    out = db._normalize([3.0, 4.0])
    assert out[0] == pytest.approx(0.6)
    assert out[1] == pytest.approx(0.8)


def test_normalize_zero_vector_unchanged():
    assert db._normalize([0.0, 0.0]) == [0.0, 0.0]


def test_normalize_unit():
    assert db._normalize([5.0]) == [1.0]


def test_get_schema_excludes_internal_columns():
    conn = db.get_conn()
    df = pd.DataFrame({"name": ["a", "b"], "val": [1, 2]})
    name = db.create_table_from_df(conn, "s.xlsx", df, ["n a", "n b"])
    schema = db.get_schema(conn, name)
    assert schema["columns"] == [("name", "TEXT"), ("val", "INTEGER")]
    assert schema["sample_rows"] == [
        {"row_id": 1, "name": "a", "val": 1},
        {"row_id": 2, "name": "b", "val": 2},
    ]


def test_get_schema_column_samples_distinct_non_null():
    conn = db.get_conn()
    df = pd.DataFrame({"name": ["a", "a", "b", None], "val": [1, 1, 2, 3]})
    name = db.create_table_from_df(conn, "cs.xlsx", df, ["r1", "r2", "r3", "r4"])
    schema = db.get_schema(conn, name)
    samples = schema["column_samples"]
    assert set(samples["name"]) == {"a", "b"}  # distinct, non-null
    assert set(samples["val"]) == {"1", "2", "3"}
    assert "row_id" not in samples
    assert "__row_text" not in samples


def test_list_tables_excludes_vec_tables():
    conn = db.get_conn()
    name = db.create_table_from_df(conn, "real.xlsx", pd.DataFrame({"a": [1, 2]}), ["x", "y"])
    db.create_vec_table(conn, name)
    tables = db.list_tables(conn)
    assert name in tables
    assert ("vec_" + name) not in tables
    assert all(not t.startswith("vec_") for t in tables)


def test_get_preview_returns_first_n_rows():
    conn = db.get_conn()
    name = "prev_" + uuid.uuid4().hex[:8]
    df = pd.DataFrame({"name": [f"r{i}" for i in range(10)], "val": list(range(10))})
    db.create_table_from_df(conn, name, df, [f"t{i}" for i in range(10)])
    cols, rows = db.get_preview(conn, name, n=5)
    assert cols == ["name", "val"]
    assert len(rows) == 5
    assert [r["name"] for r in rows] == [f"r{i}" for i in range(5)]
    assert "__row_text" not in cols and "row_id" not in cols
