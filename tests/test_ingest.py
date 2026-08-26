import os
from unittest import mock
import pandas as pd

os.environ["SPREADSHEET_DB"] = ":memory:"
import db
import ingest


def _write_csv(path, df, encoding="utf-8"):
    df.to_csv(path, index=False, encoding=encoding)


def test_ingest_utf8_cleans_empty_rows():
    conn = db.get_conn()
    df = pd.DataFrame({"名称": ["甲", "乙", None], "值": [1, 2, None]})
    p = os.path.join(os.environ.get("TMP", "."), "t_utf8.csv")
    _write_csv(p, df)
    with mock.patch.object(llm_mod(), "embed", return_value=[[0.0] * db.EMBED_DIM] * 2):
        name = ingest.ingest_file(conn, p)
    rows = db.get_rows(conn, name)
    assert len(rows) == 2
    assert name == "t_utf8"
    vec_present = conn.execute("SELECT name FROM sqlite_master WHERE name='vec_t_utf8'").fetchone()
    assert vec_present is not None


def test_ingest_gbk_encoding():
    conn = db.get_conn()
    df = pd.DataFrame({"城市": ["北京", "上海"], "销量": [10, 20]})
    p = os.path.join(os.environ.get("TMP", "."), "t_gbk.csv")
    _write_csv(p, df, encoding="gbk")
    with mock.patch.object(llm_mod(), "embed", return_value=[[0.0] * db.EMBED_DIM] * 2):
        name = ingest.ingest_file(conn, p)
    rows = db.get_rows(conn, name)
    assert rows[0]["城市"] == "北京"


def llm_mod():
    import llm
    return llm
