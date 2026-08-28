import os
import uuid
from unittest import mock
import pandas as pd
import pytest

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
        name, _ = ingest.ingest_file(conn, p)
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
        name, _ = ingest.ingest_file(conn, p)
    rows = db.get_rows(conn, name)
    assert rows[0]["城市"] == "北京"


def llm_mod():
    import llm
    return llm


def test_read_file_unsupported_type_raises():
    with pytest.raises(ValueError, match="不支持"):
        ingest.read_file("x.txt")


def test_read_file_gbk_csv():
    import tempfile
    p = os.path.join(tempfile.gettempdir(), "t_gbk.csv")
    pd.DataFrame({"城市": ["北京", "上海"], "销量": [10, 20]}).to_csv(p, index=False, encoding="gbk")
    df, _ = ingest.read_file(p)
    assert df.iloc[0]["城市"] == "北京"


def test_read_file_xlsx():
    import tempfile
    p = os.path.join(tempfile.gettempdir(), "t.xlsx")
    pd.DataFrame({"x": [1, 2], "y": ["a", "b"]}).to_excel(p, index=False)
    df, _ = ingest.read_file(p)
    assert df.shape == (2, 3)  # x, y, src_row
    assert "src_row" in df.columns
    assert df.iloc[0]["x"] == 1


def test_clean_df_drops_all_nan_row_and_strips_columns():
    df = pd.DataFrame({" A ": [1, None], "B": [2, None]})
    out = ingest.clean_df(df)
    assert len(out) == 1
    assert list(out.columns) == ["A", "B"]


def test_clean_df_replaces_newline_in_column():
    df = pd.DataFrame({"X\nY": [1]})
    assert "X Y" in list(ingest.clean_df(df).columns)


def test_row_texts_for_df_skips_nan():
    df = pd.DataFrame({"name": ["a", "b"], "val": ["1", None]})
    assert ingest.row_texts_for_df(df) == ["name: a | val: 1", "name: b"]


def test_build_embeddings_empty_early_return():
    conn = db.get_conn()
    name = db.create_table_from_df(
        conn, "e.xlsx", pd.DataFrame({"a": pd.Series([], dtype="object")}), []
    )
    ingest.build_embeddings(conn, name)
    assert ("vec_" + name) not in db.list_tables(conn)


def test_build_embeddings_batches_and_creates_vec():
    conn = db.get_conn()
    n = 70
    df = pd.DataFrame({"c": [f"row {i}" for i in range(n)]})
    texts = [f"text {i}" for i in range(n)]
    name = db.create_table_from_df(conn, "b.xlsx", df, texts)
    chunks = []

    def fake_embed(batch):
        chunks.append(len(batch))
        return [[0.0] * db.EMBED_DIM for _ in batch]

    fracs = []
    with mock.patch.object(llm_mod(), "embed", side_effect=fake_embed):
        ingest.build_embeddings(conn, name, on_progress=lambda f, m: fracs.append(f))
    assert max(chunks) <= 32
    assert sum(chunks) == n
    vec_name = "vec_" + name
    exists = conn.execute("SELECT name FROM sqlite_master WHERE name=?", (vec_name,)).fetchone()
    assert exists is not None
    assert max(fracs) == pytest.approx(0.9, abs=1e-6)


def test_ingest_file_new_table_reports_not_updated():
    conn = db.get_conn()
    name = "upd_new_" + uuid.uuid4().hex[:8]
    df = pd.DataFrame({"a": [1, 2]})
    p = os.path.join(os.environ.get("TMP", "."), name + ".csv")
    _write_csv(p, df)
    with mock.patch.object(llm_mod(), "embed", side_effect=lambda b: [[0.0] * db.EMBED_DIM for _ in b]):
        got_name, updated = ingest.ingest_file(conn, p)
    assert got_name == name
    assert updated is False


def test_ingest_file_existing_table_is_updated_and_replaced():
    conn = db.get_conn()
    name = "upd_existing_" + uuid.uuid4().hex[:8]
    p = os.path.join(os.environ.get("TMP", "."), name + ".csv")
    _write_csv(p, pd.DataFrame({"a": [1, 2]}))
    with mock.patch.object(llm_mod(), "embed", side_effect=lambda b: [[0.0] * db.EMBED_DIM for _ in b]):
        _, updated1 = ingest.ingest_file(conn, p)
    _write_csv(p, pd.DataFrame({"a": [10, 20, 30]}))
    with mock.patch.object(llm_mod(), "embed", side_effect=lambda b: [[0.0] * db.EMBED_DIM for _ in b]):
        got_name, updated2 = ingest.ingest_file(conn, p)
    assert got_name == name
    assert updated1 is False
    assert updated2 is True
    rows = db.get_rows(conn, name)
    assert len(rows) == 3
    assert [r["a"] for r in rows] == [10, 20, 30]


def test_ingest_file_uses_explicit_name_over_path_basename():
    # The table name must come from the original filename, not the (often
    # random / non-Chinese) temp path the UI writes the upload bytes to.
    conn = db.get_conn()
    table = "中文表_" + uuid.uuid4().hex[:8]
    p = os.path.join(os.environ.get("TMP", "."), "tmp_random_abc123.csv")
    _write_csv(p, pd.DataFrame({"a": [1, 2]}))
    with mock.patch.object(llm_mod(), "embed", side_effect=lambda b: [[0.0] * db.EMBED_DIM for _ in b]):
        got_name, _ = ingest.ingest_file(conn, p, name=table + ".csv")
    assert got_name == table
    assert table in db.list_tables(conn)


def test_read_file_reads_only_first_sheet():
    import tempfile
    p = os.path.join(tempfile.gettempdir(), "multi_sheet.xlsx")
    sheet1 = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    sheet2 = pd.DataFrame({"x": [99], "y": ["zzz"]})
    with pd.ExcelWriter(p, engine="openpyxl") as w:
        sheet1.to_excel(w, sheet_name="Sheet1", index=False)
        sheet2.to_excel(w, sheet_name="Sheet2", index=False)
    df, _ = ingest.read_file(p)
    assert df.shape == (2, 3)  # x, y, src_row
    assert df.iloc[0]["x"] == 1
    assert 99 not in df["x"].tolist()


def test_read_file_uses_header_row_xlsx():
    import tempfile
    p = os.path.join(tempfile.gettempdir(), "hdr.xlsx")
    pd.DataFrame([["报表", "2024"], ["x", "y"], [1, "a"], [2, "b"]]).to_excel(
        p, index=False, header=False
    )
    out, _ = ingest.read_file(p, header_row=2)
    assert list(out.columns) == ["x", "y", "src_row"]
    assert out.shape == (2, 3)
    assert out.iloc[0]["x"] == 1


def test_read_file_uses_header_row_csv():
    import tempfile
    p = os.path.join(tempfile.gettempdir(), "hdr.csv")
    pd.DataFrame([["报表", "2024"], ["x", "y"], [1, "a"], [2, "b"]]).to_csv(
        p, index=False, header=False
    )
    out, _ = ingest.read_file(p, header_row=2)
    assert list(out.columns) == ["x", "y", "src_row"]
    assert out.shape == (2, 3)


def test_ingest_file_passes_header_row_to_read_file():
    conn = db.get_conn()
    p = os.path.join(os.environ.get("TMP", "."), "hdr_fwd.csv")
    pd.DataFrame({"a": [1]}).to_csv(p, index=False)
    with mock.patch.object(ingest, "read_file", return_value=(pd.DataFrame({"a": [1]}), "Sheet1")) as m, \
         mock.patch.object(llm_mod(), "embed", return_value=[[0.0] * db.EMBED_DIM]):
        ingest.ingest_file(conn, p, header_row=2)
    assert m.call_args is not None
    assert m.call_args.kwargs.get("header_row") == 2


def test_search_text_for_row_uses_search_cols_and_min_chars():
    row = {
        "row_id": 1,
        "__row_text": "ignored",
        "title": "这是一段很长的标题内容超过十个字",
        "code": "AB12",
        "note": "短",
    }
    text = ingest._search_text_for_row(row, ["title", "code", "note"])
    assert text == "title: 这是一段很长的标题内容超过十个字"


def test_search_text_for_row_empty_search_cols_uses_all_data_cols():
    row = {
        "row_id": 1,
        "__row_text": "ignored",
        "desc": "这是一段超过十个字的描述文本",
        "id": "x",
    }
    text = ingest._search_text_for_row(row, [])
    assert text == "desc: 这是一段超过十个字的描述文本"


def test_search_text_for_row_falls_back_when_all_short():
    row = {"row_id": 1, "__row_text": "ignored", "a": "短", "b": "小"}
    text = ingest._search_text_for_row(row, ["a", "b"])
    assert text == "a: 短 b: 小"


def test_build_embeddings_embeds_search_text_not_row_text():
    conn = db.get_conn()
    name = "emb_" + uuid.uuid4().hex[:8]
    df = pd.DataFrame({"desc": ["这是一段很长的描述文本用于向量化测试"], "other": ["短"]})
    db.create_table_from_df(conn, name, df, ["__row_text__short"])
    captured = []

    def fake_embed(batch):
        captured.extend(batch)
        return [[0.0] * db.EMBED_DIM for _ in batch]

    with mock.patch.object(llm_mod(), "embed", side_effect=fake_embed):
        with mock.patch.object(ingest, "SEARCH_COLS", ["desc"]):
            ingest.build_embeddings(conn, name)
    assert captured == ["desc: 这是一段很长的描述文本用于向量化测试"]
