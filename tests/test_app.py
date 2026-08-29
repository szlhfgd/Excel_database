import sqlite3
import types
import uuid
from pathlib import Path

import pytest

import app
import db as db_mod
import llm as llm_mod
import search as search_mod
import ingest as ingest_mod

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _uniq() -> str:
    return "t_" + uuid.uuid4().hex[:8]


# ---- _to_csv ---------------------------------------------------------------


def test_to_csv_dict_list_uses_utf8sig_bom():
    out = app._to_csv([{"姓名": "张三", "年龄": 30}])
    assert out[:3] == b"\xef\xbb\xbf"
    text = out.decode("utf-8-sig")
    assert "姓名,年龄" in text
    assert "张三,30" in text


def test_to_csv_empty_returns_empty_bytes():
    assert app._to_csv([]) == b""


def test_to_csv_list_of_lists_uses_csv_writer():
    out = app._to_csv([[1, 2], [3, 4]])
    text = out.decode("utf-8-sig")
    assert "1,2" in text
    assert "3,4" in text


# ---- _run_query ------------------------------------------------------------


def test_run_query_returns_columns_and_rows():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (a INT, b TEXT)")
    conn.execute("INSERT INTO t VALUES (1, 'x')")
    cols, rows = app._run_query(conn, "SELECT * FROM t")
    assert cols == ["a", "b"]
    assert rows == [{"a": 1, "b": "x"}]


def test_run_query_no_result_returns_empty():
    conn = sqlite3.connect(":memory:")
    cols, rows = app._run_query(conn, "CREATE TABLE t (a INT)")
    assert cols == []
    assert rows == []


# ---- _ask / ask_query ------------------------------------------------------


def test_ask_retries_once_on_sql_error(monkeypatch):
    calls = {"gen": 0, "run": 0}

    def fake_gen(schemas, query, prev_error=None):
        calls["gen"] += 1
        if calls["gen"] == 1:
            assert prev_error is None
            return "BAD SQL"
        assert prev_error is not None
        return "SELECT * FROM t"

    def fake_run(conn, sql):
        calls["run"] += 1
        if calls["run"] == 1:
            raise sqlite3.OperationalError("near BAD")
        return (["a", "b"], [{"a": 1, "b": 2}])

    monkeypatch.setattr(llm_mod, "generate_sql", fake_gen)
    monkeypatch.setattr(app, "_run_query", fake_run)
    monkeypatch.setattr(db_mod, "get_schema", lambda c, t: {"table": t, "columns": [], "sample_rows": []})

    sql, cols, rows, err = app._ask(None, ["t"], "q")
    assert sql == "SELECT * FROM t"
    assert calls["gen"] == 2
    assert calls["run"] == 2
    assert rows == [{"a": 1, "b": 2}]
    assert err is None


def test_ask_returns_error_after_max_retries(monkeypatch):
    monkeypatch.setattr(llm_mod, "generate_sql", lambda s, q, prev_error=None: "BAD")
    monkeypatch.setattr(app, "_run_query", lambda c, sql: (_ for _ in ()).throw(sqlite3.OperationalError("err")))
    monkeypatch.setattr(db_mod, "get_schema", lambda c, t: {"table": t, "columns": [], "sample_rows": []})

    sql, cols, rows, err = app._ask(None, ["t"], "q")
    assert err is not None
    assert "NL2SQL" in err
    assert rows is None


def test_ask_query_success(monkeypatch):
    monkeypatch.setattr(llm_mod, "generate_sql", lambda schemas, q, prev_error=None: "SELECT * FROM t")
    monkeypatch.setattr(app, "_run_query", lambda c, sql: (["a"], [{"a": 1}]))
    monkeypatch.setattr(db_mod, "get_schema", lambda c, t: {"table": t, "columns": [], "sample_rows": []})
    sql, cols, rows, err = app.ask_query(None, ["t"], "q")
    assert sql == "SELECT * FROM t"
    assert rows == [{"a": 1}]
    assert err is None


def test_ask_query_surfaces_error(monkeypatch):
    monkeypatch.setattr(app, "_ask", lambda c, s, q, max_attempts=2: ("SELECT * FROM t", [], None, "boom"))
    sql, cols, rows, err = app.ask_query(None, ["t"], "q")
    assert err == "boom"
    assert rows == []


# ---- hybrid_query ----------------------------------------------------------


def test_hybrid_query_success(monkeypatch):
    monkeypatch.setattr(llm_mod, "embed", lambda texts: [[0.1] * 1024])
    monkeypatch.setattr(search_mod, "hybrid_search", lambda c, tables, q, vec, recall_pool=None: [("t", 1, 0.9)])
    monkeypatch.setattr(app, "_fetch_row_by_id", lambda c, t, r: {"row_id": 1, "__row_text": "x", "a": 1})
    monkeypatch.setattr(llm_mod, "rerank", lambda q, docs: [0.5] * len(docs))
    rows, err = app.hybrid_query(None, ["t"], "q")
    assert err is None
    assert rows and rows[0]["a"] == 1
    assert rows[0]["__table"] == "t"
    assert rows[0]["__row_id"] == 1


def test_hybrid_query_error_surfaces(monkeypatch):
    monkeypatch.setattr(llm_mod, "embed", lambda texts: (_ for _ in ()).throw(RuntimeError("embed down")))
    rows, err = app.hybrid_query(None, ["t"], "q")
    assert rows == []
    assert err is not None and "搜索出错" in err


def test_hybrid_query_reranks_to_top5(monkeypatch):
    monkeypatch.setattr(llm_mod, "embed", lambda texts: [[0.1] * 1024])
    monkeypatch.setattr(
        search_mod,
        "hybrid_search",
        lambda c, tables, q, vec, recall_pool=None: [("t", i, 1.0 - i * 0.1) for i in range(8)],
    )
    monkeypatch.setattr(
        app, "_fetch_row_by_id", lambda c, t, r: {"row_id": r, "__row_text": f"text {r}", "a": r}
    )
    # rerank score = document index → the last candidate scores highest
    monkeypatch.setattr(llm_mod, "rerank", lambda q, docs: [float(i) for i in range(len(docs))])
    rows, err = app.hybrid_query(None, ["t"], "q", top_n=5)
    assert err is None
    assert len(rows) == 5
    assert rows[0]["__row_id"] == 7


def test_hybrid_query_rerank_failure_falls_back_to_rrf(monkeypatch):
    monkeypatch.setattr(llm_mod, "embed", lambda texts: [[0.1] * 1024])
    monkeypatch.setattr(
        search_mod,
        "hybrid_search",
        lambda c, tables, q, vec, recall_pool=None: [("t", i, 1.0 - i * 0.1) for i in range(8)],
    )
    monkeypatch.setattr(
        app, "_fetch_row_by_id", lambda c, t, r: {"row_id": r, "__row_text": f"text {r}", "a": r}
    )
    monkeypatch.setattr(llm_mod, "rerank", lambda q, docs: (_ for _ in ()).throw(RuntimeError("rerank down")))
    rows, err = app.hybrid_query(None, ["t"], "q", top_n=5)
    assert err is None
    assert len(rows) == 5
    # fallback keeps RRF order → row 0 first
    assert rows[0]["__row_id"] == 0


def test_hybrid_query_reranks_beyond_old_pool(monkeypatch):
    # 25 RRF candidates; the OLD pool was 20, so candidate 24 would have been
    # dropped. With rerank-all it is re-scored and (highest rerank score) surfaces.
    monkeypatch.setattr(llm_mod, "embed", lambda texts: [[0.1] * 1024])
    monkeypatch.setattr(
        search_mod,
        "hybrid_search",
        lambda c, tables, q, vec, recall_pool=None: [("t", i, 1.0 - i * 0.01) for i in range(25)],
    )
    monkeypatch.setattr(
        app, "_fetch_row_by_id", lambda c, t, r: {"row_id": r, "__row_text": f"text {r}", "a": r}
    )
    # rerank score = document index → candidate 24 scores highest
    monkeypatch.setattr(llm_mod, "rerank", lambda q, docs: [float(i) for i in range(len(docs))])
    rows, err = app.hybrid_query(None, ["t"], "q", top_n=5)
    assert err is None
    assert len(rows) == 5
    # candidate 24 (beyond the old pool of 20) now ranks first
    assert rows[0]["__row_id"] == 24


def test_hybrid_query_respects_top_n(monkeypatch):
    monkeypatch.setattr(llm_mod, "embed", lambda texts: [[0.1] * 1024])
    monkeypatch.setattr(
        search_mod,
        "hybrid_search",
        lambda c, tables, q, vec, recall_pool=None: [("t", i, 1.0 - i * 0.01) for i in range(10)],
    )
    monkeypatch.setattr(
        app, "_fetch_row_by_id", lambda c, t, r: {"row_id": r, "__row_text": f"text {r}", "a": r}
    )
    monkeypatch.setattr(llm_mod, "rerank", lambda q, docs: [float(i) for i in range(len(docs))])
    rows, err = app.hybrid_query(None, ["t"], "q", top_n=3)
    assert err is None
    assert len(rows) == 3


# ---- sql_query -------------------------------------------------------------


def test_sql_query_success():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (a INT, b TEXT)")
    conn.execute("INSERT INTO t VALUES (1, 'x')")
    cols, rows, err = app.sql_query(conn, "SELECT * FROM t")
    assert err is None
    assert rows == [{"a": 1, "b": "x"}]


def test_sql_query_error_surfaces():
    conn = sqlite3.connect(":memory:")
    cols, rows, err = app.sql_query(conn, "SELECT * FROM nope")
    assert err is not None and "SQL 执行出错" in err
    assert rows == []


# ---- _build_hybrid_rows / _row_display_json --------------------------------


def test_build_hybrid_rows_and_display_json():
    results = [("t1", 3, 1.5), ("t2", 1, 0.8)]

    def fetch(table, row_id):
        return {"row_id": row_id, "__row_text": f"full text for {table} {row_id}", "col": 1}

    rows = app._build_hybrid_rows(results, fetch)
    assert rows[0]["col"] == 1
    assert rows[0]["__table"] == "t1"
    assert rows[0]["__row_id"] == 3
    assert "表名" not in rows[0]
    assert "行号" not in rows[0]
    assert "分数" not in rows[0]
    assert "摘要" not in rows[0]
    assert app._row_display_json({"row_id": 1, "__row_text": "x", "a": 2}) == {"row_id": 1, "a": 2}


def test_build_hybrid_rows_skips_rows_with_no_visible_data():
    results = [("t1", 1, 1.5), ("t2", 2, 0.8), ("t3", 3, 0.5)]

    def fetch(table, row_id):
        if table == "t1":
            return {"row_id": row_id, "__row_text": "x", "name": "apple", "price": 10}
        if table == "t2":
            # All data columns empty/None → should be skipped (blank row).
            return {"row_id": row_id, "__row_text": "x", "name": None, "price": None, "sheet": "S", "src_row": 5}
        return {"row_id": row_id, "__row_text": "x", "name": "car", "price": 0}

    rows = app._build_hybrid_rows(results, fetch)
    assert [r["__table"] for r in rows] == ["t1", "t3"]
    # price 0 is a real value, not empty — row must be kept.
    assert rows[1]["price"] == 0


# ---- _fetch_row_by_id ------------------------------------------------------


def test_fetch_row_by_id_returns_matching_row():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE t (row_id INTEGER PRIMARY KEY, __row_text TEXT, a INT)")
    conn.execute("INSERT INTO t VALUES (1, 'text one', 10)")
    conn.execute("INSERT INTO t VALUES (3, 'text three', 30)")
    row = app._fetch_row_by_id(conn, "t", 3)
    assert row is not None
    assert row["row_id"] == 3
    assert row["a"] == 30
    assert app._fetch_row_by_id(conn, "t", 99) is None


# ---- preview ---------------------------------------------------------------


def test_preview_returns_first_n_rows_excluding_internal_cols():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    name = _uniq()
    conn.execute(f'CREATE TABLE "{name}" (row_id INTEGER PRIMARY KEY, __row_text TEXT, name TEXT, val INT)')
    for i in range(7):
        conn.execute(f'INSERT INTO "{name}" (name, val) VALUES (?, ?)', (f"r{i}", i))
    conn.commit()
    cols, rows = app.preview(conn, name, n=5)
    assert "row_id" not in cols
    assert "__row_text" not in cols
    assert len(rows) == 5
    assert rows[0]["name"] == "r0"
    assert "row_id" not in rows[0]
    assert "__row_text" not in rows[0]


def test_preview_empty_table_returns_empty():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    name = _uniq()
    conn.execute(f'CREATE TABLE "{name}" (row_id INTEGER PRIMARY KEY, __row_text TEXT, a INT)')
    conn.commit()
    cols, rows = app.preview(conn, name, n=5)
    assert cols == []
    assert rows == []


# ---- delete_table ----------------------------------------------------------


def test_delete_table_removes_table_and_vec():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    name = _uniq()
    conn.execute(f'CREATE TABLE "{name}" (row_id INTEGER PRIMARY KEY, a INT)')
    conn.execute(f'CREATE TABLE "vec_{name}" (row_id INTEGER PRIMARY KEY, vec TEXT)')
    conn.commit()
    app.delete_table(conn, name)
    assert name not in db_mod.list_tables(conn)


def test_delete_table_propagates_error(monkeypatch):
    def boom(c, n):
        raise RuntimeError("lock")

    monkeypatch.setattr(db_mod, "delete_table", boom)
    with pytest.raises(RuntimeError):
        app.delete_table(None, "x")


# ---- list_tables -----------------------------------------------------------


def test_list_tables_delegates(monkeypatch):
    dummy = types.SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(db_mod, "get_conn", lambda: dummy)
    monkeypatch.setattr(db_mod, "list_tables", lambda c: ["a", "b"])
    assert app.list_tables() == ["a", "b"]


# ---- ingest_file -----------------------------------------------------------


def test_ingest_file_returns_name_and_updated_and_wires_progress(monkeypatch):
    dummy = types.SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(db_mod, "get_conn", lambda: dummy)
    fracs = []

    def fake_ingest(c, p, on_progress=None, name=None, header_row=0):
        if on_progress:
            on_progress(0.5, "msg")
        return ("x", False)

    monkeypatch.setattr(ingest_mod, "ingest_file", fake_ingest)
    name, updated = app.ingest_file("/tmp/x.csv", on_progress=lambda f, m: fracs.append(f))
    assert name == "x"
    assert updated is False
    assert 0.5 in fracs


def test_app_ingest_file_forwards_header_row(monkeypatch):
    captured = {}

    def fake_ingest(c, p, on_progress=None, name=None, header_row=0):
        captured["header_row"] = header_row
        return ("x", False)

    monkeypatch.setattr(ingest_mod, "ingest_file", fake_ingest)
    app.ingest_file("/tmp/x.csv", header_row=3)
    assert captured["header_row"] == 3


# ---- _columns_for ----------------------------------------------------------


def test_columns_for_passthrough():
    assert app._columns_for(["a", "b"]) == ["a", "b"]


# ---- AppTest smoke ---------------------------------------------------------


def test_app_smoke_runs_without_exception():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH)
    at.run()
    assert not at.exception
    assert len(at.file_uploader) >= 1


def test_app_renders_result_dataframe_without_columns_kwarg():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH)
    at.session_state["result_rows"] = [
        {"客户": "张三", "金额": 200},
    ]
    at.run()
    assert not at.exception


# ---- rag_query -------------------------------------------------------------

def test_rag_query_success(monkeypatch):
    monkeypatch.setattr(llm_mod, "embed", lambda texts: [[0.1] * 1024])
    monkeypatch.setattr(search_mod, "hybrid_search", lambda c, tables, q, vec, recall_pool=None: [("t", 1, 0.9)])
    monkeypatch.setattr(app, "_fetch_row_by_id", lambda c, t, r: {"row_id": 1, "__row_text": "客户 张三 金额 200", "a": 1})
    monkeypatch.setattr(llm_mod, "rerank", lambda q, docs: [0.5] * len(docs))
    captured = {}

    def fake_answer(question, context, source=None):
        captured["context"] = context
        return "答案文本"

    monkeypatch.setattr(llm_mod, "answer", fake_answer)
    monkeypatch.setattr(llm_mod, "can_answer", lambda q, ctx: True)
    answer, rows, err = app.rag_query(None, ["t"], "q")
    assert err is None
    assert answer == "答案文本"
    assert rows and rows[0]["__table"] == "t"
    assert "客户 张三" in captured["context"]


def test_rag_query_rows_exist_but_cannot_answer_web_fallback(monkeypatch):
    # Rows are retrieved but the LLM judges they can't answer the question →
    # falls back to live web search.
    monkeypatch.setattr(llm_mod, "embed", lambda texts: [[0.1] * 1024])
    monkeypatch.setattr(search_mod, "hybrid_search", lambda c, tables, q, vec, recall_pool=None: [("t", 1, 0.9)])
    monkeypatch.setattr(app, "_fetch_row_by_id", lambda c, t, r: {"row_id": 1, "__row_text": "客户 张三 金额 200", "a": 1})
    monkeypatch.setattr(llm_mod, "rerank", lambda q, docs: [0.5] * len(docs))
    monkeypatch.setattr(llm_mod, "can_answer", lambda q, ctx: False)
    captured = {}

    def fake_answer(question, context, source=None):
        captured["source"] = source
        return "网络答案"

    monkeypatch.setattr(llm_mod, "answer", fake_answer)
    monkeypatch.setattr("websearch.search", lambda q, max_results=5: ("网络搜索结果文本", None))
    answer, rows, err = app.rag_query(None, ["t"], "q")
    assert err is None
    assert answer == "网络答案"
    assert rows == []
    assert captured["source"] == "网络搜索（AnySearch）"


def test_rag_query_no_results_web_fallback(monkeypatch):
    # No DB rows → falls back to live web search via AnySearch.
    monkeypatch.setattr(llm_mod, "embed", lambda texts: [[0.1] * 1024])
    monkeypatch.setattr(search_mod, "hybrid_search", lambda c, tables, q, vec, recall_pool=None: [])
    captured = {}

    def fake_answer(question, context, source=None):
        captured["source"] = source
        return "网络答案"

    monkeypatch.setattr(llm_mod, "answer", fake_answer)
    monkeypatch.setattr("websearch.search", lambda q, max_results=5: ("网络搜索结果文本", None))
    answer, rows, err = app.rag_query(None, ["t"], "q")
    assert err is None
    assert answer == "网络答案"
    assert rows == []
    assert captured["source"] == "网络搜索（AnySearch）"


def test_rag_query_no_results_web_search_fails(monkeypatch):
    monkeypatch.setattr(llm_mod, "embed", lambda texts: [[0.1] * 1024])
    monkeypatch.setattr(search_mod, "hybrid_search", lambda c, tables, q, vec, recall_pool=None: [])
    monkeypatch.setattr("websearch.search", lambda q, max_results=5: ("", "网络搜索失败：超时"))
    answer, rows, err = app.rag_query(None, ["t"], "q")
    assert answer == ""
    assert rows == []
    assert err is not None and "网络搜索" in err


def test_rag_query_success_passes_db_source(monkeypatch):
    monkeypatch.setattr(llm_mod, "embed", lambda texts: [[0.1] * 1024])
    monkeypatch.setattr(search_mod, "hybrid_search", lambda c, tables, q, vec, recall_pool=None: [("t", 1, 0.9)])
    monkeypatch.setattr(app, "_fetch_row_by_id", lambda c, t, r: {"row_id": 1, "__row_text": "客户 张三 金额 200", "a": 1})
    monkeypatch.setattr(llm_mod, "rerank", lambda q, docs: [0.5] * len(docs))
    captured = {}

    def fake_answer(question, context, source=None):
        captured["source"] = source
        return "答案文本"

    monkeypatch.setattr(llm_mod, "answer", fake_answer)
    monkeypatch.setattr(llm_mod, "can_answer", lambda q, ctx: True)
    answer, rows, err = app.rag_query(None, ["t"], "q")
    assert err is None
    assert answer == "答案文本"
    assert captured["source"] == "数据库表格：t"


def test_rag_query_error_surfaces(monkeypatch):
    monkeypatch.setattr(llm_mod, "embed", lambda texts: (_ for _ in ()).throw(RuntimeError("embed down")))
    answer, rows, err = app.rag_query(None, ["t"], "q")
    assert answer == ""
    assert rows == []
    assert err is not None and "RAG" in err


def test_rag_query_with_code_runs(monkeypatch):
    monkeypatch.setattr(llm_mod, "embed", lambda texts: [[0.1] * 1024])
    monkeypatch.setattr(search_mod, "hybrid_search", lambda c, tables, q, vec, recall_pool=None: [("t", 1, 0.9)])
    monkeypatch.setattr(app, "_rerank_results", lambda c, q, results, top_n=5: results[:top_n])
    monkeypatch.setattr(app, "_fetch_row_by_id", lambda c, t, r: {"row_id": 1, "__row_text": "x", "a": 1})
    monkeypatch.setattr(llm_mod, "generate_code", lambda q, preview: "result = df['a'].sum()")
    monkeypatch.setattr(llm_mod, "answer", lambda q, ctx: "答案是 1")
    answer, rows, code, code_result, err = app.rag_query_with_code(None, ["t"], "q")
    assert err is None
    assert answer == "答案是 1"
    assert code == "result = df['a'].sum()"
    assert code_result == "1"
    assert rows and rows[0]["a"] == 1


def test_rag_query_with_review_runs(monkeypatch):
    monkeypatch.setattr(llm_mod, "embed", lambda texts: [[0.1] * 1024])
    monkeypatch.setattr(search_mod, "hybrid_search", lambda c, tables, q, vec, recall_pool=None: [("t", 1, 0.9)])
    monkeypatch.setattr(app, "_rerank_results", lambda c, q, results, top_n=5: results[:top_n])
    monkeypatch.setattr(app, "_fetch_row_by_id", lambda c, t, r: {"row_id": 1, "__row_text": "客户 张三 金额 200", "a": 1})
    monkeypatch.setattr(llm_mod, "answer", lambda q, ctx: "张三的金额是200")
    monkeypatch.setattr(llm_mod, "review_answer", lambda q, ctx, ans: (True, "回答正确"))
    answer, rows, verdict, critique, err = app.rag_query_with_review(None, ["t"], "q")
    assert err is None
    assert answer == "张三的金额是200"
    assert verdict is True
    assert critique == "回答正确"
    assert rows and rows[0]["__table"] == "t"


def test_rag_query_decomposed_single_subquery_delegates(monkeypatch):
    # decompose returns a single element → falls back to plain rag_query.
    monkeypatch.setattr(db_mod, "get_schema", lambda c, t: {"table": t, "columns": [], "sample_rows": []})
    monkeypatch.setattr(llm_mod, "decompose_question", lambda q, s: [q])
    monkeypatch.setattr(llm_mod, "embed", lambda texts: [[0.1] * 1024])
    monkeypatch.setattr(search_mod, "hybrid_search", lambda c, tables, q, vec, recall_pool=None: [("t", 1, 0.9)])
    monkeypatch.setattr(app, "_rerank_results", lambda c, q, results, top_n=5: results[:top_n])
    monkeypatch.setattr(app, "_fetch_row_by_id", lambda c, t, r: {"row_id": 1, "__row_text": "x", "a": 1})
    monkeypatch.setattr(llm_mod, "answer", lambda q, ctx, source=None: "答案")
    answer, rows, err = app.rag_query_decomposed(None, ["t"], "q")
    assert err is None
    assert answer == "答案"
    assert rows and rows[0]["__table"] == "t"


def test_rag_query_decomposed_multi_subquery_synthesizes(monkeypatch):
    monkeypatch.setattr(db_mod, "get_schema", lambda c, t: {"table": t, "columns": [], "sample_rows": []})
    monkeypatch.setattr(llm_mod, "decompose_question", lambda q, s: ["子1", "子2"])
    monkeypatch.setattr(llm_mod, "embed", lambda texts: [[0.1] * 1024])
    monkeypatch.setattr(search_mod, "hybrid_search", lambda c, tables, q, vec, recall_pool=None: [("t", 1, 0.9)])
    monkeypatch.setattr(app, "_rerank_results", lambda c, q, results, top_n=5: results[:top_n])
    monkeypatch.setattr(app, "_fetch_row_by_id", lambda c, t, r: {"row_id": 1, "__row_text": "x", "a": 1})
    captured = {}

    def fake_answer(question, context, source=None):
        captured["ctx"] = context
        return "综合答案"

    monkeypatch.setattr(llm_mod, "answer", fake_answer)
    answer, rows, err = app.rag_query_decomposed(None, ["t"], "q")
    assert err is None
    assert answer == "综合答案"
    assert "子1" in captured["ctx"] and "子2" in captured["ctx"]
    assert rows and rows[0]["__table"] == "t"


def test_rag_query_dual_success(monkeypatch):
    monkeypatch.setattr(llm_mod, "embed", lambda texts: [[0.1] * 1024])
    monkeypatch.setattr(search_mod, "hybrid_search", lambda c, tables, q, vec, recall_pool=None: [("t", 1, 0.9)])
    monkeypatch.setattr(app, "_rerank_results", lambda c, q, results, top_n=5: results[:top_n])
    monkeypatch.setattr(app, "_fetch_row_by_id", lambda c, t, r: {"row_id": 1, "__row_text": "客户 张三 金额 200", "a": 1})
    monkeypatch.setattr(app, "ask_query", lambda c, s, q, max_attempts=2: ("SELECT 1", ["a"], [{"a": 1}], None))
    monkeypatch.setattr(llm_mod, "cross_validate", lambda q, sql_ctx, text_ctx: "交叉验证答案")
    answer, rows, sql, sql_ctx, err = app.rag_query_dual(None, ["t"], "q")
    assert err is None
    assert answer == "交叉验证答案"
    assert sql == "SELECT 1"
    assert "1" in sql_ctx
    assert rows and rows[0]["__table"] == "t"


def test_rag_query_dual_no_results(monkeypatch):
    monkeypatch.setattr(llm_mod, "embed", lambda texts: [[0.1] * 1024])
    monkeypatch.setattr(search_mod, "hybrid_search", lambda c, tables, q, vec, recall_pool=None: [])
    monkeypatch.setattr(app, "ask_query", lambda c, s, q, max_attempts=2: ("SELECT 1", [], [], None))
    answer, rows, sql, sql_ctx, err = app.rag_query_dual(None, ["t"], "q")
    assert answer == ""
    assert rows == []
    assert err is not None and "未找到" in err
