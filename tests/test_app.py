import json
import sqlite3

import pytest

import app
from dash import Dash


def _collect_ids(component):
    ids = []
    cid = getattr(component, "id", None)
    if cid is not None:
        ids.append(cid)
    children = getattr(component, "children", None)
    if isinstance(children, list):
        for child in children:
            ids.extend(_collect_ids(child))
    elif children is not None:
        ids.extend(_collect_ids(children))
    return ids


def test_app_is_dash_instance():
    assert isinstance(app.app, Dash)


def test_layout_has_key_ids():
    ids = _collect_ids(app.app.layout)
    for expected in ("upload-data", "table-select", "mode-radio", "result-table", "download-csv"):
        assert expected in ids, f"missing id: {expected}"


def test_ticket02_table_management_ids_present():
    ids = _collect_ids(app.app.layout)
    for expected in ("selected-tables", "delete-table", "delete-btn", "import-spinner", "init-interval"):
        assert expected in ids, f"missing id: {expected}"


def test_ticket02_table_management_callbacks_registered():
    cm = app.app.callback_map
    joined = " ".join(cm.keys())
    assert "table-select.options" in joined, "table list refresh callback missing"
    assert "selected-tables.data" in joined, "selected tables store callback missing"
    assert "delete-table.options" in joined, "delete dropdown refresh callback missing"


def test_to_csv_uses_utf8sig_bom():
    out = app._to_csv([{"姓名": "张三", "年龄": 30}])
    assert out[:3] == b"\xef\xbb\xbf"
    text = out.decode("utf-8-sig")
    assert "姓名,年龄" in text
    assert "张三,30" in text


def test_run_query_returns_columns_and_rows():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (a INT, b TEXT)")
    conn.execute("INSERT INTO t VALUES (1, 'x')")
    cols, rows = app._run_query(conn, "SELECT * FROM t")
    assert cols == ["a", "b"]
    assert rows == [{"a": 1, "b": "x"}]


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

    monkeypatch.setattr(app.llm, "generate_sql", fake_gen)
    monkeypatch.setattr(app, "_run_query", fake_run)
    monkeypatch.setattr(app.db, "get_schema", lambda c, t: {"table": t, "columns": [], "sample_rows": []})

    sql, cols, rows = app._ask(None, ["t"], "q")
    assert sql == "SELECT * FROM t"
    assert calls["gen"] == 2
    assert calls["run"] == 2
    assert rows == [{"a": 1, "b": 2}]


def test_ask_raises_after_max_retries(monkeypatch):
    monkeypatch.setattr(app.llm, "generate_sql", lambda s, q, prev_error=None: "BAD")
    monkeypatch.setattr(app, "_run_query", lambda c, sql: (_ for _ in ()).throw(sqlite3.OperationalError("err")))
    monkeypatch.setattr(app.db, "get_schema", lambda c, t: {"table": t, "columns": [], "sample_rows": []})

    with pytest.raises(RuntimeError):
        app._ask(None, ["t"], "q")


def test_build_hybrid_rows_and_display_json():
    results = [("t1", 3, 1.5), ("t2", 1, 0.8)]

    def fetch(table, row_id):
        return {"row_id": row_id, "__row_text": f"full text for {table} {row_id}", "col": 1}

    rows = app._build_hybrid_rows(results, fetch)
    assert rows[0]["表名"] == "t1"
    assert rows[0]["行号"] == 3
    assert rows[0]["分数"] == 1.5
    assert "full text" in rows[0]["摘要"]
    assert rows[0]["__table"] == "t1"
    assert rows[0]["__row_id"] == 3
    assert app._row_display_json({"row_id": 1, "__row_text": "x", "a": 2}) == {"row_id": 1, "a": 2}


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


def test_query_mode_layout_ids():
    ids = _collect_ids(app.app.layout)
    for expected in (
        "sql-collapse",
        "ask-collapse",
        "hybrid-collapse",
        "sql-input",
        "ask-input",
        "hybrid-input",
        "sql-run",
        "ask-run",
        "hybrid-run",
        "error-alert",
        "empty-hint",
        "empty-result",
        "ask-sql-block",
        "download-btn",
        "detail-pre",
        "detail-collapse",
    ):
        assert expected in ids, f"missing id: {expected}"


def test_query_callbacks_registered():
    cm = app.app.callback_map
    joined = " ".join(cm.keys())
    for out in (
        "result-table.data",
        "ask-sql-block.children",
        "error-alert.children",
        "empty-result.children",
        "download-csv.data",
        "detail-pre.children",
        "empty-hint.children",
        "sql-collapse.is_open",
    ):
        assert out in joined, f"missing callback output: {out}"


def test_mode_toggle_opens_only_selected_mode():
    assert app.toggle_mode("sql") == (True, False, False)
    assert app.toggle_mode("ask") == (False, True, False)
    assert app.toggle_mode("hybrid") == (False, False, True)


def test_empty_hint_messages():
    assert app.update_empty_hint("sql", [], []) == ("请先上传 Excel / CSV 表格。", True)
    assert app.update_empty_hint("ask", [], [{"label": "t", "value": "t"}]) == (
        "请先在左侧勾选至少一个参与查询的表。",
        True,
    )
    assert app.update_empty_hint("sql", [], [{"label": "t", "value": "t"}]) == ("", False)


def test_llm_client_has_timeout_and_no_retries(monkeypatch):
    import llm as llm_mod

    captured = {}

    def fake(*args, **kwargs):
        captured.update(kwargs)
        return type("C", (), {})()

    monkeypatch.setattr(llm_mod, "OpenAI", fake)
    monkeypatch.setenv("LLM_TIMEOUT", "7")

    llm_mod._client()
    assert captured.get("timeout") == 7.0
    assert captured.get("max_retries") == 0


def test_import_progress_ids_present():
    ids = _collect_ids(app.app.layout)
    for expected in ("import-progress", "import-error", "import-job", "import-interval"):
        assert expected in ids, f"missing id: {expected}"


def test_run_ingest_records_failure(monkeypatch):
    import types

    monkeypatch.setattr(app.db, "get_conn", lambda: types.SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(
        app.ingest,
        "ingest_file",
        lambda c, p, on_progress=None: (_ for _ in ()).throw(RuntimeError("embed api down")),
    )
    job_id = "fail1"
    app._ingest_jobs[job_id] = {"progress": 0, "status": "", "done": False, "error": None, "table": None}
    app._run_ingest(job_id, "fake.csv", "t.csv")
    job = app._ingest_jobs[job_id]
    assert job["done"] is True
    assert job["error"] == "embed api down"
    assert job["table"] is None


def test_run_ingest_records_success(monkeypatch):
    import types

    monkeypatch.setattr(app.db, "get_conn", lambda: types.SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(app.ingest, "ingest_file", lambda c, p, on_progress=None: "mytable")
    job_id = "ok1"
    app._ingest_jobs[job_id] = {"progress": 0, "status": "", "done": False, "error": None, "table": None}
    app._run_ingest(job_id, "fake.csv", "t.csv")
    job = app._ingest_jobs[job_id]
    assert job["done"] is True
    assert job["table"] == "mytable"
    assert job["error"] is None


def test_refresh_tables_poll_surfaces_import_failure(monkeypatch):
    import types

    job_id = "fail1"
    app._ingest_jobs[job_id] = {
        "progress": 40,
        "status": "生成向量 4/10",
        "done": True,
        "error": "embed api down",
        "table": None,
    }
    app.ctx = types.SimpleNamespace(triggered_id="import-interval")
    monkeypatch.setattr(app.db, "list_tables", lambda c: [])
    out = app.refresh_tables(
        contents=None,
        filename=None,
        del_clicks=None,
        init_n=0,
        import_n=1,
        del_value=None,
        import_job=job_id,
    )
    assert out[4] == "导入失败：embed api down"
    assert out[5] is True
    assert out[9] is None


def test_refresh_tables_poll_surfaces_import_success(monkeypatch):
    import types

    job_id = "ok1"
    app._ingest_jobs[job_id] = {
        "progress": 100,
        "status": "导入完成",
        "done": True,
        "error": None,
        "table": "mytable",
    }
    app.ctx = types.SimpleNamespace(triggered_id="import-interval")
    monkeypatch.setattr(app.db, "list_tables", lambda c: ["mytable"])
    out = app.refresh_tables(
        contents=None,
        filename=None,
        del_clicks=None,
        init_n=0,
        import_n=1,
        del_value=None,
        import_job=job_id,
    )
    assert "已导入表：mytable" in out[3]
    assert out[5] is False
    assert out[0] == [{"label": "mytable", "value": "mytable"}]
    assert out[9] is None


def test_ingest_file_emits_progress(monkeypatch):
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2]})
    monkeypatch.setattr(app.ingest, "read_file", lambda p: df)
    monkeypatch.setattr(app.ingest, "clean_df", lambda d: d)
    monkeypatch.setattr(app.db, "create_table_from_df", lambda c, n, d, t: "t1")
    monkeypatch.setattr(app.ingest, "build_embeddings", lambda c, n, on_progress=None: None)

    fracs = []
    name = app.ingest.ingest_file(None, "x.csv", on_progress=lambda f, m: fracs.append(f))
    assert name == "t1"
    assert fracs[0] <= 0.10
    assert fracs[-1] == 1.0
