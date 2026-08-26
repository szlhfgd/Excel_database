import base64
import io
import json
import os
import sqlite3
import tempfile
import threading
import uuid
from typing import Any

import pandas as pd
from dash import Dash, dcc, html, dash_table, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc
from dotenv import load_dotenv

load_dotenv()

import db
import ingest
import llm
import search


app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="电子表格数据库",
)

app.layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row(
            [
                dbc.Col(
                    width=3,
                    children=[
                        html.H4("📊 电子表格数据库"),
                        dcc.Upload(
                            id="upload-data",
                            children=dbc.Button("上传 Excel / CSV", color="primary"),
                            multiple=False,
                        ),
                        dbc.Spinner(
                            id="import-spinner",
                            children=[html.Div(id="import-status", className="text-muted small mt-2")],
                        ),
                        dbc.Progress(
                            id="import-progress",
                            value=0,
                            striped=True,
                            animated=True,
                            className="mt-2",
                            style={"display": "none"},
                        ),
                        dbc.Alert(id="import-error", color="danger", is_open=False, className="mt-2"),
                        html.Hr(),
                        dbc.Label("选择参与搜索的表"),
                        dbc.Checklist(id="table-select", options=[], value=[]),
                        html.Hr(),
                        dbc.Label("删除表"),
                        dbc.Select(
                            id="delete-table",
                            options=[],
                            value=None,
                            placeholder="选择要删除的表",
                        ),
                        dbc.Button("删除选中表", id="delete-btn", color="danger", className="mt-2"),
                    ],
                ),
                dbc.Col(
                    width=9,
                    children=[
                        html.H4("🔍 查询"),
                        dbc.RadioItems(
                            id="mode-radio",
                            options=[
                                {"label": "hybrid（语义+BM25）", "value": "hybrid"},
                                {"label": "ask（自然语言转SQL）", "value": "ask"},
                                {"label": "sql（手写SQL）", "value": "sql"},
                            ],
                            value="hybrid",
                            inline=True,
                        ),
                        html.Hr(),
                        dbc.Collapse(
                            id="sql-collapse",
                            is_open=False,
                            children=[
                                dbc.Label("手写 SQL"),
                                dbc.Textarea(id="sql-input", placeholder="SELECT * FROM 表名", rows=3),
                                dbc.Button("执行", id="sql-run", color="primary", className="mt-2"),
                            ],
                        ),
                        dbc.Collapse(
                            id="ask-collapse",
                            is_open=False,
                            children=[
                                dbc.Label("用自然语言提问"),
                                dbc.Textarea(id="ask-input", placeholder="例如：找出金额大于 100 的记录", rows=3),
                                dbc.Button("提问", id="ask-run", color="primary", className="mt-2"),
                                dcc.Markdown(id="ask-sql-block", className="mt-2 border rounded p-2 bg-light"),
                            ],
                        ),
                        dbc.Collapse(
                            id="hybrid-collapse",
                            is_open=False,
                            children=[
                                dbc.Label("关键词 / 短语搜索"),
                                dbc.Input(id="hybrid-input", placeholder="输入要搜索的内容"),
                                dbc.Button("搜索", id="hybrid-run", color="primary", className="mt-2"),
                            ],
                        ),
                        dbc.Alert(id="empty-hint", color="info", is_open=False, className="mt-3"),
                        dbc.Alert(id="error-alert", color="danger", is_open=False, className="mt-3"),
                        dbc.Alert(id="empty-result", color="info", is_open=False, className="mt-3"),
                        dbc.Spinner(
                            children=[dash_table.DataTable(id="result-table", data=[], columns=[])],
                        ),
                        dbc.Button("下载 CSV", id="download-btn", color="secondary", className="mt-2"),
                        dcc.Download(id="download-csv"),
                        dbc.Collapse(
                            id="detail-collapse",
                            is_open=False,
                            children=[html.Pre(id="detail-pre", className="mt-3 border rounded p-2 bg-light")],
                        ),
                    ],
                ),
            ]
        ),
        dcc.Store(id="selected-tables", data=[]),
        dcc.Store(id="import-job", data=None),
        dcc.Interval(id="init-interval", n_intervals=0, max_intervals=1),
        dcc.Interval(id="import-interval", interval=400, n_intervals=0),
    ],
)


_ingest_jobs: dict[str, dict] = {}


def _set_progress(job_id: str, frac: float, msg: str):
    job = _ingest_jobs.get(job_id)
    if job:
        job["progress"] = int(round(frac * 100))
        job["status"] = msg


def _finish_job(job_id: str, table: str | None = None, error: str | None = None):
    job = _ingest_jobs.get(job_id)
    if job:
        job.update(done=True, table=table, error=error)


def _run_ingest(job_id: str, tmp_path: str, filename: str):
    try:
        conn = db.get_conn()
        try:
            name = ingest.ingest_file(
                conn, tmp_path, on_progress=lambda f, m: _set_progress(job_id, f, m)
            )
        finally:
            conn.close()
        _finish_job(job_id, table=name)
    except Exception as e:  # noqa: BLE001 - reported to UI via job store
        _finish_job(job_id, error=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _start_ingest(contents: str, filename: str) -> str:
    _, b64 = contents.split(",", 1)
    raw = base64.b64decode(b64)
    suffix = os.path.splitext(filename)[1].lower()
    tmp_path = tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name
    with open(tmp_path, "wb") as f:
        f.write(raw)
    job_id = str(uuid.uuid4())
    _ingest_jobs[job_id] = {
        "progress": 0,
        "status": "准备导入…",
        "done": False,
        "error": None,
        "table": None,
    }
    threading.Thread(target=_run_ingest, args=(job_id, tmp_path, filename), daemon=True).start()
    return job_id


def _current_tables():
    conn = db.get_conn()
    try:
        tables = db.list_tables(conn)
    finally:
        conn.close()
    options = [{"label": t, "value": t} for t in tables]
    return options, tables, options


@app.callback(
    Output("table-select", "options"),
    Output("table-select", "value"),
    Output("delete-table", "options"),
    Output("import-status", "children"),
    Output("import-error", "children"),
    Output("import-error", "is_open"),
    Output("import-progress", "value"),
    Output("import-progress", "children"),
    Output("import-progress", "style"),
    Output("import-job", "data"),
    Input("upload-data", "contents"),
    Input("upload-data", "filename"),
    Input("delete-btn", "n_clicks"),
    Input("init-interval", "n_intervals"),
    Input("import-interval", "n_intervals"),
    State("delete-table", "value"),
    State("import-job", "data"),
)
def refresh_tables(contents, filename, del_clicks, init_n, import_n, del_value, import_job):
    triggered = ctx.triggered_id
    status = no_update
    error_children = no_update
    error_open = no_update
    progress_val = no_update
    progress_children = no_update
    progress_style = no_update
    new_job = import_job
    value = no_update
    options = no_update
    delete_options = no_update

    if triggered == "upload-data" and contents:
        new_job = _start_ingest(contents, filename)
        status = "正在导入…"
        progress_val = 0
        progress_children = "0%"
        progress_style = {"display": "block"}
    elif triggered == "import-interval" and import_job:
        job = _ingest_jobs.get(import_job)
        if job:
            if not job["done"]:
                status = job["status"]
                progress_val = job["progress"]
                progress_children = f"{job['progress']}%"
                progress_style = {"display": "block"}
            else:
                if job["error"]:
                    error_children = f"导入失败：{job['error']}"
                    error_open = True
                    status = "导入失败"
                else:
                    status = f"已导入表：{job['table']}"
                    progress_val = 100
                    progress_children = "100%"
                    error_children = ""
                    error_open = False
                progress_style = {"display": "none"}
                new_job = None
                options, value, delete_options = _current_tables()
    elif triggered == "delete-btn" and del_clicks and del_value:
        try:
            conn = db.get_conn()
            try:
                db.delete_table(conn, del_value)
            finally:
                conn.close()
            status = f"已删除表：{del_value}"
        except Exception as e:
            status = f"删除失败：{e}"
        options, value, delete_options = _current_tables()
    elif triggered == "init-interval":
        options, value, delete_options = _current_tables()

    return (
        options,
        value,
        delete_options,
        status,
        error_children,
        error_open,
        progress_val,
        progress_children,
        progress_style,
        new_job,
    )


@app.callback(
    Output("selected-tables", "data"),
    Input("table-select", "value"),
)
def update_selected_store(value):
    return value or []


def _to_csv(records: Any) -> bytes:
    buf = io.StringIO()
    pd.DataFrame(records).to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue().encode("utf-8-sig")


def _run_query(conn: sqlite3.Connection, sql: str):
    cur = conn.execute(sql)
    if cur.description is None:
        return [], []
    columns = [d[0] for d in cur.description]
    rows = [dict(zip(columns, r)) for r in cur.fetchall()]
    return columns, rows


def _ask(conn: sqlite3.Connection, selected: list[str], question: str, max_attempts: int = 2):
    schemas = [db.get_schema(conn, t) for t in selected]
    last_error: str | None = None
    sql = ""
    for _ in range(max_attempts):
        sql = llm.generate_sql(schemas, question, prev_error=last_error)
        try:
            columns, rows = _run_query(conn, sql)
            return sql, columns, rows
        except Exception as e:  # noqa: BLE001 - surface to user after retries
            last_error = str(e)
    raise RuntimeError(f"NL2SQL 生成或执行失败：{last_error}")


def _build_hybrid_rows(results: list[tuple[str, int, float]], fetch_row):
    out = []
    for table, row_id, score in results:
        row = fetch_row(table, row_id)
        if not row:
            continue
        text = row.get("__row_text") or ""
        out.append(
            {
                "表名": table,
                "行号": row_id,
                "分数": round(score, 4),
                "摘要": text[:80],
                "__table": table,
                "__row_id": row_id,
            }
        )
    return out


def _row_display_json(row: dict) -> dict:
    return {k: v for k, v in row.items() if k != "__row_text"}


def _fetch_row_by_id(conn: sqlite3.Connection, table: str, row_id: int):
    rows = db.get_rows(conn, table)
    return next((r for r in rows if r.get("row_id") == row_id), None)


def _columns_for(headers: list[str]):
    return [{"name": h, "id": h} for h in headers]


@app.callback(
    Output("sql-collapse", "is_open"),
    Output("ask-collapse", "is_open"),
    Output("hybrid-collapse", "is_open"),
    Input("mode-radio", "value"),
)
def toggle_mode(mode: str):
    return mode == "sql", mode == "ask", mode == "hybrid"


@app.callback(
    Output("empty-hint", "children"),
    Output("empty-hint", "is_open"),
    Input("mode-radio", "value"),
    Input("selected-tables", "data"),
    Input("table-select", "options"),
)
def update_empty_hint(mode: str, selected, options):
    if not options:
        return "请先上传 Excel / CSV 表格。", True
    if mode in ("hybrid", "ask") and not selected:
        return "请先在左侧勾选至少一个参与查询的表。", True
    return "", False


@app.callback(
    Output("result-table", "data"),
    Output("result-table", "columns"),
    Output("error-alert", "children"),
    Output("error-alert", "is_open"),
    Output("empty-result", "children"),
    Output("empty-result", "is_open"),
    Input("sql-run", "n_clicks"),
    State("sql-input", "value"),
    prevent_initial_call=True,
)
def run_sql(n_clicks, sql):
    if not sql or not sql.strip():
        return [], [], "", False, "", False
    conn = db.get_conn()
    try:
        columns, rows = _run_query(conn, sql)
    except Exception as e:  # noqa: BLE001
        return [], [], f"SQL 执行出错：{e}", True, "", False
    finally:
        conn.close()
    empty = ("未找到匹配的行。", True) if not rows else ("", False)
    return rows, _columns_for(columns), "", False, empty[0], empty[1]


@app.callback(
    Output("ask-sql-block", "children"),
    Output("result-table", "data"),
    Output("result-table", "columns"),
    Output("error-alert", "children"),
    Output("error-alert", "is_open"),
    Output("empty-result", "children"),
    Output("empty-result", "is_open"),
    Input("ask-run", "n_clicks"),
    State("ask-input", "value"),
    State("selected-tables", "data"),
    prevent_initial_call=True,
)
def run_ask(n_clicks, question, selected):
    if not question or not question.strip():
        return "", [], [], "", False, "", False
    if not selected:
        return "", [], [], "请先勾选至少一个表。", True, "", False
    conn = db.get_conn()
    try:
        sql, columns, rows = _ask(conn, selected, question)
    except Exception as e:  # noqa: BLE001
        return "", [], [], f"查询出错：{e}", True, "", False
    finally:
        conn.close()
    empty = ("未找到匹配的行。", True) if not rows else ("", False)
    return f"```sql\n{sql}\n```", rows, _columns_for(columns), "", False, empty[0], empty[1]


@app.callback(
    Output("result-table", "data"),
    Output("result-table", "columns"),
    Output("error-alert", "children"),
    Output("error-alert", "is_open"),
    Output("empty-result", "children"),
    Output("empty-result", "is_open"),
    Input("hybrid-run", "n_clicks"),
    State("hybrid-input", "value"),
    State("selected-tables", "data"),
    prevent_initial_call=True,
)
def run_hybrid(n_clicks, query, selected):
    if not query or not query.strip():
        return [], [], "", False, "", False
    if not selected:
        return [], [], "请先勾选至少一个表。", True, "", False
    conn = db.get_conn()
    try:
        vec = llm.embed([query])[0]
        results = search.hybrid_search(conn, selected, query, vec, k=None)
        rows = _build_hybrid_rows(results, lambda t, r: _fetch_row_by_id(conn, t, r))
    except Exception as e:  # noqa: BLE001
        return [], [], f"搜索出错：{e}", True, "", False
    finally:
        conn.close()
    empty = ("未找到匹配的搜索结果。", True) if not rows else ("", False)
    return rows, _columns_for(["表名", "行号", "分数", "摘要"]), "", False, empty[0], empty[1]


@app.callback(
    Output("detail-collapse", "is_open"),
    Output("detail-pre", "children"),
    Input("result-table", "active_cell"),
    State("result-table", "data"),
    prevent_initial_call=True,
)
def show_detail(active_cell, data):
    if not active_cell or not data:
        return False, ""
    row = data[active_cell["row"]]
    table = row.get("__table")
    rid = row.get("__row_id")
    if table is None or rid is None:
        return False, ""
    conn = db.get_conn()
    try:
        rows = db.get_rows(conn, table)
    finally:
        conn.close()
    full = next((r for r in rows if r.get("row_id") == rid), None)
    if full is None:
        return False, ""
    return True, json.dumps(_row_display_json(full), ensure_ascii=False, indent=2, default=str)


@app.callback(
    Output("download-csv", "data"),
    Input("download-btn", "n_clicks"),
    State("result-table", "data"),
    prevent_initial_call=True,
)
def download_csv(n_clicks, data):
    if not data:
        return None
    return dcc.send_bytes(_to_csv(data), "result.csv")


if __name__ == "__main__":
    app.run(debug=True, port=8050)
