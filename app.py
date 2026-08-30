"""电子表格数据库 — Streamlit single-page app.

Reuses the existing backend modules (db, ingest, search, llm).
All ``st.*`` calls live inside :func:`main`; module-level helpers are
framework-free so tests can import them directly.
"""

from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
import tempfile
from typing import Any

# ---------------------------------------------------------------------------
# Framework-agnostic helpers (no ``st.*`` inside)
# ---------------------------------------------------------------------------


def list_tables() -> list[str]:
    """Return user-facing table names (excludes ``vec_`` tables)."""
    import db as _db

    conn = _db.get_conn()
    try:
        return _db.list_tables(conn)
    finally:
        conn.close()


def delete_table(conn: sqlite3.Connection, name: str) -> None:
    """Delete *name* and its vector table."""
    import db as _db

    _db.delete_table(conn, name)


def preview(conn: sqlite3.Connection, table: str, n: int = 5) -> tuple[list[str], list[dict]]:
    """Return (columns, rows) for the first *n* rows of *table*."""
    import db as _db

    return _db.get_preview(conn, table, n=n)


def ingest_file(path: str, on_progress=None, name: str | None = None, header_row: int = 1, key_col: str | None = None, mode: str = "replace") -> tuple[str, bool]:
    """Open a fresh connection, ingest *path*, close, return ``(name, updated)``."""
    import db as _db
    import ingest as _ingest

    conn = _db.get_conn()
    try:
        kwargs: dict[str, Any] = dict(on_progress=on_progress, name=name, header_row=header_row)
        if key_col is not None:
            kwargs["key_col"] = key_col
        if mode != "replace":
            kwargs["mode"] = mode
        name, updated = _ingest.ingest_file(conn, path, **kwargs)
    finally:
        conn.close()
    return name, updated


def _to_csv(records: Any) -> bytes:
    """Serialise a list-of-dict (or list-of-list) to CSV bytes."""
    buf = io.StringIO()
    if not records:
        return b""
    if isinstance(records[0], dict):
        pd = __import__("pandas")
        pd.DataFrame(records).to_csv(buf, index=False, encoding="utf-8-sig")
    else:
        writer = csv.writer(buf)
        writer.writerows(records)
    return buf.getvalue().encode("utf-8-sig")


# ---- internal helpers ----------------------------------------------------


def _run_query(conn: sqlite3.Connection, sql: str) -> tuple[list[str], list[dict]]:
    """Execute *sql* and return ``(columns, rows_as_dicts)``."""
    cur = conn.execute(sql)
    if cur.description is None:
        return [], []
    columns = [d[0] for d in cur.description]
    rows = [dict(zip(columns, r)) for r in cur.fetchall()]
    return columns, rows


def _ask(
    conn: sqlite3.Connection,
    selected: list[str],
    question: str,
    max_attempts: int = 2,
) -> tuple[str, list[str], list[dict] | None, str | None]:
    """NL2SQL loop. Returns ``(sql, columns, rows | None, error | None)``."""
    import db as _db
    import llm as _llm

    schemas = [_db.get_schema(conn, t) for t in selected]
    last_error: str | None = None
    sql = ""
    for _ in range(max_attempts):
        sql = _llm.generate_sql(schemas, question, prev_error=last_error)
        try:
            columns, rows = _run_query(conn, sql)
            return sql, columns, rows, None
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
    return sql, [], None, f"NL2SQL 生成或执行失败：{last_error}"


def ask_query(
    conn: sqlite3.Connection,
    selected: list[str],
    question: str,
    max_attempts: int = 2,
) -> tuple[str, list[str], list[dict], str | None]:
    """Public wrapper — returns ``(sql, columns, rows, error | None)``."""
    sql, columns, rows, err = _ask(conn, selected, question, max_attempts)
    return sql, columns, rows or [], err


def _fetch_row_by_id(conn: sqlite3.Connection, table: str, row_id: int) -> dict | None:
    import db as _db

    rows = _db.get_rows(conn, table)
    return next((r for r in rows if r.get("row_id") == row_id), None)


def _build_hybrid_rows(
    results: list[tuple[str, int, float]],
    fetch_row,
) -> list[dict]:
    out = []
    for table, row_id, score in results:
        row = fetch_row(table, row_id)
        if not row:
            continue
        # Skip rows with no visible data (all data columns empty/None) — these
        # would otherwise render as blank rows in the middle of the results.
        data_cols = [k for k in row if k not in ("row_id", "__row_text", "sheet", "src_row")]
        if not any(row.get(c) not in (None, "") for c in data_cols):
            continue
        # Show the original row content (whole row); keep __table/__row_id
        # only as hidden keys for the detail view.
        data = {k: v for k, v in row.items() if k not in ("row_id", "__row_text")}
        data["__table"] = table
        data["__row_id"] = row_id
        # Relevance score for display (normalized to 0..1 by hybrid_query; raw
        # rerank score for RAG callers). Surfaced as the "相关度" column.
        data["相关度"] = round(score, 3)
        out.append(data)
    return out


def _build_source_label(source_rows: list[dict]) -> str:
    """Build a table→rows source citation, e.g.

    '数据库表格：销售记录（行 3、7、12）、客户表（行 5）'.

    Groups retrieved rows by table and lists the specific row ids that fed the
    answer, so the citation points at concrete evidence instead of whole tables.
    """
    from collections import OrderedDict

    by_table: "OrderedDict[str, list]" = OrderedDict()
    for r in source_rows:
        t = r.get("__table")
        rid = r.get("__row_id")
        if t is None or rid is None:
            continue
        by_table.setdefault(t, []).append(rid)
    if not by_table:
        return ""
    parts = []
    for t, rids in by_table.items():
        uniq = sorted(set(rids))
        parts.append(f"{t}（行 {('、'.join(str(x) for x in uniq))}）")
    return "数据库表格：" + "、".join(parts)


def _row_has_visible_data(row: dict) -> bool:
    """True if *row* has at least one non-empty visible (non-internal) value.

    Used by the results view to hide fully-empty rows. Mirrors the emptiness
    check in :func:`_build_hybrid_rows` but also covers raw SQL/NL2SQL rows.
    """
    for k, v in row.items():
        if k.startswith("__") or k in ("row_id", "__row_text", "sheet", "src_row"):
            continue
        if v not in (None, ""):
            return True
    return False


def _row_display_json(row: dict) -> dict:
    return {k: v for k, v in row.items() if k != "__row_text"}


def _columns_for(headers: list[str]) -> list[str]:
    return list(headers)


# Rerank re-scores candidates with a cross-encoder in a single API call, so we
# can afford to rerank every RRF candidate instead of a small pool. RERANK_POOL_CAP
# is a safety limit for very large tables that might exceed the rerank API's
# per-request document limit; typical spreadsheets are well under it.
RERANK_POOL_CAP = 200


def _rerank_results(conn, query, results, pool: int | None = None, top_n: int = 5):
    """Re-rank RRF-fused *results* and return the top *top_n* as (table, row_id, score).

    Reranks up to ``RERANK_POOL_CAP`` RRF candidates (all of them for typical
    tables) with the cross-encoder reranker, then returns the highest-scoring
    *top_n*. If the reranker is unavailable the RRF order is kept as a fallback.
    """
    import llm as _llm

    if not results:
        return []
    cap = RERANK_POOL_CAP if pool is None else pool
    candidates = results[:cap]
    docs = []
    for table, row_id, _ in candidates:
        row = _fetch_row_by_id(conn, table, row_id)
        if row:
            docs.append(row.get("__row_text") or json.dumps(_row_display_json(row), ensure_ascii=False))
        else:
            docs.append("")
    try:
        scores = _llm.rerank(query, docs)
    except Exception:  # noqa: BLE001 - rerank unavailable → keep RRF order
        return candidates[:top_n]
    ordered = [c for _, c in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)]
    return ordered[:top_n]


def hybrid_query(
    conn: sqlite3.Connection,
    selected: list[str],
    query: str,
    score_floor_frac: float = 0.0,
    view_mode: str = "rerank",   # "rerank" | "fusion"
    top_n: int = 20,
    recall_pool: int = 50,
    min_results: int = 0,
) -> tuple[list[dict], str | None]:
    """Hybrid search: BM25 + embedding → RRF fusion → rerank → top *top_n*.

    ``score_floor_frac`` is a 0..1 fraction of the max score (works for both
    rerank and fusion views). ``view_mode`` selects reranked vs raw-fused order.
    ``min_results`` is a safety floor: when enough candidates exist, the
    relevance threshold will not reduce results below this many rows (still
    capped by *top_n*). Returns ``(rows, error | None)``.
    """
    import llm as _llm
    import search as _search

    try:
        vec = _llm.embed([query])[0]
        fused = _search.hybrid_search(conn, selected, query, vec, recall_pool=recall_pool)
        reranked = _rerank_results(conn, query, fused, top_n=recall_pool)
        chosen = reranked if view_mode == "rerank" else fused
        if chosen:
            max_score = max(r[2] for r in chosen) or 1.0
            floor = score_floor_frac * max_score
            filtered = [r for r in chosen if r[2] >= floor]
        else:
            max_score = 1.0
            filtered = []
        top = filtered[:top_n]
        # Safety floor: never let the threshold drop results below min_results
        # when enough candidates exist (still capped by top_n).
        if min_results and len(top) < min_results and len(chosen) >= min_results:
            top = chosen[:min_results][:top_n]
        # Normalize scores to 0..1 (same scale as the 相关度阈值 slider) for display.
        top_norm = [(t, r, round(s / max_score, 3)) for (t, r, s) in top]
        rows = _build_hybrid_rows(top_norm, lambda t, r: _fetch_row_by_id(conn, t, r))
        return rows, None
    except Exception as exc:  # noqa: BLE001
        return [], f"搜索出错：{exc}"


def sql_query(
    conn: sqlite3.Connection,
    sql: str,
) -> tuple[list[str], list[dict], str | None]:
    """Run raw SQL. Returns ``(columns, rows, error | None)``."""
    try:
        columns, rows = _run_query(conn, sql)
        return columns, rows, None
    except Exception as exc:  # noqa: BLE001
        return [], [], f"SQL 执行出错：{exc}"


def rag_query(
    conn: sqlite3.Connection,
    selected: list[str],
    question: str,
    recall_pool: int = 50,
    top_n: int = 5,
) -> tuple[str, list[dict], str | None]:
    """RAG Q&A. Retrieve relevant rows via hybrid search + rerank, then ask the
    LLM to answer in natural language grounded in those rows.
    Returns ``(answer, source_rows, error | None)``."""
    import llm as _llm
    import search as _search

    try:
        vec = _llm.embed([question])[0]
        results = _search.hybrid_search(conn, selected, question, vec, recall_pool=recall_pool)
        top = _rerank_results(conn, question, results, top_n=top_n)
        source_rows = _build_hybrid_rows(top, lambda t, r: _fetch_row_by_id(conn, t, r))
        context_parts: list[str] = []
        for row in source_rows:
            full = _fetch_row_by_id(conn, row["__table"], row["__row_id"])
            if full:
                context_parts.append(full.get("__row_text") or json.dumps(_row_display_json(full), ensure_ascii=False))
        if not source_rows:
            # No relevant rows in the database → fall back to live web search
            # via AnySearch so the user still gets an answer.
            import websearch as _web

            web_text, web_err = _web.search(question, max_results=5)
            if web_err:
                return "", [], web_err
            if not web_text:
                return "", [], "未找到相关的数据行，且网络搜索无结果。"
            answer = _llm.answer(question, web_text, source="网络搜索（AnySearch）")
            return answer, [], None
        context = "\n".join(context_parts)
        # Rows were retrieved, but they may not actually answer the question
        # (hybrid search returns *some* rows even when the DB lacks the info).
        # If the context can't answer, fall back to live web search.
        if not _llm.can_answer(question, context):
            import websearch as _web

            web_text, web_err = _web.search(question, max_results=5)
            if web_err:
                return "", [], web_err
            if not web_text:
                return "", [], "未找到相关的数据行，且网络搜索无结果。"
            answer = _llm.answer(question, web_text, source="网络搜索（AnySearch）")
            return answer, [], None
        answer = _llm.answer(
            question,
            context,
            source=_build_source_label(source_rows),
        )
        return answer, source_rows, None
    except Exception as exc:  # noqa: BLE001
        return "", [], f"RAG 问答出错：{exc}"


_INTERNAL_COLS = {"__table", "__row_id", "sheet", "src_row"}


def rag_query_with_code(
    conn: sqlite3.Connection,
    selected: list[str],
    question: str,
    recall_pool: int = 50,
    top_n: int = 5,
) -> tuple[str, list[dict], str, str, str | None]:
    """RAG Q&A with a code interpreter.

    Retrieves relevant rows, asks the LLM to generate pandas code over those
    rows, executes it in a sandbox, and returns
    ``(answer, source_rows, code, code_result, error | None)``.
    """
    import llm as _llm
    import search as _search
    import pandas as pd
    from code_exec import run_code

    try:
        vec = _llm.embed([question])[0]
        results = _search.hybrid_search(conn, selected, question, vec, recall_pool=recall_pool)
        top = _rerank_results(conn, question, results, top_n=top_n)
        source_rows = _build_hybrid_rows(top, lambda t, r: _fetch_row_by_id(conn, t, r))
        if not source_rows:
            return "", [], "", "", "未找到相关的数据行。"
        data_rows = [{k: v for k, v in r.items() if k not in _INTERNAL_COLS} for r in source_rows]
        df = pd.DataFrame(data_rows)
        code = _llm.generate_code(question, df.head(5).to_string())
        _ok, code_result = run_code(code, df)
        context = (
            f"代码计算结果：\n{code_result}\n\n原始数据行：\n"
            + "\n".join(
                (_fetch_row_by_id(conn, r["__table"], r["__row_id"]) or {}).get("__row_text", "")
                for r in source_rows
            )
        )
        answer = _llm.answer(question, context, source=_build_source_label(source_rows))
        return answer, source_rows, code, code_result, None
    except Exception as exc:  # noqa: BLE001
        return "", [], "", "", f"代码解释器出错：{exc}"


def rag_query_with_review(
    conn: sqlite3.Connection,
    selected: list[str],
    question: str,
    recall_pool: int = 50,
    top_n: int = 5,
) -> tuple[str, list[dict], bool, str, str | None]:
    """RAG Q&A with a proposer-reviewer loop.

    Generates an answer, then has the LLM review it against the retrieved
    context. Returns ``(answer, source_rows, verdict, critique, error | None)``.
    """
    import llm as _llm
    import search as _search

    try:
        vec = _llm.embed([question])[0]
        results = _search.hybrid_search(conn, selected, question, vec, recall_pool=recall_pool)
        top = _rerank_results(conn, question, results, top_n=top_n)
        source_rows = _build_hybrid_rows(top, lambda t, r: _fetch_row_by_id(conn, t, r))
        if not source_rows:
            return "", [], False, "", "未找到相关的数据行。"
        context_parts: list[str] = []
        for row in source_rows:
            full = _fetch_row_by_id(conn, row["__table"], row["__row_id"])
            if full:
                context_parts.append(
                    full.get("__row_text") or json.dumps(_row_display_json(full), ensure_ascii=False)
                )
        context = "\n".join(context_parts)
        answer = _llm.answer(question, context, source=_build_source_label(source_rows))
        verdict, critique = _llm.review_answer(question, context, answer)
        return answer, source_rows, verdict, critique, None
    except Exception as exc:  # noqa: BLE001
        return "", [], False, "", f"RAG 审核问答出错：{exc}"


def rag_query_decomposed(
    conn: sqlite3.Connection,
    selected: list[str],
    question: str,
    recall_pool: int = 50,
    top_n: int = 5,
) -> tuple[str, list[dict], str | None]:
    """RAG Q&A with subquery decomposition.

    Asks the LLM whether the question should be split into subqueries, solves
    each via :func:`rag_query`, then synthesizes a final answer.
    Returns ``(answer, source_rows, error | None)``.
    """
    import db as _db
    import llm as _llm

    try:
        schemas = [_db.get_schema(conn, t) for t in selected]
        subqueries = _llm.decompose_question(question, schemas)
        if len(subqueries) <= 1:
            return rag_query(conn, selected, question, recall_pool=recall_pool, top_n=top_n)
        parts: list[str] = []
        all_rows: list[dict] = []
        for sq in subqueries:
            ans, rows, err = rag_query(conn, selected, sq, recall_pool=recall_pool, top_n=top_n)
            if err:
                parts.append(f"子问题：{sq}\n答案：{err}")
            else:
                parts.append(f"子问题：{sq}\n答案：{ans}")
            all_rows.extend(rows)
        if not all_rows:
            return "", [], "未找到相关的数据行。"
        final = _llm.answer(question, "\n\n".join(parts), source=_build_source_label(all_rows))
        return final, all_rows, None
    except Exception as exc:  # noqa: BLE001
        return "", [], f"子查询分解问答出错：{exc}"


def rag_query_dual(
    conn: sqlite3.Connection,
    selected: list[str],
    question: str,
    recall_pool: int = 50,
    top_n: int = 5,
    max_attempts: int = 2,
) -> tuple[str, list[dict], str, str, str | None]:
    """RAG Q&A with SQL + text dual-path cross-validation.

    Runs NL2SQL (via :func:`ask_query`) and text retrieval (the same hybrid
    search + rerank pipeline as :func:`rag_query`), then has the LLM arbitrate
    between the two. Returns ``(answer, source_rows, sql, sql_context, error)``.
    """
    import llm as _llm
    import search as _search

    try:
        # SQL path
        sql, _cols, sql_rows, sql_err = ask_query(conn, selected, question, max_attempts=max_attempts)
        if sql_err or not sql_rows:
            sql_context = f"SQL 执行失败：{sql_err}" if sql_err else "SQL 无结果"
        else:
            sql_context = "\n".join(str(r) for r in sql_rows)

        # Text path
        vec = _llm.embed([question])[0]
        results = _search.hybrid_search(conn, selected, question, vec, recall_pool=recall_pool)
        top = _rerank_results(conn, question, results, top_n=top_n)
        source_rows = _build_hybrid_rows(top, lambda t, r: _fetch_row_by_id(conn, t, r))
        context_parts: list[str] = []
        for row in source_rows:
            full = _fetch_row_by_id(conn, row["__table"], row["__row_id"])
            if full:
                context_parts.append(
                    full.get("__row_text") or json.dumps(_row_display_json(full), ensure_ascii=False)
                )
        text_context = "\n".join(context_parts)
        if not source_rows and (sql_err or not sql_rows):
            return "", [], sql, sql_context, "未找到相关的数据行。"
        answer = _llm.cross_validate(question, sql_context, text_context)
        label = _build_source_label(source_rows)
        if label:
            answer = f"{answer}\n\n【来源：{label}】"
        return answer, source_rows, sql, sql_context, None
    except Exception as exc:  # noqa: BLE001
        return "", [], "", "", f"SQL+文本交叉验证出错：{exc}"


def stats_query(
    conn: sqlite3.Connection,
    selected: list[str],
) -> tuple[dict | None, str | None]:
    """Column summary stats for the first selected table.

    Returns ``(summary, error | None)`` where *summary* is the dict from
    :func:`db.summarize`.
    """
    import db as _db

    if not selected:
        return None, "请先勾选至少一个参与统计的表。"
    try:
        summary = _db.summarize(conn, selected[0])
        return summary, None
    except Exception as exc:  # noqa: BLE001
        return None, f"统计出错：{exc}"


def build_stats_data(
    conn: sqlite3.Connection,
    table: str,
    bins: int = 20,
    top_n: int = 10,
) -> dict:
    """Gather all extra data needed for the enhanced stats visualizations.

    Returns a dict with keys:
    - ``numeric_bins``: {col_name: [(label, count), ...]}
    - ``text_top_n``: {col_name: [(value, count), ...]}
    - ``missing``: [{列名, 非空数, 缺失数, 填充率}, ...]
    - ``numeric_compare``: [{列名, 求和, 平均}, ...] (only columns with non-null aggregates)

    Framework-free for testability.
    """
    import db as _db

    summary = _db.summarize(conn, table)
    cols_meta = summary["columns"]
    row_count = summary["row_count"]

    # Numeric column histograms
    num_bins: dict[str, list[tuple[str, int]]] = {}
    for c in cols_meta:
        if c["类型"] in ("INTEGER", "REAL") and c["非空数"] and c["非空数"] > 0:
            num_bins[c["列名"]] = _db.numeric_bins(conn, table, c["列名"], bins=bins)

    # Text column top-N value counts
    text_top: dict[str, list[tuple[str, int]]] = {}
    for c in cols_meta:
        if c["类型"] == "TEXT" and c["非空数"] and c["非空数"] > 0:
            text_top[c["列名"]] = _db.column_value_counts(conn, table, c["列名"], limit=top_n)

    # Missing value data
    missing = []
    for c in cols_meta:
        nn = c["非空数"] or 0
        missing.append({
            "列名": c["列名"],
            "类型": c["类型"],
            "非空数": nn,
            "缺失数": row_count - nn,
            "填充率": round(nn / row_count * 100, 1) if row_count else 0.0,
        })

    # Numeric comparison (求和/平均 side by side)
    num_compare = []
    for c in cols_meta:
        if c["类型"] in ("INTEGER", "REAL") and c["求和"] is not None:
            num_compare.append({"列名": c["列名"], "求和": c["求和"], "平均": round(c["平均"], 4) if c["平均"] else 0})

    return {
        "numeric_bins": num_bins,
        "text_top_n": text_top,
        "missing": missing,
        "numeric_compare": num_compare,
    }


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def main() -> None:
    import streamlit as st
    import db as _db
    import search as _search
    import pandas as pd

    st.set_page_config(
        page_title="电子表格数据库",
        page_icon=":material/database:",
        layout="wide",
    )
    st.title("电子表格数据库", text_alignment="left")

    # ---- session state ---------------------------------------------------
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "result_rows" not in st.session_state:
        st.session_state.result_rows = []
    if "result_mode" not in st.session_state:
        st.session_state.result_mode = None  # "hybrid" | "ask" | "sql"
    if "ask_sql" not in st.session_state:
        st.session_state.ask_sql = None
    if "show_detail_row" not in st.session_state:
        st.session_state.show_detail_row = None
    if "rag_answer" not in st.session_state:
        st.session_state.rag_answer = None
    if "rag_dual_sql" not in st.session_state:
        st.session_state.rag_dual_sql = None
    if "rag_dual_ctx" not in st.session_state:
        st.session_state.rag_dual_ctx = None
    if "auto_used_tables" not in st.session_state:
        st.session_state.auto_used_tables = None

    # ---- main area: table selection (above sidebar so `selected` is in scope) --
    tables = list_tables()
    with st.container():
        selected = st.multiselect(
            "选择参与搜索的表",
            options=tables,
            default=[],
        )
        auto_select = st.checkbox(
            "自动选择表",
            value=False,
            help="开启后，每次查询自动从所有表中选出与问题最相关的表（最多 3 张），无需手动勾选。",
        )
        if tables:
            st.caption(f"共 {len(tables)} 张表" + (f" · 已选 {len(selected)}" if selected else ""))

    # ---- sidebar ---------------------------------------------------------
    with st.sidebar:
        st.header("数据管理", divider=False)

        # Upload
        header_row = st.number_input("表头所在行（第几行为列名）", min_value=1, value=1, step=1)
        import_mode = st.radio(
            "导入方式",
            ["新建表", "替换", "更新", "合并"],
            horizontal=True,
            key="import_mode",
            help="新建表：作为全新表创建，若表名已存在则报错。替换：整表覆盖。更新：按主键更新已有行，删除新文件中不存在的行。合并：按主键更新已有行，保留原有行。",
        )
        _mode_map = {"替换": "replace", "更新": "update", "合并": "merge", "新建表": "create"}
        _selected_mode = _mode_map[import_mode]

        key_col: str | None = None
        if _selected_mode in ("update", "merge"):
            # Build candidate key columns from the first selected table's schema.
            _key_options: list[str] = []
            if selected:
                try:
                    _conn_schema = _db.get_conn()
                    try:
                        _schema = _db.get_schema(_conn_schema, selected[0])
                        _key_options = [c[0] for c in _schema["columns"]]
                    finally:
                        _conn_schema.close()
                except Exception:  # noqa: BLE001
                    _key_options = []
            if _key_options:
                key_col = st.selectbox("主键列", options=_key_options, index=0)
            else:
                st.info("没有可用的主键列。请先导入表，或切换为替换模式。")

        uploaded = st.file_uploader(
            "上传 Excel / CSV",
            type=["xlsx", "xls", "csv"],
            key=f"upload_{st.session_state.uploader_key}",
            help="支持 .xlsx / .xls / .csv",
        )
        if uploaded is not None:
            suffix = os.path.splitext(uploaded.name)[1].lower()
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            try:
                tmp.write(uploaded.getvalue())
                tmp.close()

                progress_bar = st.progress(0, text="准备导入…")

                def _progress(frac: float, msg: str) -> None:
                    progress_bar.progress(frac, text=msg)

                with st.spinner("正在导入…"):
                    name, updated = ingest_file(
                        tmp.name, on_progress=_progress, name=uploaded.name, header_row=int(header_row), key_col=key_col, mode=_selected_mode,
                    )
                verb = "更新" if updated else "导入"
                st.success(f"已{verb}表：{name}")
                st.session_state.uploader_key += 1
                st.rerun()
            except Exception as exc:
                st.error(f"导入失败：{exc}")
            finally:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass

        st.divider()

        # Delete table
        if tables:
            del_target = st.selectbox("删除表", options=tables, index=None, placeholder="选择要删除的表")
            if st.button("删除选中表", type="primary", disabled=del_target is None, icon=":material/delete:"):
                conn_del = _db.get_conn()
                try:
                    _db.delete_table(conn_del, del_target)
                    st.success(f"已删除表：{del_target}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"删除失败：{exc}")
                finally:
                    conn_del.close()

    # ---- two-column layout: main (left) + data preview (right) ----------
    col_main, col_preview = st.columns([3, 2])

    with col_main:
      st.subheader("查询", divider=False)

      _MODE_LABELS = {
          "hybrid": "混合搜索",
          "ask": "自然语言",
          "sql": "SQL",
          "rag": "智能问答",
          "统计": "统计",
      }
      mode = st.segmented_control(
          "查询模式",
          ["hybrid", "ask", "sql", "rag", "统计"],
          default="hybrid",
          format_func=lambda m: _MODE_LABELS.get(m, m),
      )

      # 显示条数 — 仅对使用它的查询方式（混合搜索 / 智能问答）显示，且各自独立记忆
      if mode == "hybrid":
          top_n = int(st.number_input("显示条数", min_value=1, value=20, step=1, key="top_n_hybrid"))
      elif mode == "rag":
          top_n = int(st.number_input("显示条数", min_value=1, value=5, step=1, key="top_n_rag"))
      else:
          top_n = 20

      result_rows: list[dict] = st.session_state.result_rows
      result_mode: str | None = st.session_state.result_mode

      # -- hybrid mode -------------------------------------------------------
      if mode == "hybrid":
          query = st.text_input("关键词 / 短语搜索", placeholder="输入要搜索的内容")
          score_floor = st.slider("相关度阈值", min_value=0.0, max_value=1.0, value=0.0, step=0.05,
                                  help="0 = 不筛选，1 = 仅保留最高分结果")
          min_results = int(st.number_input("最少返回条数", min_value=0, value=5, step=1, key="min_results_hybrid",
                                              help="默认 5（阈值再高也至少返回 5 条，避免空结果）；0 = 不限制；N = 至少返回 N 条（受显示条数上限约束）"))
          view_mode_label = st.segmented_control("排序方式", options=["按重排分", "按原始混合分"], default="按重排分")
          view_mode = "rerank" if view_mode_label == "按重排分" else "fusion"
          if st.button("搜索", type="primary", icon=":material/search:"):
              if not query or not query.strip():
                  st.warning("请输入搜索内容。")
              elif not selected and not auto_select:
                  st.warning("请先在上方选择至少一个参与查询的表，或开启自动选择表。")
              else:
                  conn = _db.get_conn()
                  try:
                      use_tables = _search.select_tables(conn, query, k=3) if auto_select else selected
                      rows, err = hybrid_query(conn, use_tables, query, score_floor_frac=score_floor,
                                               view_mode=view_mode, top_n=top_n, recall_pool=50,
                                               min_results=min_results)
                  finally:
                      conn.close()
                  st.session_state.auto_used_tables = use_tables if auto_select else None
                  if err:
                      st.error(err)
                  elif not rows:
                      st.info("未找到匹配的搜索结果。")
                  else:
                      st.session_state.result_rows = rows
                      st.session_state.result_mode = "hybrid"
                      st.session_state.ask_sql = None
                      st.session_state.show_detail_row = None
                      result_rows = rows
                      result_mode = "hybrid"
          if result_mode == "hybrid" and result_rows:
              st.caption(f"共 {len(result_rows)} 条结果")

      # -- ask mode ----------------------------------------------------------
      elif mode == "ask":
          question = st.text_area("用自然语言提问", placeholder="例如：找出金额大于 100 的记录", height=80)
          if st.button("提问", type="primary", icon=":material/help:"):
              if not question or not question.strip():
                  st.warning("请输入问题。")
              elif not selected and not auto_select:
                  st.warning("请先在上方选择至少一个参与查询的表，或开启自动选择表。")
              else:
                  conn = _db.get_conn()
                  try:
                      use_tables = _search.select_tables(conn, question, k=3) if auto_select else selected
                      sql, cols, rows, err = ask_query(conn, use_tables, question)
                  finally:
                      conn.close()
                  st.session_state.auto_used_tables = use_tables if auto_select else None
                  if err:
                      st.error(err)
                  else:
                      st.session_state.ask_sql = sql
                      if not rows:
                          st.info("未找到匹配的行。")
                          st.session_state.result_rows = []
                          st.session_state.result_mode = "ask"
                      else:
                          st.session_state.result_rows = rows
                          st.session_state.result_mode = "ask"
                      st.session_state.show_detail_row = None
                      result_rows = st.session_state.result_rows
                      result_mode = "ask"

          if st.session_state.get("ask_sql"):
              st.code(st.session_state.ask_sql, language="sql")

      # -- sql mode ----------------------------------------------------------
      elif mode == "sql":
          sql_input = st.text_area("手写 SQL", placeholder="SELECT * FROM 表名", height=80)
          if st.button("执行", type="primary", icon=":material/play_arrow:"):
              if not sql_input or not sql_input.strip():
                  st.warning("请输入 SQL 语句。")
              else:
                  conn = _db.get_conn()
                  try:
                      cols, rows, err = sql_query(conn, sql_input)
                  finally:
                      conn.close()
                  if err:
                      st.error(err)
                  elif not rows:
                      st.info("未找到匹配的行。")
                      st.session_state.result_rows = []
                      st.session_state.result_mode = "sql"
                  else:
                      st.session_state.result_rows = rows
                      st.session_state.result_mode = "sql"
                  st.session_state.ask_sql = None
                  st.session_state.show_detail_row = None
                  result_rows = st.session_state.result_rows
                  result_mode = "sql"

      # -- rag mode -----------------------------------------------------------
      elif mode == "rag":
          question = st.text_area("用自然语言提问（基于数据库内容回答）", placeholder="例如：哪些客户的金额超过 100？", height=80)
          use_code_interpreter = st.toggle("启用代码解释器", key="rag_use_code")
          use_decompose = st.toggle(
              "启用子查询分解",
              key="rag_decompose",
              help="复杂问题先拆分为多个子查询逐个求解，再综合回答。",
          )
          use_dual = st.toggle(
              "SQL+文本交叉验证",
              key="rag_dual",
              help="同时用 SQL 和文本检索回答，交叉验证冲突。与子查询分解互斥，分解优先。",
          )
          if st.button("问答", type="primary", icon=":material/chat:"):
              if not question or not question.strip():
                  st.warning("请输入问题。")
              elif not selected and not auto_select:
                  st.warning("请先在上方选择至少一个参与查询的表，或开启自动选择表。")
              else:
                  conn = _db.get_conn()
                  try:
                      use_tables = _search.select_tables(conn, question, k=3) if auto_select else selected
                      if use_decompose:
                          answer_text, rows, err = rag_query_decomposed(conn, use_tables, question, top_n=top_n)
                          st.session_state.rag_code = None
                          st.session_state.rag_code_result = None
                          st.session_state.rag_dual_sql = None
                          st.session_state.rag_dual_ctx = None
                      elif use_dual:
                          answer_text, rows, sql_str, sql_ctx, err = rag_query_dual(conn, use_tables, question, top_n=top_n)
                          st.session_state.rag_code = None
                          st.session_state.rag_code_result = None
                          st.session_state.rag_dual_sql = sql_str
                          st.session_state.rag_dual_ctx = sql_ctx
                      elif use_code_interpreter:
                          answer_text, rows, code, code_result, err = rag_query_with_code(conn, use_tables, question, top_n=top_n)
                          st.session_state.rag_code = code
                          st.session_state.rag_code_result = code_result
                          st.session_state.rag_dual_sql = None
                          st.session_state.rag_dual_ctx = None
                      else:
                          answer_text, rows, err = rag_query(conn, use_tables, question, top_n=top_n)
                          st.session_state.rag_code = None
                          st.session_state.rag_code_result = None
                          st.session_state.rag_dual_sql = None
                          st.session_state.rag_dual_ctx = None
                  finally:
                      conn.close()
                  st.session_state.auto_used_tables = use_tables if auto_select else None
                  if err:
                      st.error(err)
                  else:
                      st.session_state.rag_answer = answer_text
                      if not rows:
                          st.info("未找到相关的数据行。")
                          st.session_state.result_rows = []
                      else:
                          st.session_state.result_rows = rows
                      st.session_state.result_mode = "rag"
                      st.session_state.ask_sql = None
                      st.session_state.show_detail_row = None
                      result_rows = st.session_state.result_rows
                      result_mode = "rag"

          if st.session_state.get("rag_answer"):
              st.info(st.session_state.rag_answer, icon=":material/chat:")

          if st.session_state.get("rag_code"):
              st.markdown("**生成的计算代码**")
              st.code(st.session_state.rag_code, language="python")
              st.markdown("**代码执行结果**")
              st.code(st.session_state.rag_code_result)

          if st.session_state.get("rag_dual_sql"):
              with st.expander("SQL 执行结果"):
                  st.code(st.session_state.rag_dual_sql, language="sql")
                  st.code(st.session_state.rag_dual_ctx)

      # -- stats mode --------------------------------------------------------
      elif mode == "统计":
          if st.button("统计", type="primary", icon=":material/analytics:"):
              if not selected:
                  st.warning("请先在上方选择至少一个参与统计的表。")
              else:
                  conn = _db.get_conn()
                  try:
                      summary, err = stats_query(conn, selected)
                      extra = build_stats_data(conn, selected[0]) if not err else {}
                  finally:
                      conn.close()
                  if err:
                      st.error(err)
                  elif summary:
                      cols_meta = summary["columns"]
                      n_cols = len(cols_meta)
                      n_numeric = sum(1 for c in cols_meta if c["类型"] in ("INTEGER", "REAL"))
                      n_text = n_cols - n_numeric

                      # ── KPI row ────────────────────────────────────────
                      with st.container(horizontal=True):
                          st.metric("总行数", summary["row_count"], border=True)
                          st.metric("列数", n_cols, border=True)
                          st.metric("数值列", n_numeric, border=True)
                          st.metric("文本列", n_text, border=True)

                      # ── Column type distribution ───────────────────────
                      with st.container(border=True):
                          st.markdown("**列类型分布**")
                          type_counts: dict[str, int] = {}
                          for c in cols_meta:
                              type_counts[c["类型"]] = type_counts.get(c["类型"], 0) + 1
                          dist_df = pd.DataFrame(
                              {"类型": list(type_counts.keys()), "列数": list(type_counts.values())}
                          )
                          st.bar_chart(dist_df, x="类型", y="列数")

                      # ── Tabs for richer charts ─────────────────────────
                      tab_labels = []
                      if n_numeric > 0:
                          tab_labels.append("数值列分布")
                      if extra.get("numeric_compare"):
                          tab_labels.append("数值列对比")
                      if extra.get("text_top_n"):
                          tab_labels.append("文本列 Top 值")
                      if extra.get("missing"):
                          tab_labels.append("缺失值概览")
                      tab_labels.append("各列统计")

                      if tab_labels:
                          tabs = st.tabs(tab_labels)
                          tab_idx = 0

                          # Tab: Numeric distribution
                          if n_numeric > 0:
                              with tabs[tab_idx]:
                                  st.markdown("**数值列分布**（直方图）")
                                  for col_name, bins_data in extra.get("numeric_bins", {}).items():
                                      if bins_data:
                                          bin_df = pd.DataFrame({"区间": [b[0] for b in bins_data], "数量": [b[1] for b in bins_data]})
                                          with st.expander(f"**{col_name}**", expanded=(len(extra.get("numeric_bins", {})) == 1)):
                                              st.bar_chart(bin_df, x="区间", y="数量", height=200)
                              tab_idx += 1

                          # Tab: Numeric comparison
                          if extra.get("numeric_compare"):
                              with tabs[tab_idx]:
                                  st.markdown("**数值列对比**（求和 / 平均）")
                                  nc = extra["numeric_compare"]
                                  # Horizontal bar chart for 求和
                                  sum_df = pd.DataFrame({"列名": [r["列名"] for r in nc], "求和": [r["求和"] for r in nc]})
                                  st.markdown("*求和*")
                                  st.bar_chart(sum_df, x="列名", y="求和", horizontal=True)
                                  # Average
                                  avg_df = pd.DataFrame({"列名": [r["列名"] for r in nc], "平均": [r["平均"] for r in nc]})
                                  st.markdown("*平均值*")
                                  st.bar_chart(avg_df, x="列名", y="平均", horizontal=True)
                              tab_idx += 1

                          # Tab: Text top-N
                          if extra.get("text_top_n"):
                              with tabs[tab_idx]:
                                  st.markdown("**文本列 Top 值**（出现频次最高的值）")
                                  for col_name, pairs in extra["text_top_n"].items():
                                      if pairs:
                                          val_df = pd.DataFrame({"值": [p[0] for p in pairs], "数量": [p[1] for p in pairs]})
                                          _dedup = next((c["去重数"] for c in cols_meta if c["列名"] == col_name), "—")
                                          with st.expander(f"**{col_name}**（去重 {_dedup} 个）", expanded=(len(extra["text_top_n"]) == 1)):
                                              st.bar_chart(val_df, x="值", y="数量", height=200)
                              tab_idx += 1

                          # Tab: Missing values
                          if extra.get("missing"):
                              with tabs[tab_idx]:
                                  st.markdown("**缺失值概览**")
                                  miss_df = pd.DataFrame(extra["missing"])
                                  # Fill-rate bar chart
                                  fr_df = pd.DataFrame({"列名": miss_df["列名"], "填充率 (%)": miss_df["填充率"]})
                                  st.bar_chart(fr_df, x="列名", y="填充率 (%)", height=250)
                                  # Detailed table below
                                  st.dataframe(miss_df, hide_index=True, use_container_width=True)
                              tab_idx += 1

                          # Tab: Per-column aggregates (always last)
                          with tabs[tab_idx]:
                              st.markdown("**各列统计**")
                              st.dataframe(cols_meta, hide_index=True, use_container_width=True)

      # ---- results area ----------------------------------------------------
      auto_used = st.session_state.get("auto_used_tables")
      if auto_used:
          st.caption(f"自动选择表：{'、'.join(auto_used)}")
      if result_rows:
          st.subheader("查询结果", divider=False)
          # Group results by source table so each table is shown in its own block.
          _order = []
          for r in result_rows:
              t = r.get("__table")
              if t is not None and t not in _order:
                  _order.append(t)
          groups = {}
          for r in result_rows:
              groups.setdefault(r.get("__table"), []).append(r)
          _order = _order + ([None] if None in groups else [])

          for tbl in _order:
              rows = groups[tbl]
              # Drop rows with no visible data so empty rows are not shown.
              visible = [r for r in rows if _row_has_visible_data(r)]
              if not visible:
                  continue
              if tbl is not None:
                  st.subheader(f"表格：{tbl}", divider=False)
              df_data = [
                  {k: v for k, v in row.items() if not k.startswith("__")}
                  for row in visible
              ]
              event = st.dataframe(
                  df_data,
                  on_select="rerun",
                  selection_mode="single-row",
                  key=f"result_df_{tbl}",
                  hide_index=True,
              )

              # Row detail
              sel = event.selection.rows if event and event.selection else []
              if sel:
                  row = visible[sel[0]]
                  table = row.get("__table")
                  rid = row.get("__row_id")
                  if table and rid:
                      conn = _db.get_conn()
                      try:
                          full = _fetch_row_by_id(conn, table, rid)
                      finally:
                          conn.close()
                      if full:
                          st.subheader("行详情", divider=False)
                          st.code(
                              json.dumps(_row_display_json(full), ensure_ascii=False, indent=2, default=str),
                              language="json",
                          )

              # CSV download (per table)
              csv_bytes = _to_csv(df_data)
              st.download_button(
                  f"下载 CSV（{tbl}）" if tbl is not None else "下载 CSV",
                  data=csv_bytes,
                  file_name=f"result_{tbl}.csv" if tbl is not None else "result.csv",
                  mime="text/csv",
                  icon=":material/download:",
                  key=f"dl_{tbl}",
              )

    # ---- right column: data preview --------------------------------------
    with col_preview:
        st.subheader("数据预览（各表前 5 行）", divider=False)
        if not selected:
            st.info("请先在上方选择参与搜索的表。")
        else:
            conn_pv = _db.get_conn()
            try:
                for tbl in selected:
                    _pv_cols, pv_rows = _db.get_preview(conn_pv, tbl, n=5)
                    st.markdown(f"**{tbl}**")
                    if pv_rows:
                        st.dataframe(pv_rows, hide_index=True, key=f"preview_{tbl}")
                    else:
                        st.info(f"{tbl}：暂无数据。")
            finally:
                conn_pv.close()


if __name__ == "__main__":
    main()
