"""Framework-free query / analysis orchestration layer.

This module is the application core: it imports the lower-layer modules
(``db``, ``llm``, ``ingest``, ``search``, ``execute_sql``, ``code_exec``,
``websearch``) and orchestrates them into the high-level operations the UI
calls (hybrid search, NL2SQL, stats, RAG Q&A). Importing the *modules*
(``from ..ai import llm``) rather than their members lets tests monkeypatch
``llm.embed`` / ``db.get_schema`` etc. on the module attributes.
"""

import csv
import io
import json
import sqlite3

import pandas as pd

from ..data import db
from ..ai import llm
from ..ai import execute_sql, code_exec, websearch
from ..services import ingest, search


# Internal columns that are bookkeeping, not user data.
_INTERNAL_COLS = ("row_id", "__row_text", "sheet", "src_row")


# ---------------------------------------------------------------------------
# CSV / row helpers
# ---------------------------------------------------------------------------


def _to_csv(rows) -> bytes:
    """Render *rows* to CSV bytes (utf-8-sig with BOM).

    ``rows`` may be a list of dicts (``csv.DictWriter``) or a list of lists
    (``csv.writer``). An empty list returns ``b""``.
    """
    if not rows:
        return b""
    buf = io.StringIO()
    first = rows[0]
    if isinstance(first, dict):
        cols = list(first.keys())
        writer = csv.DictWriter(buf, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    else:
        writer = csv.writer(buf)
        writer.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")


def _run_query(conn, sql) -> tuple[list[str], list[dict]]:
    """Execute *sql* and return ``(columns, rows)``.

    columns come from ``cursor.description``; rows are dicts. Handles both
    dict-like sqlite3.Row objects and plain tuples (tests use plain in-memory
    connections). Read-only validation is *not* performed here — that is
    :func:`sql_query`'s job.
    """
    cur = conn.execute(sql)
    cols = [d[0] for d in (cur.description or [])]
    raw = cur.fetchall()
    rows = [
        dict(r) if hasattr(r, "keys") else dict(zip(cols, r))
        for r in raw
    ]
    return cols, rows


def _fetch_row_by_id(conn, table, row_id) -> dict | None:
    """Fetch one row by ``row_id``; return a dict, or None if unmatched."""
    rows = conn.execute(f'SELECT * FROM "{table}" WHERE row_id=?', (row_id,)).fetchall()
    if not rows:
        return None
    return dict(rows[0])


def _row_display_json(row: dict) -> dict:
    """Drop the internal ``__row_text`` key before displaying a row."""
    return {k: v for k, v in row.items() if k != "__row_text"}


def _row_has_visible_data(row: dict) -> bool:
    """Return True if any *data* column holds a non-None value.

    Internal columns (``row_id`` / ``__row_text`` / ``sheet`` / ``src_row``)
    are excluded. Note ``0`` counts as a real value (only None means empty).
    """
    for col, val in row.items():
        if col in _INTERNAL_COLS:
            continue
        if val is not None:
            return True
    return False


def _columns_for(cols) -> list[str]:
    """Passthrough helper for column name lists."""
    return list(cols)


def _build_hybrid_rows(results, fetch) -> list[dict]:
    """Turn ``[(table, row_id, score), ...]`` into display rows.

    For each result, *fetch*(table, row_id) returns the row dict (or None).
    Rows that fail to fetch or have no visible data are skipped; surviving
    rows get ``__table`` / ``__row_id`` attached and any Chinese display keys
    (表名/行号/分数/摘要) removed. Order matches *results* (after filtering).
    """
    rows = []
    for table, row_id, _score in results:
        row = fetch(table, row_id)
        if row is None:
            continue
        if not _row_has_visible_data(row):
            continue
        clean = {k: v for k, v in row.items() if k not in ("表名", "行号", "分数", "摘要")}
        clean["__table"] = table
        clean["__row_id"] = row_id
        rows.append(clean)
    return rows


def _rerank_results(conn, query, results, top_n=5) -> list:
    """Rerank *results* by LLM relevance and truncate to *top_n*.

    *results* is ``[(table, row_id, score), ...]`` (RRF order). Docs are the
    fetched ``__row_text`` values; rows are re-sorted by descending rerank
    score and truncated. On any rerank failure the original order (RRF) is
    kept and truncated.
    """
    if not results:
        return []
    docs = []
    for table, row_id, _ in results:
        row = _fetch_row_by_id(conn, table, row_id)
        if row is not None:
            docs.append(row.get("__row_text") or "")
        else:
            docs.append("")
    try:
        scores = llm.rerank(query, docs)
    except Exception:  # noqa: BLE001 - fall back to RRF order
        return results[:top_n]
    paired = sorted(zip(results, scores), key=lambda p: p[1], reverse=True)
    return [r for r, _ in paired][:top_n]


# ---------------------------------------------------------------------------
# SQL / table management
# ---------------------------------------------------------------------------


def sql_query(conn, sql) -> tuple[list[str], list[dict], str | None]:
    """Validate *sql* is read-only, then run it. Returns (cols, rows, err)."""
    try:
        execute_sql.assert_readonly_sql(sql)
    except Exception as exc:
        return [], [], f"只读校验失败：{exc}"
    try:
        cols, rows = _run_query(conn, sql)
    except Exception as exc:
        return [], [], f"SQL 执行出错: {exc}"
    return cols, rows, None


def list_tables() -> list[str]:
    conn = db.get_conn()
    try:
        return db.list_tables(conn)
    finally:
        conn.close()


def delete_table(conn, table) -> None:
    db.delete_table(conn, table)


def preview(conn, name, n=5) -> tuple[list[str], list[dict]]:
    return db.get_preview(conn, name, n)


def ingest_file(path, on_progress=None, name=None, header_row=1, **kwargs) -> tuple[str, bool]:
    conn = db.get_conn()
    try:
        return ingest.ingest_file(
            conn, path, on_progress=on_progress, name=name, header_row=header_row, **kwargs
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def stats_query(conn, tables: list[str]) -> tuple[dict | None, str | None]:
    if not tables:
        return None, "请先勾选至少一个表。"
    summary = db.summarize(conn, tables[0])
    return summary, None


def build_stats_data(conn, name, bins=20, top_n=10) -> dict:
    schema = db.get_schema(conn, name)
    columns = schema.get("columns", [])  # [(col_name, col_type), ...]

    numeric_bins: dict = {}
    text_top_n: dict = {}
    missing: list[dict] = []
    numeric_compare: list[dict] = []

    total = conn.execute(f'SELECT COUNT(*) AS c FROM "{name}"').fetchone()["c"] or 0

    for col, col_type in columns:
        # Missing value stats for every column.
        n_null = conn.execute(f'SELECT COUNT(*) AS c FROM "{name}" WHERE "{col}" IS NULL').fetchone()["c"]
        pct = round((total - n_null) / total * 100, 1) if total else 0.0
        missing.append({"列名": col, "缺失数": n_null, "填充率": pct})

        if col_type in ("INTEGER", "REAL"):
            col_bins = db.numeric_bins(conn, name, col, bins=bins)
            if col_bins:
                numeric_bins[col] = col_bins
            row = conn.execute(
                f'SELECT COUNT("{col}") AS nn, SUM("{col}") AS s, AVG("{col}") AS a, '
                f'MIN("{col}") AS mn, MAX("{col}") AS mx FROM "{name}"'
            ).fetchone()
            numeric_compare.append({
                "列名": col,
                "类型": col_type,
                "非空数": row["nn"],
                "求和": row["s"],
                "平均": row["a"],
                "最小": row["mn"],
                "最大": row["mx"],
                "去重数": None,
            })
        else:
            counts = db.column_value_counts(conn, name, col, limit=top_n)
            if counts:
                text_top_n[col] = counts

    return {
        "numeric_bins": numeric_bins,
        "text_top_n": text_top_n,
        "missing": missing,
        "numeric_compare": numeric_compare,
    }


def category_counts(conn, name, col, limit=10) -> list[dict]:
    counts = db.column_value_counts(conn, name, col, limit=limit)
    return [{"值": value, "数量": count} for value, count in counts]


# ---------------------------------------------------------------------------
# NL2SQL
# ---------------------------------------------------------------------------


def _ask(conn, tables, query, max_attempts=2):
    """Loop NL2SQL generation + execution up to *max_attempts* times.

    Returns ``(sql, cols, rows, err)``. On success err is None; if every
    attempt produced an OperationalError, the last sql is returned with
    ``(None, None)`` for cols/rows and an error message containing "NL2SQL".
    """
    prev_error = None
    sql = ""
    for _ in range(max_attempts):
        schemas = [db.get_schema(conn, t) for t in tables]
        sql = llm.generate_sql(schemas, query, prev_error=prev_error)
        try:
            cols, rows = _run_query(conn, sql)
            return sql, cols, rows, None
        except sqlite3.OperationalError as exc:
            prev_error = str(exc)
    return sql, None, None, f"NL2SQL 生成失败/执行出错: {prev_error}"


def ask_query(conn, tables, query, max_attempts=2):
    sql, cols, rows, err = _ask(conn, tables, query, max_attempts=max_attempts)
    if err is not None:
        return sql, [], [], err
    return sql, cols, rows, None


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------


def hybrid_query(conn, tables, query, top_n=5, score_floor_frac=None, view_mode=None,
                 recall_pool=None, min_results=None):
    """Hybrid (BM25 + vector + RRF) search, reranked, truncated to *top_n*.

    Returns ``(rows, err)``. Extra kwargs (score_floor_frac/view_mode/min_results)
    are accepted for app.py compatibility and currently ignored.
    """
    try:
        vec = llm.embed([query])[0]
    except Exception as exc:
        return [], f"搜索出错: {exc}"

    results = search.hybrid_search(conn, tables, query, vec)

    rows = _build_hybrid_rows(results, lambda t, r: _fetch_row_by_id(conn, t, r))

    # Rerank ALL candidates (not just an old fixed pool) then truncate.
    if rows:
        docs = [row["__row_text"] for row in rows]
        try:
            scores = llm.rerank(query, docs)
            paired = sorted(zip(rows, scores), key=lambda p: p[1], reverse=True)
            rows = [r for r, _ in paired]
        except Exception:  # noqa: BLE001 - fall back to RRF order
            pass

    return rows[:top_n], None


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------


def _db_source(tables, row_ids):
    """Build the DB source annotation ``数据库表格：t1,t2（行 1,2）``."""
    return "数据库表格：" + ",".join(tables) + "（行 " + ",".join(map(str, row_ids)) + "）"


def rag_query(conn, tables, query, top_n=5):
    """Naive RAG: retrieve rows, rerank, answer — with a web-search fallback.

    Returns ``(answer, rows, err)``.
    """
    try:
        vec = llm.embed([query])[0]
    except Exception as exc:
        return "", [], f"RAG 问答出错: {exc}"

    results = search.hybrid_search(conn, tables, query, vec)
    rows = _rerank_results(conn, query, results, top_n=top_n)
    rows = _build_hybrid_rows(rows, lambda t, r: _fetch_row_by_id(conn, t, r))

    if rows:
        context = "\n".join(row["__row_text"] for row in rows)
        source = _db_source(tables, [row["__row_id"] for row in rows])
        try:
            can = llm.can_answer(query, context)
        except Exception:
            can = True
        if can:
            answer = llm.answer(query, context, source=source)
            return answer, rows, None

    # Web-search fallback (no rows, or LLM judged rows insufficient).
    web_text, web_err = websearch.search(query, max_results=5)
    if web_err is not None and not web_text:
        return "", [], f"网络搜索失败：{web_err}"
    if web_text:
        answer = llm.answer(query, web_text, source="网络搜索（AnySearch）")
        return answer, [], None
    return "", [], "网络搜索未返回结果。"


def rag_query_stream(conn, tables, query, top_n=5, history=None, out_rows=None):
    """Streaming RAG; yields answer chunks. When *out_rows* is given it is
    filled with the retrieved rows (only when the DB path is taken)."""
    try:
        vec = llm.embed([query])[0]
    except Exception as exc:
        yield f"RAG 问答出错: {exc}"
        return

    results = search.hybrid_search(conn, tables, query, vec)
    rows = _rerank_results(conn, query, results, top_n=top_n)
    rows = _build_hybrid_rows(rows, lambda t, r: _fetch_row_by_id(conn, t, r))

    if rows:
        context = "\n".join(row["__row_text"] for row in rows)
        source = _db_source(tables, [row["__row_id"] for row in rows])
        try:
            can = llm.can_answer(query, context)
        except Exception:
            can = True
        if can:
            if out_rows is not None:
                out_rows[:] = rows
            for chunk in llm.answer_stream(query, context, source=source, history=history):
                yield chunk
            return

    # Web-search fallback.
    web_text, web_err = websearch.search(query, max_results=5)
    if web_err is not None and not web_text:
        yield f"网络搜索失败：{web_err}"
        return
    if web_text:
        for chunk in llm.answer_stream(query, web_text, source="网络搜索（AnySearch）", history=history):
            yield chunk
        return
    yield "网络搜索未返回结果。"


def rag_query_with_code(conn, tables, query, top_n=5):
    """RAG with an LLM-generated pandas code step; returns
    ``(answer, rows, code, code_result, err)``.
    """
    try:
        vec = llm.embed([query])[0]
    except Exception as exc:
        return "", [], "", "", f"RAG 问答出错: {exc}"

    results = search.hybrid_search(conn, tables, query, vec)
    results = _rerank_results(conn, query, results, top_n=top_n)
    rows = _build_hybrid_rows(results, lambda t, r: _fetch_row_by_id(conn, t, r))

    if not rows:
        return "", [], "", "", "未检索到相关数据。"

    df = pd.DataFrame(rows)
    df_preview = df.head().to_string()
    code = llm.generate_code(query, df_preview)
    _ok, code_result = code_exec.run_code(code, df)
    context = "\n".join(row["__row_text"] for row in rows)
    source = _db_source(tables, [row["__row_id"] for row in rows])
    answer = llm.answer(query, context, source=source)
    return answer, rows, code, code_result, None


def rag_query_with_review(conn, tables, query, top_n=5):
    """RAG with an answer-review step; returns
    ``(answer, rows, verdict, critique, err)``.
    """
    try:
        vec = llm.embed([query])[0]
    except Exception as exc:
        return "", [], False, "", f"RAG 问答出错: {exc}"

    results = search.hybrid_search(conn, tables, query, vec)
    results = _rerank_results(conn, query, results, top_n=top_n)
    rows = _build_hybrid_rows(results, lambda t, r: _fetch_row_by_id(conn, t, r))

    if not rows:
        return "", [], False, "", "未检索到相关数据。"

    context = "\n".join(row["__row_text"] for row in rows)
    source = _db_source(tables, [row["__row_id"] for row in rows])
    answer = llm.answer(query, context, source=source)
    verdict, critique = llm.review_answer(query, context, answer)
    return answer, rows, verdict, critique, None


def rag_query_decomposed(conn, tables, query, top_n=5):
    """Decompose complex questions into subqueries and synthesize an answer.

    Returns ``(answer, rows, err)``. A single subquery delegates to
    :func:`rag_query`; multiple subqueries are each retrieved and merged.
    """
    schemas = [db.get_schema(conn, t) for t in tables]
    subs = llm.decompose_question(query, schemas)

    if len(subs) == 1:
        return rag_query(conn, tables, subs[0], top_n)

    merged_rows = []
    context_parts = []
    seen = set()
    for sub in subs:
        try:
            vec = llm.embed([sub])[0]
        except Exception as exc:
            return "", [], f"RAG 问答出错: {exc}"
        results = search.hybrid_search(conn, tables, sub, vec)
        results = _rerank_results(conn, query, results, top_n=top_n)
        rows = _build_hybrid_rows(results, lambda t, r: _fetch_row_by_id(conn, t, r))
        for row in rows:
            key = (row["__table"], row["__row_id"])
            if key not in seen:
                seen.add(key)
                merged_rows.append(row)
        if rows:
            context_parts.append("子问题: " + sub + "\n上下文: " + "\n".join(r["__row_text"] for r in rows))

    if not merged_rows:
        return "", [], "未检索到相关数据。"

    ctx = "\n\n".join(context_parts)
    source = _db_source(tables, [r["__row_id"] for r in merged_rows])
    answer = llm.answer(query, ctx, source=source)
    return answer, merged_rows, None


def rag_query_dual(conn, tables, query, top_n=5):
    """Dual-path RAG: text retrieval *and* SQL + cross-validation.

    Returns ``(answer, rows, sql, sql_ctx, err)``.
    """
    try:
        vec = llm.embed([query])[0]
    except Exception as exc:
        return "", [], "", "", f"RAG 问答出错: {exc}"

    results = search.hybrid_search(conn, tables, query, vec)
    results = _rerank_results(conn, query, results, top_n=top_n)
    rows = _build_hybrid_rows(results, lambda t, r: _fetch_row_by_id(conn, t, r))

    sql, _cols, sql_rows, sql_err = ask_query(conn, tables, query, max_attempts=2)

    if rows:
        text_context = "\n".join(row["__row_text"] for row in rows)
        sql_ctx = json.dumps(sql_rows, ensure_ascii=False) if sql_rows is not None else str(sql_rows)
        answer = llm.cross_validate(query, sql_ctx, text_context)
        source = _db_source(tables, [row["__row_id"] for row in rows])
        answer = answer + "\n【来源：" + source + "】"
        return answer, rows, sql, sql_ctx, None

    if not sql_rows:
        return "", [], "", "", "未找到相关数据。"

    sql_ctx = json.dumps(sql_rows, ensure_ascii=False) if sql_rows is not None else str(sql_rows)
    return "", [], sql, sql_ctx, "未找到相关数据。"
