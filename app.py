"""电子表格数据库 — Streamlit single-page app.

The UI layer. All framework-free query / analysis logic lives in
:mod:`src.services.queries`; :func:`main` only wires Streamlit widgets to those
functions. All ``st.*`` calls live inside :func:`main`.
"""

from __future__ import annotations

import json
import os
import tempfile

from dotenv import load_dotenv

load_dotenv()

from src.services.queries import (
    ask_query,
    build_stats_data,
    category_counts,
    hybrid_query,
    ingest_file,
    list_tables,
    rag_query,
    rag_query_decomposed,
    rag_query_dual,
    rag_query_stream,
    rag_query_with_code,
    sql_query,
    stats_query,
    _fetch_row_by_id,
    _row_display_json,
    _row_has_visible_data,
    _to_csv,
)

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def main() -> None:
    import streamlit as st
    from src.data import db as _db
    from src.services import search as _search
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
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []  # list[{"role", "content"}]
    if "rag_dual_sql" not in st.session_state:
        st.session_state.rag_dual_sql = None
    if "rag_dual_ctx" not in st.session_state:
        st.session_state.rag_dual_ctx = None
    if "auto_used_tables" not in st.session_state:
        st.session_state.auto_used_tables = None
    if "_result_mode_active" not in st.session_state:
        st.session_state._result_mode_active = "hybrid"

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
            st.caption(f":gray[共 {len(tables)} 张表]" + (f" · :violet[已选 {len(selected)} 张]" if selected else ""))

    # ---- sidebar ---------------------------------------------------------
    with st.sidebar:
        st.header(":violet[数据管理]", divider=False)

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

      # 切换查询模式时清空上一次查询的结果，避免不同模式的结果互相串扰；
      # 结果区会在当前模式真正查询到新结果后才显示。
      if st.session_state.get("_result_mode_active") != mode:
          for _k in ("result_rows", "result_mode", "ask_sql", "rag_answer",
                     "rag_code", "rag_code_result", "rag_dual_sql", "rag_dual_ctx",
                     "show_detail_row", "auto_used_tables"):
              st.session_state[_k] = [] if _k == "result_rows" else None
          st.session_state._result_mode_active = mode

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
              st.caption(f":green[共 {len(result_rows)} 条结果]")

      # -- ask mode ----------------------------------------------------------
      elif mode == "ask":
          question = st.text_area("用自然语言提问", placeholder="例如：找出金额大于 100 的记录", height=80)
          if st.button("提问", type="primary", icon=":material/help:"):
              if not question or not question.strip():
                  st.warning("请输入问题。")
              elif not selected and not auto_select:
                  st.warning("请先在上方选择至少一个参与查询的表，或开启自动选择表。")
              else:
                  conn = _db.get_readonly_conn()
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
                  conn = _db.get_readonly_conn()
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
          # 多轮对话（流式输出）仅用于纯 rag_query 路径；分解/交叉验证/代码
          # 解释器为分阶段多调用，继续使用原有的单轮非流式交互。
          use_legacy = use_decompose or use_dual or use_code_interpreter

          if use_legacy:
              question = st.text_area("用自然语言提问（基于数据库内容回答）", placeholder="例如：哪些客户的金额超过 100？", height=80)
              if st.button("问答", type="primary", icon=":material/chat:"):
                  if not question or not question.strip():
                      st.warning("请输入问题。")
                  elif not selected and not auto_select:
                      st.warning("请先在上方选择至少一个参与查询的表，或开启自动选择表。")
                  else:
                      conn = _db.get_readonly_conn()
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
                          st.session_state.rag_messages = []  # 非流式子模式不走多轮对话
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
          else:
              # ---- 纯智能问答：多轮会话 + 流式输出 ----
              if st.button("清空对话", icon=":material/delete_sweep:", help="清除当前会话的历史问答记录"):
                  st.session_state.rag_messages = []
                  st.session_state.rag_answer = None
                  st.session_state.result_rows = []
                  st.session_state.result_mode = None
                  st.rerun()

              # 渲染历史对话
              for msg in st.session_state.rag_messages:
                  with st.chat_message(msg["role"]):
                      st.markdown(msg["content"])

              prompt = st.chat_input("输入问题（基于数据库内容回答）…")
              if prompt:
                  if not selected and not auto_select:
                      st.warning("请先在上方选择至少一个参与查询的表，或开启自动选择表。")
                  else:
                      st.chat_message("user").markdown(prompt)
                      st.session_state.rag_messages.append({"role": "user", "content": prompt})
                      history = list(st.session_state.rag_messages)  # 已在 llm.answer_stream 内裁剪到上限
                      st.session_state.auto_used_tables = None
                      conn = _db.get_readonly_conn()
                      _out_rows: list = []
                      try:
                          with st.chat_message("assistant"):
                              with st.spinner("检索中…"):
                                  use_tables = _search.select_tables(conn, prompt, k=3) if auto_select else selected
                              st.session_state.auto_used_tables = use_tables if auto_select else None
                              # 首次迭代会执行阻塞检索（填充 _out_rows），随后流式输出最终答案。
                              answer_text = st.write_stream(
                                  rag_query_stream(conn, use_tables, prompt, history=history, top_n=top_n, out_rows=_out_rows)
                              )
                      finally:
                          conn.close()
                      st.session_state.rag_answer = answer_text
                      st.session_state.rag_messages.append({"role": "assistant", "content": answer_text})
                      st.session_state.result_rows = _out_rows
                      st.session_state.result_mode = "rag"
                      st.session_state.ask_sql = None
                      st.session_state.show_detail_row = None

      # -- stats mode --------------------------------------------------------
      elif mode == "统计":
          # ── 分类统计（按列计总个数 → 柱状图）──────────────────────────
          if selected:
              cat_table = selected[0]
              st.markdown("**分类统计（按列计总个数）**")
              conn_cat = _db.get_conn()
              try:
                  _cat_summary, _cat_err = stats_query(conn_cat, [cat_table])
              finally:
                  conn_cat.close()
              if _cat_err:
                  st.error(_cat_err)
              else:
                  cat_cols = [c["列名"] for c in _cat_summary["columns"]]
                  cat_col = st.selectbox("选择分类列", options=cat_cols, key="stats_cat_col")
                  cat_limit = int(st.number_input(
                      "最多显示类别数",
                      min_value=0, value=50, step=10, key="stats_cat_limit",
                      help="0 = 不限制，显示该列全部类别的个数。",
                  ))
                  conn_cat2 = _db.get_conn()
                  try:
                      cat_rows = category_counts(
                          conn_cat2, cat_table, cat_col,
                          limit=None if cat_limit == 0 else cat_limit,
                      )
                  finally:
                      conn_cat2.close()
                  if cat_rows:
                      st.caption(f":green[共 {len(cat_rows)} 个类别值]")
                      st.bar_chart(pd.DataFrame(cat_rows), x="值", y="数量", horizontal=True)
                      st.dataframe(cat_rows, hide_index=True, width="stretch")
                  else:
                      st.info("该列没有可统计的非空值。")
              st.divider()

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
          st.caption(f":blue[自动选择表：{'、'.join(auto_used)}]")
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
                  st.subheader(f"表格：:violet[{tbl}]", divider=False)
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
        st.subheader(":blue[数据预览]（各表前 5 行）", divider=False)
        if not selected:
            st.info("请先在上方选择参与搜索的表。")
        else:
            conn_pv = _db.get_conn()
            try:
                for tbl in selected:
                    _pv_cols, pv_rows = _db.get_preview(conn_pv, tbl, n=5)
                    st.markdown(f"**:blue[{tbl}]**")
                    if pv_rows:
                        st.dataframe(pv_rows, hide_index=True, key=f"preview_{tbl}")
                    else:
                        st.info(f"{tbl}：暂无数据。")
            finally:
                conn_pv.close()


if __name__ == "__main__":
    main()
