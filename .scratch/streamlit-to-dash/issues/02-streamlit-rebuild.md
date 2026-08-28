# 02: 用 Streamlit 重建 UI

**What to build:** 把 `app.py` 从 Dash 改写回 Streamlit 单页应用，复用既有后端模块（`db` / `ingest` / `search` / `llm`）。`requirements.txt` 改为含 `streamlit`、移除 `dash` / `dash-bootstrap-components`。所有 `st.*` 调用只放在 `main()` 内，业务逻辑抽到模块级无框架函数以便单测。功能覆盖：上传导入（带进度）、左侧表多选、删除表、前 5 行数据预览；主区 `hybrid` / `ask` / `sql` 三种查询模式；结果 `st.dataframe` 行选中看详情 + CSV 下载。用户能 `streamlit run app.py` 启动并使用。

**Blocked by:** None (can start immediately)

**Status:** done

- [x] `requirements.txt` 列出 streamlit 及既有后端依赖，不含 dash / dash-bootstrap-components
- [x] `app.py` 重写为 Streamlit 单页应用，`main()` 入口，`streamlit run app.py` 可启动
- [x] 业务逻辑抽到模块级函数（`list_tables` / `delete_table` / `preview` / `ingest_file` / `sql_query` / `ask_query` / `hybrid_query` 等），无 `st.*`
- [x] 上传导入带进度条；左侧表多选（默认空）、删除表、前 5 行预览
- [x] 主区 `hybrid` / `ask` / `sql` 三种模式；结果 `st.dataframe` 行详情 + CSV 下载
- [x] `tests/test_app.py` 改写为逻辑函数测试 + `AppTest` 冒烟测试（断言无异常且含上传控件）
- [x] 全量 `pytest` 通过；`streamlit run app.py` 能在本机启动
