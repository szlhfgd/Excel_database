# 05: hybrid 混合搜索模式

**What to build:** 打通"语义 + BM25 融合搜索"的完整路径：模式切到 `hybrid` 时显示搜索输入框；查询按钮调用 `llm.embed([q])` + `search.hybrid_search`（RRF 融合），结果用 `dash.dash_table.DataTable` 列出命中行的表名/行号/分数/摘要文本；点击某行（`active_cell`）在下方折叠区展示该整行去 `__row_text` 后的完整 JSON。用户可端到端：搜索→命中按 RRF 排序→点开看整行全部字段。

**Blocked by:** 02 (表管理垂直切片)

**Status:** ready-for-agent

- [ ] 模式切到 hybrid 时显隐对应输入区
- [ ] 查询调用 `llm.embed([q])` 与 `search.hybrid_search(conn, selected, q, vec, k=None)`
- [ ] `DataTable` 列出表名/行号/分数/摘要（命中计数提示）
- [ ] `active_cell` 点击回调从 `db.get_rows` 取完整行，下方折叠区展示去 `__row_text` 的 JSON
- [ ] 端到端验证：搜索返回 RRF 排序命中，点击行可下钻完整 JSON
