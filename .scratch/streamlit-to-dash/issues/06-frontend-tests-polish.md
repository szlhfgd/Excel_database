# 06: 前端测试与体验打磨 + 全套验证

**What to build:** 在三个查询模式完毕后收口：完善 `tests/test_app.py` 冒烟断言（关键 `id` 与回调注册），补齐交互体验——未勾选任何表时主区提示"请先上传并勾选表"、导入/查询期间 `dbc.Spinner`、空结果友好中文文案；最后运行后端既有 `tests/` + 前端冒烟测试全套通过。

**Blocked by:** 03 (SQL 查询模式), 04 (ask 模式), 05 (hybrid 模式)

**Status:** ready-for-agent

- [ ] `tests/test_app.py` 断言布局含上传/表多选/模式切换/结果 DataTable/下载等关键 `id`
- [ ] 未勾选任何表时主区显示友好中文提示
- [ ] 导入与查询期间显示 `dbc.Spinner` 加载态
- [ ] 空结果（ask/sql/hybrid 无命中）显示友好中文提示而非报错
- [ ] 运行 `pytest` 全套（后端 test_db/test_ingest/test_search/test_llm + test_app）全部通过
