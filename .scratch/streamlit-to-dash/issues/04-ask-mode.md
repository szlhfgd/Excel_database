# 04: ask（NL2SQL）模式

**What to build:** 打通"自然语言转 SQL"的完整路径：模式切到 `ask` 时显示自然语言输入框；查询按钮把勾选表 schema（`db.get_schema`）与问题送 `llm.generate_sql`，执行结果；SQL 执行失败自动重试修正一次（共 2 次尝试上限）；生成的 SQL 以代码块展示在结果上方；结果用 `DataTable` 展示整行并复用下载。用户可端到端：提问→看到生成 SQL→返回匹配整行。

**Blocked by:** 02 (表管理垂直切片)

**Status:** ready-for-agent

- [ ] 模式切到 ask 时显隐对应输入区
- [ ] 读取 `dcc.Store` 中勾选表，调用 `llm.generate_sql(schemas, q)`
- [ ] SQL 执行异常时回传错误让 `llm.generate_sql(..., prev_error=...)` 修正，最多重试 1 次
- [ ] 生成的 SQL 在结果上方以代码块展示
- [ ] 结果 `DataTable` 展示整行，空结果给友好中文提示，复用 CSV 下载
- [ ] 端到端验证：自然语言提问→生成并展示 SQL→返回匹配整行
