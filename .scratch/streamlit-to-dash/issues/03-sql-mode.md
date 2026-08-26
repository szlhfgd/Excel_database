# 03: SQL 查询模式

**What to build:** 打通"手写 SQL 直接执行"的完整路径：右侧主区 `dbc.RadioItems` 切到 `sql` 模式时显示 `dbc.Textarea`；执行按钮调用 `db` 连接运行 SQL，结果用 `dash.dash_table.DataTable` 展示整行全部列；提供 `dcc.Download` 下载 UTF-8-SIG CSV；执行异常用 `dbc.Alert` 红色提示。用户可端到端：手写 `SELECT` → 返回行 → 下载 CSV。每个回调内 `db.get_conn()` + `close()` 保证线程安全。

**Blocked by:** 02 (表管理垂直切片)

**Status:** ready-for-agent

- [ ] 模式 RadioItems 切到 sql 时显隐对应输入区（dbc.Collapse 或条件渲染）
- [ ] Textarea 收 SQL，执行回调用 `db.get_conn()` 新建连接、`close()` 收尾
- [ ] 结果用 `DataTable` 展示整行全部列
- [ ] `dcc.Download` 下载结果为 UTF-8-SIG CSV（复用 `_to_csv` 逻辑）
- [ ] SQL 执行异常捕获并以红色 Alert 提示，不崩溃
- [ ] 端到端验证：手写 SELECT 返回行并可下载 CSV
