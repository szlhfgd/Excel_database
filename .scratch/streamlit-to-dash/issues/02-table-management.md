# 02: 表管理垂直切片（上传/列表/多选/删除）

**What to build:** 在脚手架基础上打通"数据进入应用"的完整路径：左侧 `dcc.Upload` 接收 Excel/CSV 字节流并调用 `ingest.ingest_file` 导入（处理中显示 `dbc.Spinner`），导入后表列表自动刷新；`dbc.Checklist` 多选参与搜索的表并存入 `dcc.Store`（默认全选）；`dbc.Select` + 按钮删除表后刷新列表。用户可端到端：上传文件→列表出现新表→勾选→删除成功。后端 `db/ingest` 复用不改。

**Blocked by:** 01 (Dash 脚手架与布局骨架)

**Status:** ready-for-agent

- [ ] `dcc.Upload` 接收 xlsx/xls/csv 字节，调用 `ingest.ingest_file`（必要时内存→临时文件）完成导入
- [ ] 导入中显示 `dbc.Spinner`，导入后左侧表列表自动刷新
- [ ] `dbc.Checklist` 列出所有表，默认全选，选中集合写入 `dcc.Store` 供查询模式读取
- [ ] 删除表控件（Select + 按钮）调用 `db.delete_table` 删除数据与向量表，删除后列表与 Store 刷新
- [ ] 端到端验证：上传示例文件→表出现并可勾选→删除→列表更新正确
