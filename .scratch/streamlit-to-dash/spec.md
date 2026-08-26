# Spec: 用 Dash 替换 Streamlit 重写前端

**Status:** ready-for-agent

> 来源：grilling 访谈设计树（2026-08-26），用户对 5 项决策全部选择"按推荐"。
> 后端模块 `db/ingest/llm/search` 保持不变；本次仅重写前端 `app.py` 为 Dash 应用。

## Problem Statement

现有前端基于 Streamlit，运行时在浏览器控制台会引入无关噪声，且 Streamlit 的「整脚本重跑」模型对需要精细交互状态（如表选择持久化、逐行展开、点击下钻）的体验不够可控。用户希望用 Dash（Plotly）重写前端，获得更明确的回调式交互模型、稳定的组件库（dash-bootstrap-components），并保持与现有后端完全一致的功能与行为。

## Solution

一个 **Dash 单页应用 + dash-bootstrap-components** 替换原 `app.py`，复用既有 `db/ingest/llm/search` 四个模块：

- **布局**：左侧 `dbc.Col` 控制栏（上传、表多选、删除表），右侧主区放模式切换与结果；用 bootstrap 栅格实现 Streamlit sidebar 的等价视觉。
- **导入**：`dcc.Upload` 接收 Excel/CSV 字节流（内存处理，不再落临时文件），调用 `ingest.ingest_file` 后自动刷新表列表与向量索引；处理中显示 `dbc.Spinner`。
- **三种查询模式**（用 `dbc.RadioItems` 切换，回调控制输入区显隐）：
  - `hybrid`：语义向量 + BM25 融合（RRF），结果用 `dash.dash_table.DataTable` 列出命中行的表名/行号/分数/行文本摘要；点击某行在下方折叠区展示完整 JSON。
  - `ask`：自然语言经硅基流动 DeepSeek-V3 转 SQL（NL2SQL），执行后返回整行；SQL 出错自动重试修正一次（2 次尝试上限）；生成的 SQL 以代码块展示在结果上方。
  - `sql`：用户手写 SQL 直接执行。
- **结果**：`DataTable` 展示整行全部列；提供「下载 CSV」按钮（`dcc.Download`）；空结果给友好中文提示。
- **管理**：左侧栏上传、勾选参与搜索的表（`dbc.Checklist`，默认全选，存于 `dcc.Store`）、删除表（`dbc.Select` + 按钮，删除后刷新）。
- **运行**：`python app.py` 启动 `app.run(debug=True, port=8050)`；移除 streamlit 依赖，新增 `requirements.txt`。

全部中文 UI，单用户本地运行，无认证。

## User Stories

1. As a 本地分析用户, I want 在 Dash 页面左侧上传 .xlsx 文件, so that 它能被导入为一张可搜索的表。
2. As a 本地分析用户, I want 上传 .csv 文件（编码自动识别 UTF-8/GBK）, so that 它也能被导入。
3. As a 本地分析用户, I want 导入时看到加载提示（spinner）, so that 我知道索引正在构建。
4. As a 本地分析用户, I want 导入后左侧表列表自动刷新, so that 我无需手动重载即可勾选新表。
5. As a 本地分析用户, I want 在左侧勾选参与搜索的表（可多选，默认全选）, so that 我能控制 hybrid/ask 的查询范围。
6. As a 本地分析用户, I want 删除某张表（同时删数据与向量索引）, so that 我能清理不再需要的数据，且删除后列表刷新。
7. As a 本地分析用户, I want 用 RadioItems 切换 hybrid / ask / sql 三种模式, so that 不同模式显示对应的输入控件。
8. As a 本地分析用户, I want 在 hybrid 模式输入关键词或短语, so that 我能同时获得语义相关与关键词命中的整行结果。
9. As a 本地分析用户, I want hybrid 结果按 RRF 融合排序, so that 最相关行排在最前。
10. As a 本地分析用户, I want hybrid 结果以表格列出（表名/行号/分数/摘要）, so that 我能快速浏览命中概况。
11. As a 本地分析用户, I want 点击 hybrid 某行后在下方查看完整 JSON, so that 我能看到整行全部字段。
12. As a 本地分析用户, I want 在 ask 模式用自然语言提问, so that 系统自动生成 SQL 并返回匹配整行。
13. As a 本地分析用户, I want ask 模式生成的 SQL 展示在结果上方, so that 我能核对系统生成的查询。
14. As a 本地分析用户, I want NL2SQL 执行失败时自动重试修正一次, so that 偶发错误不影响使用。
15. As a 本地分析用户, I want 在 sql 模式手写并执行 SQL, so that 我能做任意精细查询。
16. As a 本地分析用户, I want 查询结果（ask/sql）返回整行全部列, so that 我能看到完整上下文。
17. As a 本地分析用户, I want 把结果下载为 CSV, so that 我能离线使用或二次分析。
18. As a 本地分析用户, I want 搜不到/无匹配时看到友好中文提示, so that 不会因报错而困惑。
19. As a 本地分析用户, I want 未勾选任何表时主区提示先上传并勾选, so that 我知道该如何开始。
20. As a 开发者, I want 前端逻辑由 Dash 回调驱动（无整脚本重跑）, so that 交互状态更可控、易调试。
21. As a 开发者, I want `requirements.txt` 列出 dash / dash-bootstrap-components 及既有后端依赖、移除 streamlit, so that 安装可复现。
22. As a 开发者, I want 用 `python app.py` 启动应用（监听 8050）, so that 启动方式直观、可调试。

## Implementation Decisions

- **前端框架**：Dash（`dash`）+ `dash-bootstrap-components`（dbc）。布局用 `dbc.Container`/`Row`/`Col` 模拟左侧控制栏 + 右侧主区；不再使用 streamlit。
- **状态持久化**：勾选的表集合存于 `dcc.Store`（前端存储），上传/删除后回调更新；替代 Streamlit `st.session_state`。
- **文件上传**：`dcc.Upload` 直接接收上传字节，用 `io.BytesIO` 传给 `ingest.read_file` 等价路径（新增内存读取入口或复用 `ingest` 时把字节落临时文件路径），不再依赖 `tempfile` 落盘（仅必要时在 `ingest` 内临时落盘）。
- **模式切换**：`dbc.RadioItems`（hybrid/ask/sql）作为 `Input`，回调根据所选模式显隐对应输入区（`dbc.Collapse` 或条件渲染），并路由到不同后端调用。
- **hybrid 展示**：`dash.dash_table.DataTable` 列出 `[(table, row_id, score, 摘要文本)]`；`active_cell` 点击回调从 `db.get_rows` 取完整行，在下方 `dbc.Collapse`/`pre` 展示去 `__row_text` 后的 JSON。
- **ask 模式**：沿用 `llm.generate_sql` + 2 次重试；生成的 SQL 通过 `dash.dcc.Markdown` 代码块展示在 `DataTable` 上方；执行走 `db` 连接。
- **sql 模式**：`dbc.Textarea` 收 SQL，直接 `conn.execute`；异常捕获后 `dbc.Alert` 红色提示。
- **CSV 下载**：`dcc.Download` + `dcc.send_bytes` 回传 UTF-8-SIG 编码 CSV（复用 `_to_csv` 逻辑）。
- **数据库连接**：每个回调内通过 `db.get_conn()` 新建连接、用后 `conn.close()`，保证多线程回调线程安全；不再共享全局连接。
- **密钥**：沿用 `.env`（`python-dotenv`），不改动 `llm`/`db` 读取方式。
- **后端模块零改动**：`db.py`/`ingest.py`/`llm.py`/`search.py` 接口完全复用，本次只新增/替换前端文件。
- **入口**：`app.py` 末尾 `if __name__ == "__main__": app.run(debug=True, port=8050)`，移除原 streamlit `sys.argv` 守卫。

## Testing Decisions

- **测试哲学**：只测外部行为（给定上传文件/查询 → 期望结果集/顺序/UI 组件存在），不测 Dash 内部渲染实现。
- **既有后端测试**：`tests/` 下 `test_db/test_ingest/test_search/test_llm` 保持不变、继续适用（后端未改）。
- **前端测试 seams（新增）**：
  - **冒烟测试**：`tests/test_app.py` 导入 `app` 对象，断言其为 `dash.Dash` 实例、布局含关键 `id`（上传组件、表多选、模式 RadioItems、结果 DataTable、下载组件），即"应用能构建且回调已注册"。
  - **纯函数测试**：将 `_to_csv`、hybrid 结果摘要拼装等无 Dash 依赖的辅助函数保留为可单测纯函数，复用 `test_search` 风格对结果顺序做断言。
  - 不启动真实 HTTP 服务做端到端点击测试（超出本次范围，避免脆弱的集成测试）。
- **Prior art**：沿用 `tests/` 现有 pytest 范式与 backend 外部行为测试风格。

## Out of Scope

- 后端模块重构（db/ingest/llm/search 不变）。
- 新增查询模式或分析能力（仅迁移，不改功能集）。
- 多用户 / 登录认证。
- 端到端浏览器自动化测试（Playwright 等）。
- 表重命名、跨文件 JOIN 优化等原 spec 已列为 out-of-scope 的项。
- 结果导出 Excel（仅 CSV，与原 spec 一致）。

## Further Notes

- 依赖新增：`dash`、`dash-bootstrap-components`；移除 `streamlit`。其余 `pandas`、`openai`、`sqlite-vec`、`rank-bm25`、`python-dotenv` 沿用。
- `ingest.read_file` 当前按路径读取；为满足 `dcc.Upload` 字节流，可在 `ingest` 增加内存读取入口或于前端把字节写入临时文件后调用（实现时择线程安全方案）。
- 设计完全来自 grilling 确认的设计树，未引入额外未确认假设。
- 验收以"功能与原 Streamlit 版本逐项一致 + 冒烟测试通过"为准。
