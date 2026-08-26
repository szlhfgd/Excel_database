# Spec: 电子表格数据库（TableRAG 风格本地版）

> 状态：待发布到 issue tracker（需先运行 `/setup-matt-pocock-skills`）
> 来源：通过 grilling 访谈与用户达成的一致设计树（2026-08-26）

## Problem Statement

用户需要一套本地运行的「电子表格数据库」工具：把多个 Excel / CSV 文件导入后统一管理，既能用自然语言或手写 SQL 做精确查询，也能在表格里做语义 + 关键词的混合搜索，搜索命中后整行返回，并通过浏览器 UI 完成导入、查询、导出全流程。现有 TableRAG 依赖 MySQL + 本地 embedding 服务，部署重、难以本地单用户直接使用。本工具是其轻量本地化实现。

## Solution

一个 Streamlit 单页应用 + 四个 Python 模块组成本地工具：

- 导入：上传 Excel/CSV，自动清洗、推断类型，按文件名建表，导入后自动用 bge-m3 把每行拼接文本转向量存入 SQLite（sqlite-vec 扩展）。
- 搜索：单一查询输入框 + 模式切换按钮，三种模式：
  - `hybrid`：语义向量检索 + BM25 关键词检索，RRF 融合排序，返回整行。
  - `ask`：自然语言经硅基流动 DeepSeek-V3 转 SQL（NL2SQL），执行后返回整行；SQL 出错自动重试修正一次（有上限）。
  - `sql`：用户手写 SQL 直接执行。
- 结果：分两个 Tab 展示（hybrid 结果 / SQL 结果），可下载 CSV；空结果给友好提示。
- 管理：侧边栏上传、勾选参与搜索的表、删除表。

全部为中文 UI，单用户本地运行，无认证。

## User Stories

1. As a 本地分析用户, I want 上传一个 .xlsx 文件, so that 它能被导入为一张可搜索的表。
2. As a 本地分析用户, I want 上传一个 .csv 文件, so that 它也能被导入（编码自动识别 UTF-8/GBK）。
3. As a 本地分析用户, I want 导入时自动清洗空行并推断列类型, so that 后续查询无需手动整理。
4. As a 本地分析用户, I want 表名自动取文件名（去扩展名）, so that 我无需每次命名。
5. As a 本地分析用户, I want 重新导入同名文件时覆盖旧数据, so that 数据始终是最新版。
6. As a 本地分析用户, I want 导入后自动构建向量索引, so that 上传完即可直接搜索。
7. As a 本地分析用户, I want 在侧边栏看到所有已导入的表, so that 我知道有哪些数据可用。
8. As a 本地分析用户, I want 勾选参与搜索的表（可多选，默认全选）, so that 我能控制 hybrid/ask 的查询范围。
9. As a 本地分析用户, I want 删除某张表（同时删数据与向量索引）, so that 我能清理不再需要的数据。
10. As a 本地分析用户, I want 在 hybrid 模式下输入关键词或短语, so that 我能同时获得语义相关与关键词命中的整行结果。
11. As a 本地分析用户, I want hybrid 结果按语义与 BM25 融合排序（RRF）, so that 最相关行排在最前。
12. As a 本地分析用户, I want 在 ask 模式下用自然语言提问（如「销售额大于1000的行」）, so that 系统自动生成 SQL 并返回匹配整行。
13. As a 本地分析用户, I want ask 模式能在勾选的多张表里让 LLM 自行选表/连表, so that 多表查询无需我手写 JOIN。
14. As a 本地分析用户, I want NL2SQL 生成的 SQL 执行失败时自动重试修正一次, so that 偶发错误不影响使用。
15. As a 本地分析用户, I want 在 sql 模式下手写并执行 SQL, so that 我能做任意精细查询。
16. As a 本地分析用户, I want 搜索/查询结果返回整行全部列, so that 我能看到完整上下文。
17. As a 本地分析用户, I want 结果不限制返回行数, so that 我能拿到全部命中数据。
18. As a 本地分析用户, I want hybrid 结果与 SQL 结果分 Tab 展示, so that 我能对比两种模式的产出。
19. As a 本地分析用户, I want 把结果下载为 CSV, so that 我能离线使用或二次分析。
20. As a 本地分析用户, I want 搜不到时看到「未找到匹配行」之类的友好提示, so that 不会因红错而困惑。
21. As a 本地分析用户, I want API key 放在 .env 文件里（不入库、不提交）, so that 凭证安全且配置方便。
22. As a 本地分析用户, I want 用硅基流动的统一 OpenAI 兼容接口做 NL2SQL 与 embedding, so that 我只配置一个 provider。
23. As a 单用户, I want 应用无需登录即可使用, so that 本地自用零摩擦。
24. As a 开发者, I want 代码按 app/db/search/llm/ingest 分模块, so that 职责清晰易维护。
25. As a 开发者, I want 依赖仅含 streamlit/pandas/sqlite-vec/rank-bm25/python-dotenv/openai, so that 安装轻量。

## Implementation Decisions

- **存储引擎**：SQLite 单文件数据库；向量通过 `sqlite-vec` 扩展同库存储（已确认 cp314 wheel 可用，版本 0.1.9）。
- **表模型**：每个导入文件 = 一张表，表名 = 文件名去扩展名；原始列原样保留（经清洗）。额外维护一张向量表，存 `row_id` + embedding（bge-m3, 1024 维）。
- **导入清洗**：pandas 读取后去空行、统一列名（去空格/特殊字符）；类型推断交 pandas/numpy。Excel 用 `read_excel`，CSV 编码依次试 UTF-8 → GBK/GB18030。
- **Embedding 输入**：每行所有字段拼接为一段文本（列名:值 拼接），整体送 bge-m3。
- **hybrid 检索**：
  - 语义路：sqlite-vec 向量相似度（cosine）取 top-k 候选。
  - 关键词路：`rank_bm25` 对所有行做 BM25（查询词分词后打分）。
  - 融合：RRF（Reciprocal Rank Fusion, `score = Σ 1/(k+rank)`, k=60）合并两路排名，去重按 `row_id`，输出排序整行。
- **ask（NL2SQL）**：
  - 调用硅基流动 DeepSeek-V3（OpenAI 兼容 `/chat/completions`）。
  - Prompt 注入：勾选表的表名 + 列名 + 列类型 + 每表前 3 行样本数据；要求只输出可执行 SQLite SQL。
  - 容错：SQL 执行异常时，把错误回传 LLM 让其修正，最多重试 1 次（含首次共 2 次尝试上限）。
- **sql 模式**：文本框直收 SQL，交由 SQLite 执行，结果原样返回。
- **结果层**：hybrid 与 SQL 各自结果集全量返回（不截断）；UI 分两个 `st.tabs` 展示；每 Tab 顶部「下载 CSV」按钮（用 `st.download_button`）。
- **空结果**：结果为空时显示友好中文文案，不抛异常。
- **密钥管理**：`python-dotenv` 读取项目根 `.env` 的 `SILICONFLOW_API_KEY`；`.env` 须加入 `.gitignore`，绝不入库/日志。
- **LLM 客户端**：`openai` SDK 配置 `base_url` 指向硅基流动兼容端点；NL2SQL 用 DeepSeek-V3，embedding 用 `BAAI/bge-m3`。
- **前端布局**：Streamlit 左 `st.sidebar` 放文件上传器 + 表勾选 `st.multiselect` + 删除表按钮；主区域放模式 `st.radio`（hybrid/ask/sql）+ 查询 `st.text_input`/`st.text_area` + 结果 Tabs。UI 文案全中文。
- **模块划分**：
  - `ingest.py`：读文件、清洗、建表、构建 embedding、写入向量表。
  - `db.py`：SQLite 连接、建表 DDL、行 CRUD、向量表读写（sqlite-vec 加载）。
  - `search.py`：BM25（`rank_bm25`）、语义检索（sqlite-vec 查询）、RRF 融合。
  - `llm.py`：硅基流动客户端封装（NL2SQL 生成 + embedding 调用 + 错误重试）。
  - `app.py`：Streamlit UI、模式路由、调用上述模块、渲染结果。

## Testing Decisions

- **测试哲学**：只测外部行为（给定输入文件/查询 → 期望结果集/顺序），不测模块内部实现细节。
- **测试范围**：
  - `ingest`：喂入样例 xlsx/csv（含 GBK 编码）→ 断言表已建、行数正确、向量表有对应 embedding。
  - `search`：注入已知行集 → 断言 hybrid 返回顺序合理、RRF 融合去重正确；BM25 对关键词精确行排名靠前。
  - `llm`：用 mock 替换硅基流动 HTTP 调用 → 断言 NL2SQL prompt 含 schema+样本、断言执行失败重试逻辑触发一次。
  - `db`：断言 sqlite-vec 向量表读写往返一致。
- **测试框架**：引入 `pytest` 作为开发依赖（仓库当前无测试框架）；在 `.venv` 安装，测试文件放 `tests/`。
- **Prior art**：仓库目前无测试先例，按上述外部行为测试范式新建。

## Out of Scope

- 多用户 / 登录认证 / 权限隔离。
- MySQL / PostgreSQL 后端（仅 SQLite）。
- 实时流式 embedding 服务（用硅基流动 API 而非本地模型）。
- 表重命名（仅删除 + 重新导入覆盖）。
- 跨文件 JOIN 的复杂查询优化（依赖 LLM 自行生成，不保证）。
- 结果导出 Excel（仅 CSV）。
- 大规模（百万行级）性能优化与分块。

## Further Notes

- 依赖已验证可在 Python 3.14 解析安装：streamlit 1.62, pandas 3.0.5, sqlite-vec 0.1.9, rank-bm25 0.2.2, python-dotenv 1.2.3, openai 3.3.1。
- `.env` 与 `*.db` 应加入 `.gitignore`；密钥绝不出现在代码或日志。
- 设计完全来自 grilling 访谈确认的设计树，未引入额外未确认假设。
