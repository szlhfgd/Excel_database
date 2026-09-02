# AGENTS.md

Spreadsheet-Database: a Streamlit app that imports Excel/CSV into SQLite + a
vector store, then supports hybrid search, NL2SQL, RAG Q&A and stats. Entry
point is `app.py` (Streamlit UI); the framework-free query logic lives in
`src/services/queries.py`. Modules are grouped into a `src/` package by
layer: `services/` (orchestration), `data/` (SQLite access), `ai/` (LLM,
code exec, web search, SQL safety).

## Run

```
streamlit run app.py
```

Tests: `pytest`.

## Notes

- `.venv/` holds the deps; install with `pip install -r requirements.txt`.
- `.idea/` is the PyCharm project config — do not modify manually.

## Agent skills

### Issue tracker

Issues and specs live as local markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context layout: `CONTEXT.md` at root + `docs/adr/` for ADRs. See `docs/agents/domain.md`.

### Content

```
StrutData_db/
├── app.py          # Streamlit 主应用（仅 UI 层，调用 src.services）
├── eval.py         # 问答评测工具（导入 src.services.queries）
├── src/
│   ├── services/   # 业务编排层（混合搜索、NL2SQL、RAG、统计、导入、检索）
│   │   ├── queries.py    # 查询/分析编排（无 st.* 依赖）
│   │   ├── ingest.py     # 文件导入与向量化
│   │   └── search.py     # 混合检索（BM25 + 向量 + RRF 融合）
│   ├── data/       # 数据访问层
│   │   └── db.py         # SQLite + 向量表（建表、读写）
│   └── ai/         # 模型与外部能力层
│       ├── llm.py        # 大模型调用（向量化、重排、NL2SQL、RAG 回答）
│       ├── code_exec.py   # 代码解释器沙箱
│       ├── websearch.py   # 网络搜索兜底
│       └── execute_sql.py # 只读 SQL 校验
├── 启动.cmd        # Windows 一键启动脚本
├── requirements.txt
├── specs/
├── tests/          # pytest 测试
└── docs/
```
