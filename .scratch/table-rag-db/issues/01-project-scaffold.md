# 01: 项目骨架�?SQLite 底座

**What to build:** 一个可运行的项目底座：安装全部依赖、建立密钥与忽略规则、提�?SQLite 连接与建�?读写原语，并用冒烟测试验证「能建一张表并读写行」。这是后续所有切片的地基，本身可独立验证�?
**Blocked by:** None (can start immediately)

**Status:** resolved

- [ ] `.venv` 中安装依赖：streamlit、pandas、sqlite-vec、rank-bm25、python-dotenv、openai、pytest
- [ ] 根目录存�?`.env`（含 `SILICONFLOW_API_KEY` 占位）与 `.gitignore`（忽�?`.env` �?`*.db`�?- [ ] 数据库模块可连接�?SQLite 文件、创建表、插入并读回�?- [ ] pytest 冒烟测试通过：建�?�?插入一�?�?读回该行断言一�?
