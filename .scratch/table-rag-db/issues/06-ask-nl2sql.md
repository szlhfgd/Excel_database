# 06: ask NL2SQL 模式

**What to build:** ask 模式用自然语言提问，系统调用硅基流动 DeepSeek-V3 生成 SQLite SQL（prompt 注入勾选表的 schema + 前 3 行样本），执行后整行结果在 Tab 展示；SQL 执行失败时把错误回传 LLM 自动修正一次（共 2 次尝试上限）。

**Blocked by:** 02（文件导入 + 侧边栏表管理）、04（SQL 模式端到端，复用执行与结果展示）

**Status:** ready-for-agent

- [ ] LLM 模块封装硅基流动 DeepSeek-V3 NL2SQL 调用
- [ ] prompt 包含勾选表的表名、列名、列类型及每表前 3 行样本
- [ ] SQL 执行异常时回传错误给 LLM 修正，最多重试 1 次
- [ ] ask 模式自然语言 → SQL → 整行结果在 Tab 展示
- [ ] 测试用 mock 替换硅基流动 HTTP，验证 prompt 构成与失败重试逻辑触发一次
