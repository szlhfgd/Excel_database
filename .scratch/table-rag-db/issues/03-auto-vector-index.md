# 03: 导入即自动向量索�?
**What to build:** 导入完成即自动构建语义检索所需的向量索引：封装硅基流动 bge-m3 embedding 调用，将每行所有字段拼接为文本转成向量，存�?sqlite-vec 向量表（�?row_id 关联）；用测试验证向量写�?读回往返一致�?
**Blocked by:** 02（文件导�?+ 侧边栏表管理�?
**Status:** resolved

- [ ] embedding 模块封装硅基流动 bge-m3（OpenAI 兼容接口），读取 `.env` 中的 key
- [ ] 每行文本 = 各列「列�?值」拼接，整体�?embedding
- [ ] 导入流程在写表后自动写入对应向量（row_id �?embedding）到 sqlite-vec 向量�?- [ ] pytest 验证：给定若干行，embedding 写入后按 row_id 读回向量与原始一致（或余弦一致）
