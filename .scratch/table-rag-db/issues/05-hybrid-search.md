# 05: hybrid 混合搜索

**What to build:** hybrid 模式输入查询词/短语，系统同时做语义向量检索与 BM25 关键词检索，用 RRF 融合两路排名、按 row_id 去重，输出排序后的整行结果并在 Tab 展示，且遵守侧边栏勾选的表作用域。

**Blocked by:** 02（文件导入 + 侧边栏表管理）、03（导入即自动向量索引）

**Status:** ready-for-agent

- [ ] search 模块用 rank_bm25 对所有候选行做关键词打分
- [ ] search 模块用 sqlite-vec 做语义向量相似度检索
- [ ] 两路结果按 RRF（k=60）融合排序、按 row_id 去重
- [ ] hybrid 模式在界面返回融合排序的整行结果（Tab 展示），仅作用于勾选表
- [ ] 测试验证：注入已知行集，关键词精确行排名靠前、去重正确
