import json
import os
import urllib.request
from openai import OpenAI


def _client() -> OpenAI:
    return OpenAI(
        api_key=os.environ.get("SILICONFLOW_API_KEY", ""),
        base_url=os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        timeout=float(os.environ.get("LLM_TIMEOUT", "120")),
        max_retries=0,
    )


# Maximum characters sent to the embedding model per input. The SiliconFlow
# embedding API rejects over-long inputs with code 20015, so we truncate
# defensively. bge-m3's 8192-token context is exceeded well before 10k CJK
# characters; 4000 chars stays safely under the limit for both CJK and Latin text.
MAX_EMBED_CHARS = int(os.environ.get("EMBED_MAX_CHARS", "4000"))


def _sanitize_embed_inputs(texts: list) -> list[str]:
    """Coerce embed inputs to safe strings and truncate to ``MAX_EMBED_CHARS``.

    The API returns code 20015 ("parameter invalid") for non-string elements
    (None / float) and for inputs that exceed the model's token limit, so every
    element is normalized to a string and capped in length before being sent.
    """
    cleaned: list[str] = []
    for t in texts:
        if t is None:
            cleaned.append("")
        elif not isinstance(t, str):
            cleaned.append(str(t))
        else:
            cleaned.append(t)
    return [t[:MAX_EMBED_CHARS] for t in cleaned]


def embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
    safe = _sanitize_embed_inputs(texts)
    try:
        resp = _client().embeddings.create(model=model, input=safe)
    except Exception as exc:
        # SiliconFlow returns code 20015 for oversized batches or inputs that
        # still exceed the model's token limit after truncation.  Split the
        # batch in half and retry recursively so that a single problematic
        # item doesn't crash the entire import.
        if "20015" in str(exc) and len(safe) > 1:
            mid = len(safe) // 2
            return embed(safe[:mid]) + embed(safe[mid:])
        raise
    return [d.embedding for d in resp.data]


def generate_sql(table_schemas: list[dict], query: str, prev_error: str | None = None) -> str:
    model = os.environ.get("NL2SQL_MODEL", "deepseek-ai/DeepSeek-V3")
    schema_block = "\n\n".join(
        f"表 `{s['table']}`:\n列: " + ", ".join(f"{c} {t}" for c, t in s["columns"])
        + "\n样本行:\n" + "\n".join(str(r) for r in s["sample_rows"])
        for s in table_schemas
    )
    system = (
        "你是一个 SQLite 专家。根据用户自然语言问题，只输出一条可在 SQLite 执行的 SQL，"
        "不要解释，不要 markdown 代码块，只输出 SQL 本身。\n可用表结构：\n" + schema_block
    )
    user_msg = query
    if prev_error:
        user_msg = f"之前生成的 SQL 执行出错：{prev_error}\n请修正并只输出正确 SQL。\n原问题：{query}"
    resp = _client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
    )
    sql = resp.choices[0].message.content.strip()
    if sql.startswith("```"):
        sql = sql.strip("`")
        if sql.lower().startswith("sql"):
            sql = sql[3:]
    return sql.strip().rstrip(";")


RAG_SYSTEM_PROMPT = """你是一名知识库问答助手，请严格依据【参考上下文】的内容回答用户问题。
规则：
1. 回答问题只能使用参考上下文提供的信息，禁止编造、臆测不存在的内容；
2. 如果上下文没有答案，直接回复："根据现有知识库，未查询到相关信息"，不要强行回答；
3. 不要输出上下文里没有提到的数据、人名、结论；
4. 回答尽量简洁清晰，优先用原文事实，可适当改写，不要照搬大段原文；
5. 不要在回答里提及"参考上下文、知识库、文档"这类字眼；
6. 如果用户问题和上下文无关，直接告知无法解答。"""


def answer(question: str, context: str) -> str:
    """Answer *question* in natural language, grounded in *context* (retrieved rows)."""
    model = os.environ.get("RAG_MODEL", "deepseek-ai/DeepSeek-V3")
    system = RAG_SYSTEM_PROMPT + "\n\n【参考上下文】\n" + context + "\n【/参考上下文】"
    resp = _client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


def rerank(query: str, documents: list[str], top_n: int | None = None) -> list[float]:
    """Return a relevance score for each *document* (same order as input).

    Uses the SiliconFlow rerank API (bge-reranker-v2-m3 by default). Scores are
    aligned to *documents* by index so callers can zip them back together.
    """
    if not documents:
        return []
    api_key = os.environ.get("SILICONFLOW_API_KEY", "")
    if not api_key:
        raise RuntimeError("SILICONFLOW_API_KEY 未配置，无法执行 rerank。")
    model = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": top_n or len(documents),
        "return_input": False,
    }
    url = os.environ.get("SILICONFLOW_RERANK_URL", "https://api.siliconflow.cn/v1/rerank")
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=float(os.environ.get("LLM_TIMEOUT", "20"))) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    score_by_index = {r["index"]: float(r["score"]) for r in data.get("results", [])}
    return [score_by_index.get(i, 0.0) for i in range(len(documents))]


CODE_SYSTEM_PROMPT = """你是一名数据分析助手。用户会给你一个 pandas DataFrame `df`（已加载召回的数据行）和一个自然语言问题。
请只输出一段可在 Python 中执行的代码，对 `df` 进行计算来回答问题。
规则：
1. 只输出代码，不要解释，不要 markdown 代码块；
2. 代码必须基于变量 `df`（pandas DataFrame）进行计算；
3. 将最终答案赋值给变量 `result`（字符串或数字），或用 `print(...)` 输出；
4. 不要访问网络、文件或任何外部资源；
5. 可以使用的库：pandas（已导入为 pd）、numpy（已导入为 np）。"""


def generate_code(question: str, df_preview: str) -> str:
    """Generate python code (operating on a ``df`` DataFrame) to answer *question*."""
    model = os.environ.get("CODE_MODEL", "deepseek-ai/DeepSeek-V3")
    system = CODE_SYSTEM_PROMPT + "\n\n【df 预览（前几行）】\n" + df_preview
    resp = _client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
    )
    code = resp.choices[0].message.content.strip()
    if code.startswith("```"):
        code = code.strip("`")
        if code.lower().startswith("python"):
            code = code[6:]
    return code.strip()


REVIEW_SYSTEM_PROMPT = """你是一名严谨的审核员。给定【用户问题】、【参考上下文】和【待审核回答】，判断回答是否严格基于上下文、是否存在编造或事实错误。
只输出一行 JSON：{"verdict": "pass" 或 "fail", "reason": "简短说明"}。不要输出其他内容。"""


def review_answer(question: str, context: str, answer: str) -> tuple[bool, str]:
    """Review *answer* against *question* and *context*. Returns ``(pass, reason)``."""
    import json

    model = os.environ.get("RAG_MODEL", "deepseek-ai/DeepSeek-V3")
    user = (
        f"【用户问题】\n{question}\n\n"
        f"【参考上下文】\n{context}\n\n"
        f"【待审核回答】\n{answer}"
    )
    resp = _client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )
    content = resp.choices[0].message.content.strip()
    try:
        data = json.loads(content)
        verdict = str(data.get("verdict", "fail")).lower() == "pass"
        reason = str(data.get("reason", ""))
    except Exception:  # noqa: BLE001 - treat unparseable output as a failed review
        verdict = False
        reason = content
    return verdict, reason
