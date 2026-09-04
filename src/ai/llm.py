import json
import os
import urllib.request
from openai import OpenAI, APIConnectionError, AuthenticationError, APITimeoutError


def _client() -> OpenAI:
    """Chat / NL2SQL / RAG client — routes to the JAC custom platform."""
    api_key = os.environ.get("JAC_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "模型服务未配置：JAC_API_KEY 为空。请检查 .env 是否存在、是否被加载（app.py 调用 load_dotenv()）。"
        )
    return OpenAI(
        api_key=api_key,
        base_url=os.environ.get("JAC_BASE_URL", "http://192.168.190.182:3000/v1"),
        timeout=float(os.environ.get("LLM_TIMEOUT", "120")),
        max_retries=0,
    )


def _sf_client() -> OpenAI:
    """Embedding client — routes to SiliconFlow for speed (JAC's bge-m3 is too slow)."""
    api_key = os.environ.get("SILICONFLOW_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "向量化服务未配置：SILICONFLOW_API_KEY 为空。请检查 .env 中的 SILICONFLOW_API_KEY 配置。"
        )
    return OpenAI(
        api_key=api_key,
        base_url=os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        timeout=float(os.environ.get("LLM_TIMEOUT", "120")),
        max_retries=0,
    )


# Maximum characters sent to the embedding model per input. The custom platform
# embedding API rejects over-long inputs with code 20015, so we truncate
# defensively. bge-m3's 8192-token context is exceeded well before 10k CJK
# characters; 4000 chars stays safely under the limit for both CJK and Latin text.
MAX_EMBED_CHARS = int(os.environ.get("EMBED_MAX_CHARS", "4000"))


def _truncate_token_safe(text: str) -> str:
    """Truncate *text* to ``MAX_EMBED_CHARS`` at a token (word) boundary.

    A naive ``text[:N]`` can slice mid-word / mid-sentence, degrading the
    embedding of the trailing fragment. For typical short row text (well under
    the limit) this returns the input untouched and costs nothing; only
    over-limit text pay for the jieba tokenization, which lets us stop at a
    whole-word boundary instead of splitting a word in half.
    """
    if len(text) <= MAX_EMBED_CHARS:
        return text
    import jieba

    tokens = jieba.lcut_for_search(text)
    out = ""
    for tok in tokens:
        if len(out) + len(tok) > MAX_EMBED_CHARS:
            break
        out += tok
    # jieba may over-split in ways that still leave a dangling char, so never
    # return something oddly short; fall back to a clean prefix if needed.
    if not out:
        out = text[:MAX_EMBED_CHARS]
    return out.strip()


def _sanitize_embed_inputs(texts: list) -> list[str]:
    """Coerce embed inputs to safe strings and truncate to ``MAX_EMBED_CHARS``.

    The API returns code 20015 ("parameter invalid") for non-string elements
    (None / float) and for inputs that exceed the model's token limit, so every
    element is normalized to a string and capped in length before being sent.
    Truncation is token-aware (:func:`_truncate_token_safe`).
    """
    cleaned: list[str] = []
    for t in texts:
        if t is None:
            cleaned.append("")
        elif not isinstance(t, str):
            cleaned.append(str(t))
        else:
            cleaned.append(t)
    return [_truncate_token_safe(t) for t in cleaned]


def embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
    safe = _sanitize_embed_inputs(texts)
    try:
        resp = _sf_client().embeddings.create(model=model, input=safe)
    except Exception as exc:
        # SiliconFlow returns code 20015 for oversized batches or inputs that
        # still exceed the model's token limit after truncation.  Split the
        # batch in half and retry recursively so that a single problematic
        # item doesn't crash the entire import.
        if "20015" in str(exc) and len(safe) > 1:
            mid = len(safe) // 2
            return embed(safe[:mid]) + embed(safe[mid:])
        # Translate low-level OpenAI SDK errors into clear Chinese messages so
        # the UI shows a readable st.error() instead of a cryptic traceback or
        # a bare "Connection lost." when the WebSocket dies before the error
        # surfaces. Only wrap credential / connectivity failures; let other
        # exceptions (e.g. 20015 above, programming errors) pass through.
        if isinstance(exc, AuthenticationError):
            raise RuntimeError(
                "向量化失败：API Key 无效或未授权（SILICONFLOW_API_KEY）。请检查 .env 中的 SILICONFLOW_API_KEY 配置。"
            ) from exc
        if isinstance(exc, (APITimeoutError, APIConnectionError)):
            raise RuntimeError(
                "向量化失败：无法连接 SiliconFlow 服务或请求超时。请检查网络和 SILICONFLOW_BASE_URL 配置。"
            ) from exc
        raise
    return [d.embedding for d in resp.data]


def _render_schema(s: dict) -> str:
    """Render one table schema for the NL2SQL prompt.

    Prefers per-column sample values (``column_samples``) when present —
    they help the model pick the right column when names are ambiguous.
    Falls back to whole-row ``sample_rows`` for backward compatibility.
    """
    header = f"表 `{s['table']}`:\n列: " + ", ".join(f"{c} {t}" for c, t in s["columns"])
    col_samples = s.get("column_samples") or {}
    lines = [f"{c}: {vals}" for c, vals in col_samples.items() if vals]
    if lines:
        return header + "\n列样例值:\n" + "\n".join(lines)
    return header + "\n样本行:\n" + "\n".join(str(r) for r in s["sample_rows"])


def generate_sql(table_schemas: list[dict], query: str, prev_error: str | None = None) -> str:
    model = os.environ.get("NL2SQL_MODEL", "deepseek_v4")
    schema_block = "\n\n".join(_render_schema(s) for s in table_schemas)
    system = (
        "你是一个 SQLite 专家。根据用户自然语言问题，只输出一条可在 SQLite 执行的 SQL，"
        "不要解释，不要 markdown 代码块，只输出 SQL 本身。\n"
        "硬性约束：只允许生成只读查询（SELECT / WITH / VALUES），严禁任何写操作"
        "（DELETE、UPDATE、DROP、INSERT、ALTER、CREATE、REPLACE、ATTACH、VACUUM）"
        "或多语句（分号分隔）。\n可用表结构：\n" + schema_block
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


DECOMPOSE_SYSTEM_PROMPT = """你是一个查询分解助手。判断用户问题是否需要拆分成多个子查询来回答。
规则：
1. 如果问题简单、可以直接回答，输出包含原问题的单元素 JSON 数组；
2. 如果问题复杂（涉及多步推理、多个条件、跨表或聚合计算），拆分成多个可独立求解的子查询；
3. 每个子查询必须是完整、自包含的自然语言问题，不依赖其他子查询的结果；
4. 只输出 JSON 数组，例如：["子查询1", "子查询2"]，不要输出其他内容。"""


def decompose_question(question: str, schemas: list[dict]) -> list[str]:
    """Decide whether *question* should be split into subqueries.

    Returns a list of subqueries (at least one). If the model output cannot be
    parsed, the original question is returned unchanged as a safe fallback.
    """
    model = os.environ.get("DECOMPOSE_MODEL", os.environ.get("RAG_MODEL", "deepseek_v4"))
    schema_block = "\n\n".join(
        f"表 `{s['table']}`:\n列: " + ", ".join(f"{c} {t}" for c, t in s["columns"])
        for s in schemas
    )
    system = DECOMPOSE_SYSTEM_PROMPT + "\n\n可用表结构：\n" + schema_block
    resp = _client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
    )
    content = (resp.choices[0].message.content or "").strip()
    content = content.strip("`")
    if content.lower().startswith("json"):
        content = content[4:].strip()
    try:
        data = json.loads(content)
        if isinstance(data, list):
            subs = [str(x).strip() for x in data if str(x).strip()]
            if subs:
                return subs
    except Exception:  # noqa: BLE001 - unparseable → fall back to the original question
        pass
    return [question]


RAG_SYSTEM_PROMPT = """你是一名知识库问答助手，请严格依据【参考上下文】的内容回答用户问题。
规则：
1. 回答问题只能使用参考上下文提供的信息，禁止编造、臆测不存在的内容；
2. 如果上下文没有答案，直接回复："根据现有知识库，未查询到相关信息"，不要强行回答；
3. 不要输出上下文里没有提到的数据、人名、结论；
4. 优先引用原文事实，在事实基础上适当展开思考、推理和解释，使回答完整、有深度，但不得偏离事实、不得编造；
5. 回答末尾另起一行标注信息来源，格式：【来源：...】；
6. 不要在回答正文里提及"参考上下文、知识库、文档"这类字眼；
7. 如果用户问题和上下文无关，直接告知无法解答。"""

# Max prior conversation turns injected into the streaming answer call. 10 turns
# bounds token usage for multi-turn chat while keeping conversational context.
MAX_HISTORY_TURNS = int(os.environ.get("RAG_MAX_HISTORY_TURNS", "10"))


def answer(question: str, context: str, source: str | None = None) -> str:
    """Answer *question* in natural language, grounded in *context* (retrieved rows).

    When *source* is provided (e.g. "数据库表格：t1、t2" or "网络搜索（AnySearch）"),
    the model is instructed to end the answer with a 【来源：...】 line.
    """
    model = os.environ.get("RAG_MODEL", "deepseek_v4")
    system = RAG_SYSTEM_PROMPT + "\n\n【参考上下文】\n" + context + "\n【/参考上下文】"
    if source:
        system += (
            f"\n\n本次回答的信息来源为：{source}。"
            f"请在回答末尾另起一行原样标注：【来源：{source}】，"
            f"不要删减其中的表名与行号。"
        )
    resp = _client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


def answer_stream(question: str, context: str, source: str | None = None, history: list[dict] | None = None):
    """Streaming variant of :func:`answer`. Yields the answer in text chunks.
    *history* is an optional list of prior ``{"role": "user"|"assistant",
    "content": ...}`` turns for multi-turn conversation. The current turn's
    retrieved *context* is injected for this question only; prior turns are
    sent as conversation history without their (now stale) contexts. History is
    capped to ``MAX_HISTORY_TURNS`` to bound token usage.
    """
    model = os.environ.get("RAG_MODEL", "deepseek_v4")
    system = RAG_SYSTEM_PROMPT + "\n\n【参考上下文】\n" + context + "\n【/参考上下文】"
    if source:
        system += (
            f"\n\n本次回答的信息来源为：{source}。"
            f"请在回答末尾另起一行原样标注：【来源：{source}】，"
            f"不要删减其中的表名与行号。"
        )
    messages: list[dict] = [{"role": "system", "content": system}]
    if history:
        messages.extend(history[-MAX_HISTORY_TURNS:])
    messages.append({"role": "user", "content": question})
    stream = _client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta


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
    model = os.environ.get("CODE_MODEL", "deepseek_v4")
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


CAN_ANSWER_SYSTEM_PROMPT = """你是一名判断助手。给定【用户问题】和【参考上下文】，判断参考上下文是否包含足够的信息来回答该问题。
- 如果上下文能直接回答该问题（包含所需的事实、数据或依据），输出 true；
- 如果上下文与问题无关、信息不足、或无法据此回答，输出 false。
只输出一行 JSON：{"can_answer": true 或 false, "reason": "简短说明"}。不要输出其他内容。"""


def can_answer(question: str, context: str) -> bool:
    """Return whether *context* contains enough information to answer *question*.

    Used by the RAG flow to decide whether to fall back to live web search when
    the retrieved database rows do not actually answer the user's question.
    """
    import json

    model = os.environ.get("RAG_MODEL", "deepseek_v4")
    user = f"【用户问题】\n{question}\n\n【参考上下文】\n{context}"
    resp = _client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CAN_ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )
    content = resp.choices[0].message.content.strip()
    try:
        return str(json.loads(content).get("can_answer", "false")).lower() == "true"
    except Exception:  # noqa: BLE001 - on parse failure, assume the DB context can answer
        return True


def review_answer(question: str, context: str, answer: str) -> tuple[bool, str]:
    """Review *answer* against *question* and *context*. Returns ``(pass, reason)``."""
    import json

    model = os.environ.get("RAG_MODEL", "deepseek_v4")
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


CROSS_VALIDATE_SYSTEM_PROMPT = """你是一名数据核对助手。给定【用户问题】、【SQL 执行结果】和【文本检索上下文】，综合判断并给出最终答案。
规则：
1. 如果 SQL 执行结果为空或报错，依据文本检索上下文回答；
2. 如果两者一致，直接输出一致的结果；
3. 如果两者冲突，仔细分析：数值/聚合类事实以 SQL 执行结果为准，文本上下文作为补充说明；
4. 只输出最终答案，不要解释过程，不要提及"SQL、上下文"等字眼。"""


def cross_validate(question: str, sql_result: str, text_context: str) -> str:
    """Arbitrate between an SQL execution result and text-retrieval context.

    SQL results are authoritative for numeric/aggregate facts; text context is
    the fallback when SQL is empty or errored. Returns the final answer.
    """
    model = os.environ.get("RAG_MODEL", "deepseek_v4")
    user = (
        f"【用户问题】\n{question}\n\n"
        f"【SQL 执行结果】\n{sql_result}\n\n"
        f"【文本检索上下文】\n{text_context}"
    )
    resp = _client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CROSS_VALIDATE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip()
