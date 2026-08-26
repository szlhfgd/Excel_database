import os
from openai import OpenAI


def _client() -> OpenAI:
    return OpenAI(
        api_key=os.environ.get("SILICONFLOW_API_KEY", ""),
        base_url=os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
    )


def embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
    resp = _client().embeddings.create(model=model, input=texts)
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
