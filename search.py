import re
import jieba
import db
from rank_bm25 import BM25Okapi

RRF_K = 60


def _tokenize(text: str) -> list[str]:
    """中文按 jieba 词级切分（而非逐字），英文/数字按词切分。

    逐字切分会让 BM25 对中文几乎失效（查"驾驶"命中所有含"驾"的行），
    且拆散专有名词（"比亚迪"→四字零区分度）。词级切分后 BM25 的中文
    区分度才成立。
    """
    tokens: list[str] = []
    for seg in re.split(r"([\u4e00-\u9fff]+)", text.lower()):
        if not seg:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", seg):
            tokens.extend(jieba.lcut_for_search(seg))
        else:
            tokens.extend(re.findall(r"[a-z0-9]+", seg))
    return tokens


def _semantic_ranks(conn: db.sqlite3.Connection, table: str, query_vec: list[float], k: int | None) -> list[tuple[int, float]]:
    return db.vec_search(conn, table, query_vec, k=k)


def _bm25_ranks(conn: db.sqlite3.Connection, table: str, query: str, k: int | None) -> list[tuple[int, float]]:
    rows = db.get_rows(conn, table)
    if not rows:
        return []
    corpus = [_tokenize(r["__row_text"]) for r in rows]
    bm25 = BM25Okapi(corpus)
    tokens = _tokenize(query)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    ranked = sorted(range(len(rows)), key=lambda i: scores[i], reverse=True)
    if k is not None:
        ranked = ranked[:k]
    return [(rows[i]["row_id"], float(scores[i])) for i in ranked if scores[i] > 0]


def hybrid_search(conn: db.sqlite3.Connection, tables: list[str], query: str, query_vec: list[float], recall_pool: int = 50) -> list[tuple[str, int, float]]:
    fused: dict[tuple[str, int], float] = {}
    for table in tables:
        for rank, (rid, _) in enumerate(_semantic_ranks(conn, table, query_vec, recall_pool)):
            fused[(table, rid)] = fused.get((table, rid), 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, (rid, _) in enumerate(_bm25_ranks(conn, table, query, recall_pool)):
            fused[(table, rid)] = fused.get((table, rid), 0.0) + 1.0 / (RRF_K + rank + 1)
    results = [(t, rid, score) for (t, rid), score in fused.items()]
    results.sort(key=lambda x: x[2], reverse=True)
    return results
