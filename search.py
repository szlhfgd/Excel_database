import re
import db
from rank_bm25 import BM25Okapi

RRF_K = 60


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text.lower())


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


def hybrid_search(conn: db.sqlite3.Connection, tables: list[str], query: str, query_vec: list[float], k: int | None = None) -> list[tuple[str, int, float]]:
    fused: dict[tuple[str, int], float] = {}
    for table in tables:
        for rank, (rid, _) in enumerate(_semantic_ranks(conn, table, query_vec, k)):
            fused[(table, rid)] = fused.get((table, rid), 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, (rid, _) in enumerate(_bm25_ranks(conn, table, query, k)):
            fused[(table, rid)] = fused.get((table, rid), 0.0) + 1.0 / (RRF_K + rank + 1)
    results = [(t, rid, score) for (t, rid), score in fused.items()]
    results.sort(key=lambda x: x[2], reverse=True)
    return results
