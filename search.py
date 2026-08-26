import re
import db
from rank_bm25 import BM25Okapi

RRF_K = 60


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[一-鿿]|[a-zA-Z0-9]+", text.lower())


def _semantic_ranks(conn: db.sqlite3.Connection, table: str, query_vec: list[float], k: int) -> list[tuple[int, float]]:
    return db.vec_search(conn, table, query_vec, k=k)


def _bm25_ranks(conn: db.sqlite3.Connection, table: str, query: str, k: int) -> list[tuple[int, float]]:
    rows = db.get_rows(conn, table)
    corpus = [_tokenize(r["__row_text"]) for r in rows]
    if not corpus:
        return []
    bm25 = BM25Okapi(corpus)
    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens) if tokens else [0.0] * len(corpus)
    ranked = sorted(range(len(rows)), key=lambda i: scores[i], reverse=True)[:k]
    return [(rows[i]["row_id"], float(scores[i])) for i in ranked]


def hybrid_search(conn: db.sqlite3.Connection, tables: list[str], query: str, query_vec: list[float], k: int = 20) -> list[tuple[str, int, float]]:
    fused: dict[tuple[str, int], float] = {}
    for table in tables:
        sem = _semantic_ranks(conn, table, query_vec, k)
        for rank, (rid, _) in enumerate(sem):
            fused[(table, rid)] = fused.get((table, rid), 0.0) + 1.0 / (RRF_K + rank + 1)
        bm = _bm25_ranks(conn, table, query, k)
        for rank, (rid, _) in enumerate(bm):
            fused[(table, rid)] = fused.get((table, rid), 0.0) + 1.0 / (RRF_K + rank + 1)
    results = [(t, rid, score) for (t, rid), score in fused.items()]
    results.sort(key=lambda x: x[2], reverse=True)
    return results
