import re
import jieba
from ..data import db
from rank_bm25 import BM25Okapi

RRF_K = 60

# Cache of per-table BM25 indexes. Keyed by table name; value is
# (signature, bm25, row_ids) where signature is a cheap SQL aggregate that
# changes whenever the table's row text changes (insert/delete/update), so the
# index is rebuilt lazily instead of on every query.
_BM25_CACHE: dict[str, tuple[tuple, "BM25Okapi", list[int]]] = {}


def _bm25_signature(conn: db.sqlite3.Connection, table: str) -> tuple:
    """Return a lightweight signature of *table*'s row text.

    Computed in SQL (no full row load into Python): row count, max row_id, and
    the total length of all __row_text values. Any insert/delete/update that
    changes row text alters at least one of these, invalidating the cache.
    """
    row = conn.execute(
        f'SELECT COUNT(*) AS c, COALESCE(MAX(row_id), 0) AS m, '
        f'COALESCE(SUM(length(__row_text)), 0) AS s FROM "{table}"'
    ).fetchone()
    return (row["c"], row["m"], row["s"])


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
    try:
        sig = _bm25_signature(conn, table)
    except db.sqlite3.OperationalError:
        # Table was dropped/recreated between calls — drop any stale cache.
        _BM25_CACHE.pop(table, None)
        return []
    cached = _BM25_CACHE.get(table)
    if cached is None or cached[0] != sig:
        rows = db.get_rows(conn, table)
        if not rows:
            return []
        corpus = [_tokenize(r["__row_text"]) for r in rows]
        bm25 = BM25Okapi(corpus)
        row_ids = [r["row_id"] for r in rows]
        _BM25_CACHE[table] = (sig, bm25, row_ids)
    else:
        _, bm25, row_ids = cached
    tokens = _tokenize(query)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    if k is not None:
        ranked = ranked[:k]
    return [(row_ids[i], float(scores[i])) for i in ranked if scores[i] > 0]


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


def select_tables(conn: db.sqlite3.Connection, question: str, k: int = 3, recall_pool: int = 20) -> list[str]:
    """Auto-pick the top-*k* most relevant tables for *question*.

    Runs the existing hybrid search across every user table and aggregates the
    RRF score by table, so the user doesn't have to select tables manually.
    Returns [] when there are no tables.
    """
    from ..ai import llm

    tables = db.list_tables(conn)
    if not tables:
        return []
    vec = llm.embed([question])[0]
    results = hybrid_search(conn, tables, question, vec, recall_pool=recall_pool)
    score_by_table: dict[str, float] = {}
    for table, _rid, score in results:
        score_by_table[table] = score_by_table.get(table, 0.0) + score
    ranked = sorted(score_by_table.items(), key=lambda x: x[1], reverse=True)
    return [t for t, _ in ranked[:k]]
