import os
from unittest import mock
import pandas as pd

os.environ["SPREADSHEET_DB"] = ":memory:"
from src.data import db
from src.services import search


def test_hybrid_ranks_semantic_neighbor_first():
    conn = db.get_conn()
    df = pd.DataFrame({"t": ["apple fruit", "banana fruit", "car vehicle"]})
    name = db.create_table_from_df(conn, "h.xlsx", df, ["apple fruit", "banana fruit", "car vehicle"])
    vec1 = [1.0] + [0.0] * (db.EMBED_DIM - 1)
    vec2 = [0.0] + [0.0] * (db.EMBED_DIM - 1)
    db.create_vec_table(conn, name)
    db.upsert_embeddings(conn, name, [1, 2, 3], [vec1, vec2, vec2])
    query_vec = [0.9] + [0.0] * (db.EMBED_DIM - 1)
    res = search.hybrid_search(conn, [name], "apple", query_vec, recall_pool=10)
    assert res[0][1] == 1


def llm_embed():
    from src.ai import llm
    return llm


def test_tokenize_cjk_and_latin():
    # 中文词级切分（jieba），拉丁/数字按词切分
    assert search._tokenize("Hello 世界 123") == ["hello", "世界", "123"]
    assert search._tokenize("智能驾驶") == ["智能", "驾驶"]
    assert search._tokenize("比亚迪") == ["比亚", "比亚迪"]


def test_tokenize_lowercases():
    assert search._tokenize("ABC") == ["abc"]


def test_tokenize_empty():
    assert search._tokenize("") == []


def _make_table(conn, texts):
    df = pd.DataFrame({"t": texts})
    name = db.create_table_from_df(conn, "s.xlsx", df, texts)
    return name


def test_bm25_ranks_excludes_zero_score_and_empty_corpus():
    conn = db.get_conn()
    name = _make_table(conn, ["apple banana fruit", "car dog vehicle", "apple car red"])
    res = search._bm25_ranks(conn, name, "apple", k=None)
    row_ids = [r[0] for r in res]
    assert set(row_ids) == {1, 3}
    assert 2 not in row_ids

    empty = db.create_table_from_df(conn, "e.xlsx", pd.DataFrame({"t": []}), [])
    assert search._bm25_ranks(conn, empty, "x", k=None) == []

    assert search._bm25_ranks(conn, name, "!!!$$$", k=None) == []


def test_bm25_ranks_respects_k():
    conn = db.get_conn()
    name = _make_table(conn, ["apple banana fruit", "car dog vehicle", "apple car red"])
    res = search._bm25_ranks(conn, name, "apple", k=1)
    assert len(res) <= 1


def test_hybrid_ranks_jointly_strong_doc_first():
    conn = db.get_conn()
    name = _make_table(conn, ["apple fruit red", "banana fruit yellow", "car vehicle blue"])
    vec1 = [1.0] + [0.0] * (db.EMBED_DIM - 1)
    vec2 = [0.0] + [0.0] * (db.EMBED_DIM - 1)
    db.create_vec_table(conn, name)
    db.upsert_embeddings(conn, name, [1, 2, 3], [vec1, vec2, vec2])
    query_vec = [0.9] + [0.0] * (db.EMBED_DIM - 1)
    res = search.hybrid_search(conn, [name], "apple", query_vec, recall_pool=10)
    assert res[0][1] == 1


def test_hybrid_fusion_beats_single_signal():
    conn = db.get_conn()
    # row 1: top in BOTH semantic (nearest vec) and BM25 (query token "apple")
    # row 2: present in both but lower ranked
    # row 3: only in semantic (no "apple" token) -> single signal
    name = _make_table(conn, ["apple fruit red", "apple fruit green", "zebra stripe"])
    vec1 = [1.0] + [0.0] * (db.EMBED_DIM - 1)
    vec2 = [0.0, 1.0] + [0.0] * (db.EMBED_DIM - 2)
    vec3 = [0.0] * db.EMBED_DIM
    db.create_vec_table(conn, name)
    db.upsert_embeddings(conn, name, [1, 2, 3], [vec1, vec2, vec3])
    query_vec = [0.9, 0.1] + [0.0] * (db.EMBED_DIM - 2)
    res = search.hybrid_search(conn, [name], "apple", query_vec, recall_pool=10)
    scores = {rid: score for _, rid, score in res}
    assert scores[1] > scores[2] > scores[3]


def test_hybrid_respects_k():
    conn = db.get_conn()
    name = _make_table(conn, ["apple fruit red", "banana fruit yellow", "car vehicle blue"])
    vec1 = [1.0] + [0.0] * (db.EMBED_DIM - 1)
    vec2 = [0.0] + [0.0] * (db.EMBED_DIM - 1)
    db.create_vec_table(conn, name)
    db.upsert_embeddings(conn, name, [1, 2, 3], [vec1, vec2, vec2])
    query_vec = [0.9] + [0.0] * (db.EMBED_DIM - 1)
    res = search.hybrid_search(conn, [name], "apple", query_vec, recall_pool=1)
    assert len(res) <= 2


def test_select_tables_returns_top_k_by_relevance(monkeypatch):
    # Two tables: "fruits" matches the question via BM25 token "apple",
    # "cars" does not. select_tables should rank fruits first.
    conn = db.get_conn()
    fruits = db.create_table_from_df(
        conn, "fruits.xlsx", pd.DataFrame({"t": ["apple fruit", "banana fruit"]}), ["apple fruit", "banana fruit"]
    )
    cars = db.create_table_from_df(
        conn, "cars.xlsx", pd.DataFrame({"t": ["toyota car", "honda car"]}), ["toyota car", "honda car"]
    )
    # Real tables always have vec tables; create them with zero vectors so the
    # semantic leg contributes nothing and BM25 token "apple" decides ranking.
    for t in (fruits, cars):
        db.create_vec_table(conn, t)
        db.upsert_embeddings(conn, t, [1, 2], [[0.0] * db.EMBED_DIM, [0.0] * db.EMBED_DIM])
    monkeypatch.setattr("src.ai.llm.embed", lambda texts: [[0.0] * db.EMBED_DIM])
    picked = search.select_tables(conn, "apple", k=2, recall_pool=20)
    assert picked[0] == fruits
    assert set(picked) == {fruits, cars}


def test_select_tables_empty_db_returns_empty():
    conn = db.get_conn()
    assert search.select_tables(conn, "anything") == []


def test_bm25_cache_reused_across_queries():
    conn = db.get_conn()
    name = _make_table(conn, ["apple banana fruit", "car dog vehicle", "apple car red"])
    search._BM25_CACHE.clear()
    # First query builds the index and populates the cache.
    search._bm25_ranks(conn, name, "apple", k=None)
    assert name in search._BM25_CACHE
    sig, bm25, row_ids = search._BM25_CACHE[name]
    assert row_ids == [1, 2, 3]
    # Second query with unchanged data reuses the cached index (same object).
    search._bm25_ranks(conn, name, "car", k=None)
    assert search._BM25_CACHE[name][1] is bm25


def test_bm25_cache_invalidated_on_data_change():
    conn = db.get_conn()
    name = _make_table(conn, ["apple banana fruit", "car dog vehicle", "apple car red"])
    search._BM25_CACHE.clear()
    res = search._bm25_ranks(conn, name, "apple", k=None)
    assert {r[0] for r in res} == {1, 3}
    # Update row 1's text in place (same row count and max row_id) — the
    # signature must still change so the cache is rebuilt with fresh data.
    conn.execute(f'UPDATE "{name}" SET "__row_text"=? WHERE row_id=1', ("zebra stripe",))
    conn.commit()
    res = search._bm25_ranks(conn, name, "apple", k=None)
    assert {r[0] for r in res} == {3}
    assert 1 not in {r[0] for r in res}


def test_bm25_cache_cleared_on_drop():
    conn = db.get_conn()
    name = _make_table(conn, ["apple banana fruit"])
    search._BM25_CACHE.clear()
    search._bm25_ranks(conn, name, "apple", k=None)
    assert name in search._BM25_CACHE
    db.delete_table(conn, name)
    # Dropping the table must not raise; stale cache entry is removed.
    assert search._bm25_ranks(conn, name, "apple", k=None) == []
    assert name not in search._BM25_CACHE
