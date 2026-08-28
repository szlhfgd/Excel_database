import os
from unittest import mock
import pandas as pd

os.environ["SPREADSHEET_DB"] = ":memory:"
import db
import search


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
    import llm
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
