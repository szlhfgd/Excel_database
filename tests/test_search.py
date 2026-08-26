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
    res = search.hybrid_search(conn, [name], "apple", query_vec, k=10)
    assert res[0][1] == 1


def llm_embed():
    import llm
    return llm
