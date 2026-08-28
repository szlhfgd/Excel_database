import uuid

import pandas as pd
import db


def _df(keys, vals):
    return pd.DataFrame({"k": keys, "v": vals})


def _texts(df):
    return [f"k={k} v={v}" for k, v in zip(df["k"], df["v"])]


def test_upsert_rows_update_deletes_missing():
    conn = db.get_conn()
    try:
        name = "t_upd_" + uuid.uuid4().hex[:8]
        db.create_table_from_df(conn, name, _df([1, 2, 3], ["a", "b", "c"]), _texts(_df([1, 2, 3], ["a", "b", "c"])))

        df2 = _df([1, 2, 4], ["a2", "b", "d"])
        changed, deleted = db.upsert_rows(conn, name, df2, _texts(df2), "k", "update")

        rows = db.get_rows(conn, name)
        keys = {r["k"] for r in rows}
        assert keys == {1, 2, 4}
        assert 3 not in keys
        assert len(changed) == 3
        assert len(deleted) == 1
    finally:
        conn.close()


def test_upsert_rows_merge_keeps_missing():
    conn = db.get_conn()
    try:
        name = "t_mrg_" + uuid.uuid4().hex[:8]
        db.create_table_from_df(conn, name, _df([1, 2, 3], ["a", "b", "c"]), _texts(_df([1, 2, 3], ["a", "b", "c"])))

        df2 = _df([1, 4], ["a2", "d"])
        changed, deleted = db.upsert_rows(conn, name, df2, _texts(df2), "k", "merge")

        rows = db.get_rows(conn, name)
        keys = {r["k"] for r in rows}
        assert keys == {1, 2, 3, 4}
        assert deleted == []
        assert len(changed) == 2
    finally:
        conn.close()
