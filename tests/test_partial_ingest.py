import os
import tempfile
import uuid

import pytest

import pandas as pd
import db
import ingest
import llm


def _seed(conn, name, keys, vals):
    df = pd.DataFrame({"k": keys, "v": vals})
    db.create_table_from_df(conn, name, df, [f"r{k}" for k in keys])
    db.create_vec_table(conn, name)
    rows = db.get_rows(conn, name)
    db.upsert_embeddings(
        conn, name,
        [r["row_id"] for r in rows],
        [[0.0] * 1024 for _ in rows],
    )


def _write_xlsx(keys, vals):
    path = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False).name
    pd.DataFrame({"k": keys, "v": vals}).to_excel(path, index=False)
    return path


def test_ingest_file_update_mode_replaces_changed_rows(monkeypatch):
    name = "t_ing_" + uuid.uuid4().hex[:8]
    conn = db.get_conn()
    try:
        _seed(conn, name, [1, 2], ["a", "b"])
        path = _write_xlsx([1, 3], ["a2", "c"])

        def fake_embed(texts):
            return [[0.0] * 1024 for _ in texts]

        monkeypatch.setattr(llm, "embed", fake_embed)

        res_name, updated = ingest.ingest_file(
            conn, path, name=name + ".xlsx", key_col="k", mode="update"
        )
        assert res_name == name
        assert updated is True

        rows = db.get_rows(conn, name)
        assert {r["k"] for r in rows} == {1, 3}
    finally:
        conn.close()
        os.unlink(path)


def test_ingest_file_merge_mode_keeps_missing(monkeypatch):
    name = "t_mgi_" + uuid.uuid4().hex[:8]
    conn = db.get_conn()
    try:
        _seed(conn, name, [1, 2], ["a", "b"])
        path = _write_xlsx([1, 3], ["a2", "c"])

        def fake_embed(texts):
            return [[0.0] * 1024 for _ in texts]

        monkeypatch.setattr(llm, "embed", fake_embed)

        res_name, updated = ingest.ingest_file(
            conn, path, name=name + ".xlsx", key_col="k", mode="merge"
        )
        assert res_name == name

        rows = db.get_rows(conn, name)
        assert {r["k"] for r in rows} == {1, 2, 3}
    finally:
        conn.close()
        os.unlink(path)


def test_ingest_file_create_mode_creates_new_table(monkeypatch):
    name = "t_crt_" + uuid.uuid4().hex[:8]
    conn = db.get_conn()
    try:
        path = _write_xlsx([1, 2], ["a", "b"])

        def fake_embed(texts):
            return [[0.0] * 1024 for _ in texts]

        monkeypatch.setattr(llm, "embed", fake_embed)

        res_name, updated = ingest.ingest_file(
            conn, path, name=name + ".xlsx", mode="create"
        )
        assert res_name == name
        assert updated is False

        rows = db.get_rows(conn, name)
        assert {r["k"] for r in rows} == {1, 2}
    finally:
        conn.close()
        os.unlink(path)


def test_ingest_file_create_mode_refuses_existing(monkeypatch):
    name = "t_crt_" + uuid.uuid4().hex[:8]
    conn = db.get_conn()
    try:
        _seed(conn, name, [1, 2], ["a", "b"])
        path = _write_xlsx([1, 3], ["a2", "c"])

        def fake_embed(texts):
            return [[0.0] * 1024 for _ in texts]

        monkeypatch.setattr(llm, "embed", fake_embed)

        with pytest.raises(ValueError):
            ingest.ingest_file(
                conn, path, name=name + ".xlsx", mode="create"
            )
    finally:
        conn.close()
        os.unlink(path)
