import sqlite3
import uuid

import db
import app


def test_summarize_returns_row_count_and_column_stats():
    conn = db.get_conn()
    try:
        name = f"t_sum_{uuid.uuid4().hex[:8]}"
        conn.execute(
            f'CREATE TABLE "{name}" '
            f'(row_id INTEGER PRIMARY KEY AUTOINCREMENT, __row_text TEXT, "金额" REAL, "名称" TEXT)'
        )
        conn.execute(f'INSERT INTO "{name}" (__row_text, "金额", "名称") VALUES (?, ?, ?)', ("a 1", 10.0, "x"))
        conn.execute(f'INSERT INTO "{name}" (__row_text, "金额", "名称") VALUES (?, ?, ?)', ("b 2", 20.0, "y"))
        conn.execute(f'INSERT INTO "{name}" (__row_text, "金额", "名称") VALUES (?, ?, ?)', ("c 3", None, "x"))
        conn.commit()
        summary = db.summarize(conn, name)
        assert summary["row_count"] == 3
        by_name = {c["列名"]: c for c in summary["columns"]}
        assert by_name["金额"]["求和"] == 30.0
        assert by_name["金额"]["平均"] == 15.0
        assert by_name["金额"]["最小"] == 10.0
        assert by_name["金额"]["最大"] == 20.0
        assert by_name["金额"]["非空数"] == 2
        assert by_name["名称"]["去重数"] == 2
        assert by_name["名称"]["非空数"] == 3
    finally:
        conn.close()


def test_stats_query_returns_summary_for_selected(monkeypatch):
    fake = {
        "row_count": 5,
        "columns": [
            {"列名": "a", "类型": "INTEGER", "非空数": 5, "求和": 15, "平均": 3, "最小": 1, "最大": 5, "去重数": None}
        ],
    }
    monkeypatch.setattr(db, "summarize", lambda conn, name: fake)
    conn = sqlite3.connect(":memory:")
    summary, err = app.stats_query(conn, ["任意表"])
    assert err is None
    assert summary == fake


def test_stats_query_no_selection_returns_error():
    conn = sqlite3.connect(":memory:")
    summary, err = app.stats_query(conn, [])
    assert summary is None
    assert "请先勾选" in err
