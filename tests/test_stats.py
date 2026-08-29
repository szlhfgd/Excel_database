import sqlite3
import uuid

import pytest
import db
import app


# ---------------------------------------------------------------------------
# Existing tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# New: db.column_value_counts
# ---------------------------------------------------------------------------


def test_column_value_counts_returns_top_n():
    conn = db.get_conn()
    try:
        name = f"t_vcounts_{uuid.uuid4().hex[:8]}"
        conn.execute(
            f'CREATE TABLE "{name}" '
            f'(row_id INTEGER PRIMARY KEY AUTOINCREMENT, __row_text TEXT, "类别" TEXT)'
        )
        for val in ["A", "A", "A", "B", "B", "C", None]:
            conn.execute(f'INSERT INTO "{name}" (__row_text, "类别") VALUES (?, ?)', ("row", val))
        conn.commit()
        result = db.column_value_counts(conn, name, "类别", limit=2)
        assert len(result) == 2
        assert result[0] == ("A", 3)
        assert result[1] == ("B", 2)
    finally:
        conn.close()


def test_column_value_counts_skips_nulls():
    conn = db.get_conn()
    try:
        name = f"t_vnull_{uuid.uuid4().hex[:8]}"
        conn.execute(
            f'CREATE TABLE "{name}" '
            f'(row_id INTEGER PRIMARY KEY AUTOINCREMENT, __row_text TEXT, "标签" TEXT)'
        )
        conn.execute(f'INSERT INTO "{name}" (__row_text, "标签") VALUES (?, ?)', ("r1", None))
        conn.execute(f'INSERT INTO "{name}" (__row_text, "标签") VALUES (?, ?)', ("r2", None))
        conn.commit()
        result = db.column_value_counts(conn, name, "标签")
        assert result == []
    finally:
        conn.close()


def test_column_value_counts_limit_one():
    conn = db.get_conn()
    try:
        name = f"t_vlim_{uuid.uuid4().hex[:8]}"
        conn.execute(
            f'CREATE TABLE "{name}" '
            f'(row_id INTEGER PRIMARY KEY AUTOINCREMENT, __row_text TEXT, "城市" TEXT)'
        )
        for city in ["北京", "上海", "北京", "广州", "北京"]:
            conn.execute(f'INSERT INTO "{name}" (__row_text, "城市") VALUES (?, ?)', ("r", city))
        conn.commit()
        result = db.column_value_counts(conn, name, "城市", limit=1)
        assert len(result) == 1
        assert result[0] == ("北京", 3)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# New: db.numeric_bins
# ---------------------------------------------------------------------------


def test_numeric_bins_basic():
    conn = db.get_conn()
    try:
        name = f"t_nbins_{uuid.uuid4().hex[:8]}"
        conn.execute(
            f'CREATE TABLE "{name}" '
            f'(row_id INTEGER PRIMARY KEY AUTOINCREMENT, __row_text TEXT, "值" REAL)'
        )
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            conn.execute(f'INSERT INTO "{name}" (__row_text, "值") VALUES (?, ?)', ("r", v))
        conn.commit()
        result = db.numeric_bins(conn, name, "值", bins=5)
        # 5 values 1-5 into 5 bins: each bin should have 1
        assert len(result) == 5
        total = sum(cnt for _, cnt in result)
        assert total == 5
    finally:
        conn.close()


def test_numeric_bins_skips_nulls():
    conn = db.get_conn()
    try:
        name = f"t_nnull_{uuid.uuid4().hex[:8]}"
        conn.execute(
            f'CREATE TABLE "{name}" '
            f'(row_id INTEGER PRIMARY KEY AUTOINCREMENT, __row_text TEXT, "得分" REAL)'
        )
        conn.execute(f'INSERT INTO "{name}" (__row_text, "得分") VALUES (?, ?)', ("r", 10.0))
        conn.execute(f'INSERT INTO "{name}" (__row_text, "得分") VALUES (?, ?)', ("r", None))
        conn.execute(f'INSERT INTO "{name}" (__row_text, "得分") VALUES (?, ?)', ("r", 20.0))
        conn.commit()
        result = db.numeric_bins(conn, name, "得分", bins=2)
        total = sum(cnt for _, cnt in result)
        assert total == 2  # only non-null
    finally:
        conn.close()


def test_numeric_bins_single_value():
    """When all values are the same, return one bin."""
    conn = db.get_conn()
    try:
        name = f"t_nsingle_{uuid.uuid4().hex[:8]}"
        conn.execute(
            f'CREATE TABLE "{name}" '
            f'(row_id INTEGER PRIMARY KEY AUTOINCREMENT, __row_text TEXT, "固定" REAL)'
        )
        for _ in range(3):
            conn.execute(f'INSERT INTO "{name}" (__row_text, "固定") VALUES (?, ?)', ("r", 7.0))
        conn.commit()
        result = db.numeric_bins(conn, name, "固定", bins=10)
        assert len(result) == 1
        assert result[0] == ("7.0", 3)
    finally:
        conn.close()


def test_numeric_bins_all_null():
    conn = db.get_conn()
    try:
        name = f"t_nallnull_{uuid.uuid4().hex[:8]}"
        conn.execute(
            f'CREATE TABLE "{name}" '
            f'(row_id INTEGER PRIMARY KEY AUTOINCREMENT, __row_text TEXT, "空列" REAL)'
        )
        conn.execute(f'INSERT INTO "{name}" (__row_text, "空列") VALUES (?, ?)', ("r", None))
        conn.commit()
        result = db.numeric_bins(conn, name, "空列")
        assert result == []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# New: app.build_stats_data
# ---------------------------------------------------------------------------


def test_build_stats_data_keys():
    conn = db.get_conn()
    try:
        name = f"t_build_{uuid.uuid4().hex[:8]}"
        conn.execute(
            f'CREATE TABLE "{name}" '
            f'(row_id INTEGER PRIMARY KEY AUTOINCREMENT, __row_text TEXT, '
            f'"销量" INTEGER, "价格" REAL, "产品" TEXT)'
        )
        conn.execute(
            f'INSERT INTO "{name}" (__row_text, "销量", "价格", "产品") VALUES (?, ?, ?, ?)',
            ("row1", 10, 5.5, "苹果"),
        )
        conn.execute(
            f'INSERT INTO "{name}" (__row_text, "销量", "价格", "产品") VALUES (?, ?, ?, ?)',
            ("row2", 20, 8.0, "香蕉"),
        )
        conn.execute(
            f'INSERT INTO "{name}" (__row_text, "销量", "价格", "产品") VALUES (?, ?, ?, ?)',
            ("row3", None, None, "苹果"),
        )
        conn.commit()
        data = app.build_stats_data(conn, name, bins=5, top_n=5)
        assert "numeric_bins" in data
        assert "text_top_n" in data
        assert "missing" in data
        assert "numeric_compare" in data
        # 销量 and 价格 are numeric
        assert "销量" in data["numeric_bins"]
        assert "价格" in data["numeric_bins"]
        # 产品 is text
        assert "产品" in data["text_top_n"]
        assert data["text_top_n"]["产品"] == [("苹果", 2), ("香蕉", 1)]
        # Missing values
        miss_by_name = {m["列名"]: m for m in data["missing"]}
        assert miss_by_name["销量"]["缺失数"] == 1
        assert miss_by_name["销量"]["填充率"] == pytest.approx(66.7, abs=0.1)
        # Numeric compare
        nc_names = [r["列名"] for r in data["numeric_compare"]]
        assert "销量" in nc_names
        assert "价格" in nc_names
    finally:
        conn.close()


def test_build_stats_data_empty_table():
    conn = db.get_conn()
    try:
        name = f"t_empty_{uuid.uuid4().hex[:8]}"
        conn.execute(
            f'CREATE TABLE "{name}" '
            f'(row_id INTEGER PRIMARY KEY AUTOINCREMENT, __row_text TEXT, "A" TEXT)'
        )
        conn.commit()
        data = app.build_stats_data(conn, name)
        # Column A has 0 non-null rows, so it's excluded from text_top_n / numeric_bins
        assert "A" not in data["text_top_n"]
        assert data["numeric_bins"] == {}
        assert data["missing"][0]["缺失数"] == 0
    finally:
        conn.close()
