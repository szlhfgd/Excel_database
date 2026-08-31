import sqlite3

import pytest

from execute_sql import ReadOnlySQLViolation, assert_readonly_sql


# ---- assert_readonly_sql: allow-list --------------------------------------


@pytest.mark.parametrize("sql", [
    "SELECT * FROM t",
    "SELECT a, b FROM t WHERE x > 1 ORDER BY a",
    "WITH RECURSIVE x(n) AS (SELECT 1) SELECT * FROM x",
    "VALUES (1, 2)",
    "PRAGMA table_info(t)",
    "EXPLAIN SELECT * FROM t",
    "  select * from t  ",
])
def test_allows_readonly_statements(sql):
    assert assert_readonly_sql(sql) == sql


@pytest.mark.parametrize("sql", [
    "DROP TABLE t",
    "drop table t",
    "DELETE FROM t",
    "UPDATE t SET a = 1",
    "INSERT INTO t (a) VALUES (1)",
    "REPLACE INTO t (a) VALUES (1)",
    "ALTER TABLE t ADD COLUMN b",
    "CREATE TABLE x (a INT)",
    "ATTACH DATABASE 'x' AS y",
    "VACUUM",
    "REINDEX t",
])
def test_rejects_blocked_keywords(sql):
    with pytest.raises(ReadOnlySQLViolation):
        assert_readonly_sql(sql)


@pytest.mark.parametrize("sql", [
    "SELECT * FROM t; DROP TABLE t",
    "SELECT * FROM t; SELECT * FROM u",
    "DROP TABLE t;",
])
def test_rejects_multi_statement(sql):
    with pytest.raises(ReadOnlySQLViolation):
        assert_readonly_sql(sql)


def test_comment_smuggled_keyword_is_blocked():
    # A ``;`` hidden behind a line comment must still be caught as multi-statement.
    with pytest.raises(ReadOnlySQLViolation):
        assert_readonly_sql("SELECT * FROM t; -- DROP TABLE u")


def test_allow_trailing_semicolon():
    assert assert_readonly_sql("SELECT * FROM t;") == "SELECT * FROM t;"


@pytest.mark.parametrize("sql", [
    "SELECT load_extension('x')",
    "SELECT writefile('/tmp/x', 'y')",
])
def test_rejects_dangerous_functions(sql):
    with pytest.raises(ReadOnlySQLViolation):
        assert_readonly_sql(sql)


def test_empty_and_comment_only_rejected():
    for sql in ("", "   ", "-- just a comment", "/* block */"):
        with pytest.raises(ReadOnlySQLViolation):
            assert_readonly_sql(sql)


def test_allowed_tables_restriction():
    sql = "SELECT * FROM t"
    assert assert_readonly_sql(sql, allowed_tables={"t"}) == sql
    with pytest.raises(ReadOnlySQLViolation):
        assert_readonly_sql(sql, allowed_tables={"u"})
    with pytest.raises(ReadOnlySQLViolation):
        assert_readonly_sql("SELECT * FROM t JOIN other ON 1", allowed_tables={"t"})


# ---- app integration: _run_query rejects writes ----------------------------


def test_run_query_rejects_destructive_sql(monkeypatch):
    import app

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (a INT)")

    cols, rows, err = app.sql_query(conn, "DELETE FROM t")
    assert err is not None and "只读" in err
    assert cols == []
    assert rows == []

    # Trying to mutate via a hand-written statement must not change data.
    conn.execute("INSERT INTO t VALUES (1)")
    conn.commit()
    app.sql_query(conn, "UPDATE t SET a = 99")
    assert conn.execute("SELECT a FROM t").fetchone()[0] == 1
