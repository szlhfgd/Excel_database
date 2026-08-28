import pandas as pd
import db


def test_col_letter():
    assert db.col_letter(1) == "A"
    assert db.col_letter(3) == "C"
    assert db.col_letter(26) == "Z"
    assert db.col_letter(27) == "AA"


def test_get_row_coords():
    conn = db.get_conn()
    try:
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})
        df["src_row"] = [2, 3]
        df["sheet"] = "Sheet1"
        db.create_table_from_df(conn, "t_coord", df, ["x", "y"])
        rows = db.get_rows(conn, "t_coord")
        coords = db.get_row_coords(conn, "t_coord", rows[0]["row_id"])
        coord_map = {c: v for c, col, v in coords}
        assert "Sheet1!A2" in coord_map
        assert coord_map["Sheet1!A2"] == "Alice"
        assert "Sheet1!B2" in coord_map
        assert coord_map["Sheet1!B2"] == 30
    finally:
        conn.close()
