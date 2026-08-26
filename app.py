import os
import io
import tempfile
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import db
import ingest
import llm
import search

st.set_page_config(page_title="电子表格数据库", layout="wide")


def get_conn():
    return db.get_conn()


def render_sidebar(conn):
    st.sidebar.title("📊 电子表格数据库")
    uploaded = st.sidebar.file_uploader("上传 Excel / CSV", type=["xlsx", "xls", "csv"])
    if uploaded is not None and st.sidebar.button("导入"):
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(uploaded.read())
            tmp = f.name
        with st.spinner("导入并构建向量索引..."):
            name = ingest.ingest_file(conn, tmp)
        st.sidebar.success(f"已导入：{name}")
        os.remove(tmp)

    tables = db.list_tables(conn)
    selected = st.sidebar.multiselect("选择参与搜索的表", tables, default=tables)
    st.session_state["selected"] = selected

    if selected:
        del_name = st.sidebar.selectbox("删除表", [""] + selected)
        if del_name and st.sidebar.button("删除选中表"):
            db.delete_table(conn, del_name)
            st.sidebar.success(f"已删除：{del_name}")
            st.rerun()
    return selected


def fetch_rows(conn, table, row_ids):
    rows = db.get_rows(conn, table)
    by_id = {r["row_id"]: r for r in rows}
    return [by_id[i] for i in row_ids if i in by_id]


def run_mode(conn, selected):
    mode = st.radio("查询模式", ["hybrid", "ask", "sql"], horizontal=True,
                    help="hybrid=语义+BM25 融合；ask=自然语言转SQL；sql=手写SQL")
    if mode == "sql":
        sql = st.text_area("SQL", "SELECT * FROM 表名")
        if st.button("执行") and sql.strip():
            try:
                rows = conn.execute(sql).fetchall()
                df = [dict(r) for r in rows]
            except Exception as e:
                st.error(f"执行出错：{e}")
                return
            show_results(df)
    elif mode == "ask":
        q = st.text_input("用自然语言提问")
        if st.button("查询") and q:
            schemas = [db.get_schema(conn, t) for t in selected]
            sql = llm.generate_sql(schemas, q)
            for attempt in range(2):
                try:
                    rows = conn.execute(sql).fetchall()
                    break
                except Exception as e:
                    if attempt == 0:
                        sql = llm.generate_sql(schemas, q, prev_error=str(e))
                    else:
                        st.error(f"SQL 执行失败：{e}")
                        return
            df = [dict(r) for r in rows]
            if not df:
                st.info("未找到匹配行")
                return
            st.code(sql, language="sql")
            show_results(df)
    else:
        q = st.text_input("混合搜索（关键词或短语）")
        if st.button("搜索") and q:
            vec = llm.embed([q])[0]
            res = search.hybrid_search(conn, selected, q, vec, k=None)
            st.write(f"命中 {len(res)} 行")
            for table, rid, score in res:
                with st.expander(f"{table} · 行{rid} · 分数{score:.4f}"):
                    row = fetch_rows(conn, table, [rid])[0]
                    st.json({k: v for k, v in row.items() if k != "__row_text"})


def show_results(df: list[dict]) -> None:
    if not df:
        st.info("未找到匹配行")
        return
    st.dataframe(df)
    st.download_button("下载 CSV", _to_csv(df), "result.csv", "text/csv")


def _to_csv(df):
    buf = io.StringIO()
    pd.DataFrame(df).to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue().encode("utf-8-sig")


def main():
    conn = get_conn()
    selected = render_sidebar(conn)
    st.title("🔍 查询")
    if not selected:
        st.warning("请先在左侧上传文件并勾选表")
        return
    run_mode(conn, selected)


if __name__ == "__main__":
    main()
