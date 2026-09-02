"""Lightweight eval scaffold for the RAG pipeline.

Run a list of Q&A cases through ``rag_query`` (or ``rag_query_with_code``) and
score each answer against an expected substring. Returns an aggregate summary
so the pipeline can be regression-checked without a full benchmark suite.
"""
from __future__ import annotations

from src.services import queries as _q


def _normalize(s: str) -> str:
    return (s or "").strip().lower()


def run_eval(
    conn,
    cases: list[dict],
    use_code: bool = False,
    top_n: int = 5,
) -> dict:
    """Run *cases* through the RAG pipeline and score answers.

    Each case: ``{"question": str, "selected": [table], "expected": str}``.
    A case passes when *expected* (lowercased) is a substring of the answer
    and no error occurred. Returns
    ``{"total", "passed", "pass_rate", "results": [...]}``.
    """
    results: list[dict] = []
    passed = 0
    for case in cases:
        question = case["question"]
        selected = case.get("selected", [])
        expected = case.get("expected", "")
        if use_code:
            answer, _rows, _code, _code_result, err = _q.rag_query_with_code(
                conn, selected, question, top_n=top_n
            )
        else:
            answer, _rows, err = _q.rag_query(conn, selected, question, top_n=top_n)
        ok = (err is None) and bool(expected) and _normalize(expected) in _normalize(answer)
        if ok:
            passed += 1
        results.append(
            {
                "question": question,
                "expected": expected,
                "answer": answer,
                "passed": ok,
                "error": err,
            }
        )
    total = len(cases)
    return {
        "total": total,
        "passed": passed,
        "pass_rate": (passed / total) if total else 0.0,
        "results": results,
    }
