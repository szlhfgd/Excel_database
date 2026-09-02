"""Tests for the eval scaffold (eval.run_eval)."""
from unittest import mock

import eval as _eval


def _fake_rag_query(answer, rows=None, err=None):
    def _rag(conn, selected, question, top_n=5, recall_pool=50):
        return answer, rows or [], err
    return _rag


def _fake_rag_query_with_code(answer, rows=None, code="", code_result="", err=None):
    def _rag(conn, selected, question, top_n=5, recall_pool=50):
        return answer, rows or [], code, code_result, err
    return _rag


def test_run_eval_passing_case():
    cases = [{"question": "What is X?", "selected": ["t1"], "expected": "hello"}]
    with mock.patch("src.services.queries.rag_query", side_effect=_fake_rag_query("The answer is hello world")):
        result = _eval.run_eval(None, cases)
    assert result["total"] == 1
    assert result["passed"] == 1
    assert result["pass_rate"] == 1.0
    assert result["results"][0]["passed"] is True


def test_run_eval_failing_case():
    cases = [{"question": "What is X?", "selected": ["t1"], "expected": "goodbye"}]
    with mock.patch("src.services.queries.rag_query", side_effect=_fake_rag_query("The answer is hello")):
        result = _eval.run_eval(None, cases)
    assert result["passed"] == 0
    assert result["pass_rate"] == 0.0
    assert result["results"][0]["passed"] is False


def test_run_eval_error_case():
    cases = [{"question": "bad q", "selected": [], "expected": "x"}]
    with mock.patch("src.services.queries.rag_query", side_effect=_fake_rag_query("", err="LLM timeout")):
        result = _eval.run_eval(None, cases)
    assert result["passed"] == 0
    assert result["results"][0]["error"] == "LLM timeout"


def test_run_eval_case_insensitive():
    cases = [{"question": "q", "selected": [], "expected": "Hello"}]
    with mock.patch("src.services.queries.rag_query", side_effect=_fake_rag_query("the answer is HELLO")):
        result = _eval.run_eval(None, cases)
    assert result["passed"] == 1


def test_run_eval_empty_cases():
    result = _eval.run_eval(None, [])
    assert result == {"total": 0, "passed": 0, "pass_rate": 0.0, "results": []}


def test_run_eval_with_code():
    cases = [{"question": "count", "selected": ["t1"], "expected": "42"}]
    with mock.patch("src.services.queries.rag_query_with_code", side_effect=_fake_rag_query_with_code("result is 42", code="print(42)", code_result="42")):
        result = _eval.run_eval(None, cases, use_code=True)
    assert result["passed"] == 1
