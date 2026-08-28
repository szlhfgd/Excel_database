import os
from unittest import mock
import llm


def test_generate_sql_strips_fences_and_includes_schema():
    import llm
    captured = {}

    class FakeMsg:
        def __init__(self, content):
            self.content = content

    class FakeChoice:
        def __init__(self, content):
            self.message = FakeMsg(content)

    class FakeResp:
        def __init__(self, content):
            self.choices = [FakeChoice(content)]

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeResp("```sql\nSELECT * FROM `t`\n```")

    class FakeEmbed:
        def create(self, **kwargs):
            return type("R", (), {"data": [type("D", (), {"embedding": [0.0] * 1024})()]})()

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()
        embeddings = type("E", (), {"create": FakeEmbed().create})()

    with mock.patch.object(llm, "_client", return_value=FakeClient()):
        sql = llm.generate_sql(
            [{"table": "t", "columns": [("a", "INTEGER")], "sample_rows": [{"a": 1}]}],
            "全部",
        )
    assert sql == "SELECT * FROM `t`"
    assert "t" in captured["messages"][0]["content"]
    assert "全部" in captured["messages"][1]["content"]


def test_generate_sql_passes_prev_error():
    import llm
    captured = {}

    class FakeMsg:
        def __init__(self, content):
            self.content = content

    class FakeChoice:
        def __init__(self, content):
            self.message = FakeMsg(content)

    class FakeResp:
        def __init__(self, content):
            self.choices = [FakeChoice(content)]

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeResp("SELECT 1")

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    with mock.patch.object(llm, "_client", return_value=FakeClient()):
        llm.generate_sql([{"table": "t", "columns": [], "sample_rows": []}], "q", prev_error="no such col")
    assert "no such col" in captured["messages"][1]["content"]


def test_embed_empty_returns_empty_without_calling_client():
    called = {"n": 0}

    class FakeEmbed:
        def create(self, **kwargs):
            called["n"] += 1
            return type("R", (), {"data": []})()

    class FakeClient:
        embeddings = type("E", (), {"create": FakeEmbed().create})()

    with mock.patch.object(llm, "_client", return_value=FakeClient()):
        assert llm.embed([]) == []
    assert called["n"] == 0


def test_embed_passes_input_and_returns_vectors():
    captured = {}

    class FakeEmbed:
        def create(self, **kwargs):
            captured.update(kwargs)
            return type("R", (), {"data": [type("D", (), {"embedding": [0.0] * 1024})() for _ in kwargs["input"]]})()

    class FakeClient:
        embeddings = type("E", (), {"create": FakeEmbed().create})()

    with mock.patch.object(llm, "_client", return_value=FakeClient()):
        out = llm.embed(["a", "b"])
    assert len(out) == 2
    assert all(len(v) == 1024 for v in out)
    assert captured["input"] == ["a", "b"]


def test_embed_sanitizes_and_truncates_inputs():
    # None / non-string elements and over-long inputs must be normalized before
    # hitting the API (SiliconFlow returns code 20015 for those).
    long_text = "x" * (llm.MAX_EMBED_CHARS + 50)
    cleaned = llm._sanitize_embed_inputs([None, 1.5, long_text, "keep"])
    assert cleaned[0] == ""
    assert cleaned[1] == "1.5"
    assert cleaned[2] == long_text[: llm.MAX_EMBED_CHARS]
    assert cleaned[3] == "keep"

    captured = {}

    class FakeEmbed:
        def create(self, **kwargs):
            captured.update(kwargs)
            return type("R", (), {"data": [type("D", (), {"embedding": [0.0] * 4})() for _ in kwargs["input"]]})()

    class FakeClient:
        embeddings = type("E", (), {"create": FakeEmbed().create})()

    with mock.patch.object(llm, "_client", return_value=FakeClient()):
        out = llm.embed([None, 1.5, long_text, "keep"])
    assert len(out) == 4
    assert captured["input"] == cleaned


def test_generate_sql_strips_trailing_semicolon_and_temperature_zero():
    captured = {}

    class FakeMsg:
        def __init__(self, content):
            self.content = content

    class FakeChoice:
        def __init__(self, content):
            self.message = FakeMsg(content)

    class FakeResp:
        def __init__(self, content):
            self.choices = [FakeChoice(content)]

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeResp("SELECT 1;")

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    with mock.patch.object(llm, "_client", return_value=FakeClient()):
        sql = llm.generate_sql([{"table": "t", "columns": [], "sample_rows": []}], "全部")
    assert sql == "SELECT 1"
    assert captured["temperature"] == 0.0


def test_answer_uses_context_and_returns_content():
    captured = {}

    class FakeMsg:
        def __init__(self, content):
            self.content = content

    class FakeChoice:
        def __init__(self, content):
            self.message = FakeMsg(content)

    class FakeResp:
        def __init__(self, content):
            self.choices = [FakeChoice(content)]

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeResp("这是答案")

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    with mock.patch.object(llm, "_client", return_value=FakeClient()):
        out = llm.answer("问题", "上下文资料")
    assert out == "这是答案"
    msgs = captured["messages"]
    assert "知识库问答助手" in msgs[0]["content"]
    assert "参考上下文" in msgs[0]["content"]
    assert "上下文资料" in msgs[0]["content"]
    assert "问题" in msgs[1]["content"]
    assert captured["temperature"] == 0.0


def test_rerank_returns_scores_in_document_order():
    import json
    from unittest import mock
    import llm

    fake_response = mock.Mock()
    fake_response.read.return_value = json.dumps(
        {"results": [{"index": 1, "score": 0.9}, {"index": 0, "score": 0.3}]}
    ).encode("utf-8")
    fake_cm = mock.Mock()
    fake_cm.__enter__ = mock.Mock(return_value=fake_response)
    fake_cm.__exit__ = mock.Mock(return_value=False)

    with mock.patch.dict(os.environ, {"SILICONFLOW_API_KEY": "test-key"}), \
         mock.patch("llm.urllib.request.urlopen", return_value=fake_cm):
        scores = llm.rerank("q", ["doc0", "doc1"])
    assert scores == [0.3, 0.9]


def test_rerank_empty_documents_returns_empty():
    import llm

    assert llm.rerank("q", []) == []


def test_review_answer_parses_pass():
    import llm
    from unittest import mock

    class FakeChoice:
        def __init__(self, content):
            self.message = type("M", (), {"content": content})()

    class FakeResp:
        def __init__(self, content):
            self.choices = [FakeChoice(content)]

    class FakeCompletions:
        def create(self, **kwargs):
            return FakeResp('{"verdict": "pass", "reason": "回答正确"}')

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    with mock.patch.object(llm, "_client", return_value=FakeClient()):
        verdict, reason = llm.review_answer("q", "ctx", "ans")
    assert verdict is True
    assert reason == "回答正确"


def test_review_answer_fallback_on_bad_json():
    import llm
    from unittest import mock

    class FakeChoice:
        def __init__(self, content):
            self.message = type("M", (), {"content": content})()

    class FakeResp:
        def __init__(self, content):
            self.choices = [FakeChoice(content)]

    class FakeCompletions:
        def create(self, **kwargs):
            return FakeResp("not json at all")

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    with mock.patch.object(llm, "_client", return_value=FakeClient()):
        verdict, reason = llm.review_answer("q", "ctx", "ans")
    assert verdict is False
    assert reason == "not json at all"
