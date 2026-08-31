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


def test_truncate_token_safe_keeps_short_text_unchanged():
    # Below the limit: no tokenization, input returned untouched.
    body = "短文本" * 10
    assert llm._truncate_token_safe(body) == body


def test_truncate_token_safe_short_text_is_plain_prefix():
    # A long run of one char cannot be cut mid-"word"; the hard prefix is a
    # safe fallback that never exceeds the limit.
    long_text = "x" * (llm.MAX_EMBED_CHARS + 50)
    out = llm._truncate_token_safe(long_text)
    assert len(out) <= llm.MAX_EMBED_CHARS


def test_truncate_token_safe_ends_at_word_boundary():
    # CJK tokenization lets us stop at a whole-word boundary instead of
    # slicing a sentence mid-word.
    base = "".join(f"词语{i:03d}" for i in range(500))  # clearly over the limit
    truncated = llm._truncate_token_safe(base)
    assert len(truncated) <= llm.MAX_EMBED_CHARS
    # Should be a strict prefix of the token stream (no split token tail).
    assert base.startswith(truncated) or truncated == ""


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


def test_answer_stream_yields_concatenated_chunks():
    import types
    from unittest import mock

    captured = {}

    class FakeDelta:
        def __init__(self, content):
            self.content = content

    class FakeChunk:
        def __init__(self, content):
            self.choices = [types.SimpleNamespace(delta=FakeDelta(content))]

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            assert kwargs.get("stream") is True
            return iter([FakeChunk("你"), FakeChunk("好"), FakeChunk("！")])

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    with mock.patch.object(llm, "_client", return_value=FakeClient()):
        out = "".join(llm.answer_stream("问题", "上下文资料"))
    assert out == "你好！"
    msgs = captured["messages"]
    assert "上下文资料" in msgs[0]["content"]
    assert msgs[1]["role"] == "user" and msgs[1]["content"] == "问题"


def test_answer_stream_skips_none_delta():
    import types
    from unittest import mock

    class FakeDelta:
        def __init__(self, content):
            self.content = content

    class FakeChunk:
        def __init__(self, content):
            self.choices = [types.SimpleNamespace(delta=FakeDelta(content))]

    class FakeCompletions:
        def create(self, **kwargs):
            # First chunk has delta.content=None (e.g. role-only chunk) → must be skipped.
            return iter([FakeChunk(None), FakeChunk("答"), FakeChunk(None), FakeChunk("案")])

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    with mock.patch.object(llm, "_client", return_value=FakeClient()):
        out = "".join(llm.answer_stream("问题", "上下文资料"))
    assert out == "答案"


def test_answer_stream_injects_history_before_user_question():
    import types
    from unittest import mock

    captured = {}

    class FakeDelta:
        def __init__(self, content):
            self.content = content

    class FakeChunk:
        def __init__(self, content):
            self.choices = [types.SimpleNamespace(delta=FakeDelta(content))]

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return iter([FakeChunk("回")])

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    history = [
        {"role": "user", "content": "上一问"},
        {"role": "assistant", "content": "上一答"},
    ]
    with mock.patch.object(llm, "_client", return_value=FakeClient()):
        list(llm.answer_stream("当前问题", "上下文资料", history=history))
    msgs = captured["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[1] == {"role": "user", "content": "上一问"}
    assert msgs[2] == {"role": "assistant", "content": "上一答"}
    assert msgs[3] == {"role": "user", "content": "当前问题"}


def test_answer_stream_caps_history_turns():
    import types
    from unittest import mock

    captured = {}

    class FakeDelta:
        def __init__(self, content):
            self.content = content

    class FakeChunk:
        def __init__(self, content):
            self.choices = [types.SimpleNamespace(delta=FakeDelta(content))]

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return iter([FakeChunk("x")])

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    # 25 turns → capped to the most recent MAX_HISTORY_TURNS (10).
    history = [{"role": "user", "content": f"t{i}"} for i in range(25)]
    with mock.patch.object(llm, "_client", return_value=FakeClient()):
        list(llm.answer_stream("q", "ctx", history=history))
    msgs = captured["messages"]
    # msgs = [system] + capped_history + [current user]
    assert len(msgs) == 1 + llm.MAX_HISTORY_TURNS + 1
    assert msgs[1]["content"] == "t15"  # newest 10 of 25 → t15..t24
    assert msgs[-1]["content"] == "q"

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


def test_can_answer_true_when_context_sufficient():
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
            return FakeResp('{"can_answer": true, "reason": "上下文包含答案"}')

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    with mock.patch.object(llm, "_client", return_value=FakeClient()):
        assert llm.can_answer("q", "ctx") is True


def test_can_answer_false_when_context_insufficient():
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
            return FakeResp('{"can_answer": false, "reason": "上下文无关"}')

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    with mock.patch.object(llm, "_client", return_value=FakeClient()):
        assert llm.can_answer("q", "ctx") is False


def test_can_answer_defaults_true_on_bad_json():
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
        assert llm.can_answer("q", "ctx") is True


def test_generate_sql_renders_column_samples_when_present():
    import llm
    from unittest import mock

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
            return FakeResp("SELECT * FROM `t`")

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    schema = {
        "table": "t",
        "columns": [("a", "INTEGER"), ("b", "TEXT")],
        "sample_rows": [],
        "column_samples": {"a": ["1", "2"], "b": ["x"]},
    }
    with mock.patch.object(llm, "_client", return_value=FakeClient()):
        sql = llm.generate_sql([schema], "全部")
    content = captured["messages"][0]["content"]
    assert "列样例值" in content
    assert "a" in content and "1" in content
    assert "样本行" not in content


def test_generate_sql_falls_back_to_sample_rows_without_column_samples():
    import llm
    from unittest import mock

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
            return FakeResp("SELECT * FROM `t`")

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    schema = {"table": "t", "columns": [("a", "INTEGER")], "sample_rows": [{"a": 1}]}
    with mock.patch.object(llm, "_client", return_value=FakeClient()):
        llm.generate_sql([schema], "全部")
    content = captured["messages"][0]["content"]
    assert "样本行" in content
    assert "列样例值" not in content


def test_decompose_question_parses_json_array():
    import llm
    from unittest import mock

    captured = {}

    class FakeChoice:
        def __init__(self, content):
            self.message = type("M", (), {"content": content})()

    class FakeResp:
        def __init__(self, content):
            self.choices = [FakeChoice(content)]

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeResp('["子问题1", "子问题2"]')

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    with mock.patch.object(llm, "_client", return_value=FakeClient()):
        out = llm.decompose_question("原问题", [{"table": "t", "columns": [], "sample_rows": []}])
    assert out == ["子问题1", "子问题2"]


def test_decompose_question_fallback_on_unparseable():
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
            return FakeResp("完全不是 JSON")

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    with mock.patch.object(llm, "_client", return_value=FakeClient()):
        out = llm.decompose_question("原问题", [{"table": "t", "columns": [], "sample_rows": []}])
    assert out == ["原问题"]


def test_cross_validate_returns_content():
    import llm
    from unittest import mock

    captured = {}

    class FakeChoice:
        def __init__(self, content):
            self.message = type("M", (), {"content": content})()

    class FakeResp:
        def __init__(self, content):
            self.choices = [FakeChoice(content)]

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeResp("最终答案")

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    with mock.patch.object(llm, "_client", return_value=FakeClient()):
        final = llm.cross_validate("问题", "SQL 结果", "文本上下文")
    assert final == "最终答案"
    assert "问题" in captured["messages"][1]["content"]
    assert "SQL 结果" in captured["messages"][1]["content"]
    assert "文本上下文" in captured["messages"][1]["content"]
