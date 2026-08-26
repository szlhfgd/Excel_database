import os
from unittest import mock


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
