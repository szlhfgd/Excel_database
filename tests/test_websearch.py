import json
import os
from unittest import mock

import websearch


def _fake_urlopen(payload, error=None):
    fake_resp = mock.Mock()
    if error is not None:
        fake_resp.read.side_effect = error
    else:
        fake_resp.read.return_value = json.dumps(payload).encode("utf-8")
    fake_cm = mock.Mock()
    fake_cm.__enter__ = mock.Mock(return_value=fake_resp)
    fake_cm.__exit__ = mock.Mock(return_value=False)
    return fake_cm


def test_search_returns_text_content():
    payload = {
        "result": {
            "content": [
                {"type": "text", "text": "1. 苹果 价格 5 元\n2. 香蕉 价格 3 元"},
            ]
        }
    }
    captured = {}

    def fake_urlopen(req, timeout=30):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["auth"] = req.get_header("Authorization")
        return _fake_urlopen(payload)

    with mock.patch.dict(os.environ, {"ANYSEARCH_API_KEY": "test-key"}), \
         mock.patch("websearch.urllib.request.urlopen", side_effect=fake_urlopen):
        text, err = websearch.search("苹果价格", max_results=5)

    assert err is None
    assert "苹果 价格 5 元" in text
    assert captured["url"] == "https://api.anysearch.com/mcp"
    assert captured["auth"] == "Bearer test-key"
    # JSON-RPC payload shape matches the anysearch-skill CLI.
    assert captured["body"]["method"] == "tools/call"
    assert captured["body"]["params"]["name"] == "search"
    assert captured["body"]["params"]["arguments"]["query"] == "苹果价格"
    assert captured["body"]["params"]["arguments"]["max_results"] == 5


def test_search_error_field_returns_error():
    payload = {"error": {"message": "quota exhausted"}}
    with mock.patch("websearch.urllib.request.urlopen", return_value=_fake_urlopen(payload)):
        text, err = websearch.search("q")
    assert text == ""
    assert err is not None and "quota exhausted" in err


def test_search_network_error_returns_error():
    with mock.patch(
        "websearch.urllib.request.urlopen",
        side_effect=OSError("connection refused"),
    ):
        text, err = websearch.search("q")
    assert text == ""
    assert err is not None and "网络搜索失败" in err


def test_search_no_api_key_sends_no_auth_header():
    payload = {"result": {"content": [{"type": "text", "text": "ok"}]}}
    captured = {}

    def fake_urlopen(req, timeout=30):
        captured["auth"] = req.get_header("Authorization")
        return _fake_urlopen(payload)

    with mock.patch.dict(os.environ, {"ANYSEARCH_API_KEY": ""}), \
         mock.patch("websearch.urllib.request.urlopen", side_effect=fake_urlopen):
        websearch.search("q")
    assert captured["auth"] is None
