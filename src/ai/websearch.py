"""AnySearch web-search client.

Wraps the AnySearch JSON-RPC 2.0 endpoint (https://api.anysearch.com/mcp) so the
RAG flow can fall back to live web search when the local database has no
matching rows. Uses only the stdlib (urllib), mirroring the request shape of the
official anysearch-skill CLI.

API key is read from the ``ANYSEARCH_API_KEY`` env var (anonymous access is
allowed with lower rate limits when absent).
"""

import json
import os
import urllib.request

ANYSEARCH_ENDPOINT = os.environ.get("ANYSEARCH_ENDPOINT", "https://api.anysearch.com/mcp")
ANYSEARCH_CLIENT = os.environ.get("ANYSEARCH_CLIENT", "skill/3.0.1")


def search(query: str, max_results: int = 5, api_key: str | None = None) -> tuple[str, str | None]:
    """Search the web via AnySearch.

    Returns ``(text_results, error | None)``. ``text_results`` is the raw text
    payload returned by the API (typically a Markdown list of hits). On any
    failure, ``text_results`` is empty and ``error`` carries a message.
    """
    key = api_key or os.environ.get("ANYSEARCH_API_KEY", "")
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "search",
            "arguments": {"query": query, "max_results": max_results},
        },
    }
    headers = {
        "Content-Type": "application/json",
        "X-Anysearch-Client": ANYSEARCH_CLIENT,
    }
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(
        ANYSEARCH_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            req, timeout=float(os.environ.get("ANYSEARCH_TIMEOUT", "30"))
        ) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - surface any network/parse failure to the caller
        return "", f"网络搜索失败：{exc}"
    if "error" in data:
        return "", f"网络搜索失败：{data['error'].get('message', data['error'])}"
    texts = [
        item.get("text", "")
        for item in data.get("result", {}).get("content", [])
        if item.get("type") == "text"
    ]
    return "\n".join(texts).strip(), None
