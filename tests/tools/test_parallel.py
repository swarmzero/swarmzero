import asyncio
import json
from unittest.mock import patch

import httpx
import pytest

from swarmzero.agent import Agent
from swarmzero.tools import parallel


@pytest.fixture
def mcp_server(monkeypatch):
    streamable_http = pytest.importorskip("mcp.client.streamable_http")
    requests = []
    calls = []
    state = {"error": False, "empty": False, "delay": False}

    async def respond(request):
        requests.append(request)
        if request.method != "POST":
            return httpx.Response(405)
        message = json.loads(request.content)
        if "id" not in message:
            return httpx.Response(202)
        if message["method"] == "initialize":
            result = {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fixture", "version": "1"},
            }
        elif message["method"] == "tools/list":
            result = {
                "tools": [{"name": name, "inputSchema": {"type": "object"}} for name in ["web_search", "web_fetch"]]
            }
        elif message["method"] == "ping":
            result = {}
        else:
            calls.append(message["params"])
            if state["delay"]:
                await asyncio.sleep(5)
            result = {
                "content": [] if state["empty"] else [{"type": "text", "text": "https://example.org source excerpt"}],
                "isError": state["error"],
            }
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": result})

    original = streamable_http.streamablehttp_client

    def client_factory(headers=None, timeout=None, auth=None):
        return httpx.AsyncClient(headers=headers, timeout=timeout, auth=auth, transport=httpx.MockTransport(respond))

    def connect(*args, **kwargs):
        return original(*args, **kwargs, httpx_client_factory=client_factory)

    monkeypatch.setattr(streamable_http, "streamablehttp_client", connect)
    return requests, calls, state


def loaded_tools():
    # Exercise the maintained Agent function-loader path without model or server setup.
    agent = object.__new__(Agent)
    agent.functions = [parallel.parallel_web_search, parallel.parallel_web_fetch]
    return agent.get_tools()


async def test_loader_dispatch_and_anonymous_requests(mcp_server, monkeypatch):
    monkeypatch.setenv("PARALLEL_API_KEY", "must-not-be-used")
    requests, calls, _ = mcp_server
    search, fetch = loaded_tools()
    assert search.metadata.name == "parallel_web_search"
    assert fetch.metadata.name == "parallel_web_fetch"
    result = await search.acall(objective="Find sources", search_queries=["example sources"], session_id="conversation")
    assert "source excerpt" in result.content
    result = await fetch.acall(urls=["https://example.org"], session_id="conversation")
    assert "source excerpt" in result.content
    assert calls == [
        {
            "name": "web_search",
            "arguments": {
                "objective": "Find sources",
                "search_queries": ["example sources"],
                "session_id": "conversation",
            },
        },
        {"name": "web_fetch", "arguments": {"urls": ["https://example.org"], "session_id": "conversation"}},
    ]
    assert requests
    assert all(str(request.url) == parallel.MCP_URL for request in requests)
    assert all(request.headers["user-agent"] == parallel.USER_AGENT for request in requests)
    assert all("authorization" not in request.headers for request in requests)
    assert Agent.get_tools(type("AgentFunctions", (), {"functions": []})()) == []


@pytest.mark.parametrize("flag, message", [("error", "failed"), ("empty", "no text")])
async def test_server_errors(mcp_server, flag, message):
    mcp_server[2][flag] = True
    with pytest.raises(RuntimeError, match=message):
        await parallel.parallel_web_fetch(["https://example.org"])


async def test_whole_call_timeout(mcp_server, monkeypatch):
    mcp_server[2]["delay"] = True
    monkeypatch.setattr(parallel, "TIMEOUT", 0.05)
    with pytest.raises(TimeoutError):
        await parallel.parallel_web_fetch(["https://example.org"])


async def test_cancellation(mcp_server):
    mcp_server[2]["delay"] = True
    task = asyncio.create_task(parallel.parallel_web_fetch(["https://example.org"]))
    while not mcp_server[1]:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_missing_optional_client():
    with patch.dict("sys.modules", {"mcp": None}):
        with pytest.raises(ImportError, match=r"swarmzero\[parallel\]"):
            await parallel.parallel_web_fetch(["https://example.org"])
