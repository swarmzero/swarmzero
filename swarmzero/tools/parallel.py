"""Optional, anonymous Parallel Search MCP tools for Agent(functions=...)."""

import asyncio

MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "swarmzero/0.0.3 (+https://github.com/swarmzero/swarmzero)"
TIMEOUT = 60


async def _call(name: str, arguments: dict) -> str:
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError as exc:
        raise ImportError('Install Parallel tools with pip install "swarmzero[parallel]"') from exc

    async with asyncio.timeout(TIMEOUT):
        async with streamablehttp_client(
            MCP_URL,
            headers={"User-Agent": USER_AGENT},
            timeout=TIMEOUT,
            sse_read_timeout=TIMEOUT,
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
    text = "\n".join(item.text for item in result.content if item.type == "text")
    if result.isError:
        raise RuntimeError(f"Parallel {name} failed: {text}")
    if not text:
        raise RuntimeError(f"Parallel {name} returned no text")
    return text


async def parallel_web_search(objective: str, search_queries: list[str], session_id: str = "") -> str:
    """Search the web for an objective with related keyword queries.

    Returns source URLs and excerpts. Provide at least one concise query. If
    supplying a session_id, reuse it across search and fetch in one conversation.
    """
    arguments = {"objective": objective, "search_queries": search_queries}
    if session_id:
        arguments["session_id"] = session_id
    return await _call("web_search", arguments)


async def parallel_web_fetch(urls: list[str], objective: str = "", session_id: str = "") -> str:
    """Read excerpts from up to 20 HTTP(S) URLs, optionally focused on an objective.

    Returns extracted page text and source URLs. The objective is limited to
    200 characters. Reuse the search session_id within the same conversation.
    """
    arguments = {"urls": urls}
    if objective:
        arguments["objective"] = objective
    if session_id:
        arguments["session_id"] = session_id
    return await _call("web_fetch", arguments)
