import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from llama_index.core.callbacks.base import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType
from llama_index.core.tools.types import ToolOutput

logger = logging.getLogger(__name__)


class EventCallbackHandler(BaseCallbackHandler):
    _aqueue: asyncio.Queue
    is_done: bool = False

    def __init__(self):
        """Initialize the event callback handler."""
        ignored_events = [
            CBEventType.CHUNKING,
            CBEventType.NODE_PARSING,
            CBEventType.EMBEDDING,
            CBEventType.LLM,
            CBEventType.TEMPLATING,
        ]
        super().__init__(ignored_events, ignored_events)
        self._aqueue = asyncio.Queue()

    def get_retrieval_message(self, payload: Optional[Dict[str, Any]]) -> dict | None:
        if payload:
            nodes = payload.get("nodes")
            if nodes:
                msg = f"Retrieved {len(nodes)} sources to use as context for the query"
            else:
                msg = f"Retrieving context for query: '{payload.get('query_str')}'"
            return {
                "type": "events",
                "data": {"title": msg},
            }
        else:
            return None

    def get_tool_message(self, payload: Optional[Dict[str, Any]]) -> dict | None:
        if payload is None:
            return None
        func_call_args = payload.get("function_call")
        if func_call_args is not None and "tool" in payload:
            tool = payload.get("tool")
            if tool is None:
                return None
            return {
                "type": "events",
                "data": {
                    "title": f"Calling tool: {tool.name} with inputs: {func_call_args}",
                },
            }
        return None

    def _is_output_serializable(self, output: Any) -> bool:
        try:
            json.dumps(output)
            return True
        except TypeError:
            return False

    def get_agent_tool_response(self, payload: Optional[Dict[str, Any]]) -> dict | None:
        if payload is None:
            return None
        response = payload.get("response")
        if response is not None:
            sources = response.sources
            for source in sources:
                if isinstance(source, ToolOutput):
                    if self._is_output_serializable(source.raw_output):
                        output = source.raw_output
                    else:
                        output = source.content

                    return {
                        "type": "tools",
                        "data": {
                            "toolOutput": {
                                "output": output,
                                "isError": source.is_error,
                            },
                            "toolCall": {
                                "id": None,
                                "name": source.tool_name,
                                "input": source.raw_input,
                            },
                        },
                    }
        return None

    def to_response(self, event_type: CBEventType, payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        try:
            match event_type:
                case "retrieve":
                    return self.get_retrieval_message(payload)
                case "function_call":
                    return self.get_tool_message(payload)
                case "agent_step":
                    return self.get_agent_tool_response(payload)
                case _:
                    return None
        except Exception as e:
            logger.error(f"Error in converting event to response: {e}")
            return None

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        response = self.to_response(event_type, payload)
        if response is not None:
            self._aqueue.put_nowait(response)
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        response = self.to_response(event_type, payload)
        if response is not None:
            self._aqueue.put_nowait(response)

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """No-op."""

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """No-op."""

    async def async_event_gen(self) -> AsyncGenerator[Dict[str, Any], None]:
        while not self._aqueue.empty() or not self.is_done:
            try:
                yield await asyncio.wait_for(self._aqueue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                pass
