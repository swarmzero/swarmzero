from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.llms import ChatMessage, MessageRole

from swarmzero.chat import ChatManager


class DummyAgent:
    async def _arun_step(self, task_id):
        return type("MockResponse", (), {"is_last": True, "output": "ok"})

    def create_task(self, content, chat_history=None, extra_state=None):
        return type("MockTask", (), {"task_id": "1"})


class DummyDB:
    async def insert_data(self, table_name: str, data: dict):
        pass

    async def read_data(self, table_name: str, filters: dict):
        return []


def test_init_mem0_memory(monkeypatch):
    mock_memory = MagicMock()
    monkeypatch.setattr("swarmzero.chat.chat_manager.create_mem0_memory", lambda **_: mock_memory)
    chat_manager = ChatManager(DummyAgent(), user_id="u", session_id="s", memory_provider="mem0")
    assert chat_manager.mem0_memory is mock_memory


@pytest.mark.asyncio
def test_add_message_puts_to_mem0(monkeypatch):
    mock_memory = MagicMock()
    monkeypatch.setattr("swarmzero.chat.chat_manager.create_mem0_memory", lambda **_: mock_memory)
    chat_manager = ChatManager(DummyAgent(), user_id="u", session_id="s", memory_provider="mem0")
    db = DummyDB()
    msg = ChatMessage(role=MessageRole.USER, content="hi")
    asyncio_run = AsyncMock()
    monkeypatch.setattr(mock_memory, "aput", asyncio_run)
    import asyncio

    loop = asyncio.get_event_loop()
    loop.run_until_complete(chat_manager.add_message(db, msg.role.value, msg.content, []))
    asyncio_run.assert_called_once()
