from unittest.mock import MagicMock, patch

import pytest
from llama_index.agent.openai import OpenAIAgent  # type: ignore
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.multi_modal_llms.openai import OpenAIMultiModal  # type: ignore

from swarmzero.core.services.chat import ChatManager


class MockAgent:
    async def astream_chat(self, content, chat_history=None):
        async def async_response_gen():
            yield "chat response"

        return type("MockResponse", (), {"async_response_gen": async_response_gen})

    async def achat(self, content, chat_history=None):
        return type("MockResponse", (), {"response": "chat response"})

    def create_task(self, content, chat_history=None):
        return type("MockTask", (), {"task_id": "12345"})

    async def _arun_step(self, task_id):
        return type("MockResponse", (), {"is_last": True, "output": "chat response"})


class MockMultiModalAgent:
    def create_task(self, content, extra_state=None):
        return type("MockTask", (), {"task_id": "12345"})

    async def _arun_step(self, task_id):
        return type("MockResponse", (), {"is_last": True, "output": "multimodal response"})

    def finalize_response(self, task_id):
        return "multimodal response"


class MockDatabaseManager:
    def __init__(self):
        self.data = []

    async def insert_data(self, table_name: str, data: dict):
        self.data.append(data)

    async def read_data(self, table_name: str, filters: dict):
        return [d for d in self.data if all(d[k] == v[0] for k, v in filters.items())]


@pytest.fixture
def agent():
    return MockAgent()


@pytest.fixture
def multi_modal_agent():
    # Mocking the methods within the agent to make them MagicMock objects
    agent = MockMultiModalAgent()
    agent._arun_step = MagicMock(side_effect=agent._arun_step)
    agent.finalize_response = MagicMock(side_effect=agent.finalize_response)
    return agent


@pytest.fixture
def db_manager():
    return MockDatabaseManager()


@pytest.mark.asyncio
async def test_add_message(agent, db_manager):
    chat_manager = ChatManager(agent, user_id="123", session_id="abc")
    await chat_manager.add_message(db_manager, MessageRole.USER, "Hello!", {'event': 'event'})
    messages = await chat_manager.get_messages(db_manager)
    assert len(messages) == 1
    assert messages[0].content == "Hello!"


@pytest.mark.asyncio
async def test_generate_response_with_generic_llm(agent, db_manager):
    chat_manager = ChatManager(agent, user_id="123", session_id="abc")
    user_message = ChatMessage(role=MessageRole.USER, content="Hello!")

    response = ""
    async for chunk in chat_manager.generate_response(db_manager, user_message, []):
        if isinstance(chunk, list):
            response += ''.join(chunk)
        elif chunk is not None:
            response += chunk
    assert response.split("END_OF_STREAM")[0] == "chat response"

    messages = await chat_manager.get_messages(db_manager)
    assert len(messages) == 2
    assert messages[0].content == "Hello!"
    assert messages[1].content == "chat response"


@pytest.mark.asyncio
async def test_get_all_chats_for_user(agent, db_manager):
    chat_manager1 = ChatManager(agent, user_id="123", session_id="abc")
    await chat_manager1.add_message(db_manager, MessageRole.USER, "Hello in abc", {'event': 'event'})
    await chat_manager1.add_message(db_manager, MessageRole.ASSISTANT, "Response in abc", {'event': 'event'})

    chat_manager2 = ChatManager(agent, user_id="123", session_id="def")
    await chat_manager2.add_message(db_manager, MessageRole.USER, "Hello in def", {'event': 'event'})
    await chat_manager2.add_message(db_manager, MessageRole.ASSISTANT, "Response in def", {'event': 'event'})

    chat_manager = ChatManager(agent, user_id="123", session_id="")
    all_chats = await chat_manager.get_all_chats_for_user(db_manager)

    assert "abc" in all_chats
    assert "def" in all_chats

    assert len(all_chats["abc"]) == 2
    assert all_chats["abc"][0]["message"] == "Hello in abc"
    assert all_chats["abc"][1]["message"] == "Response in abc"

    assert len(all_chats["def"]) == 2
    assert all_chats["def"][0]["message"] == "Hello in def"
    assert all_chats["def"][1]["message"] == "Response in def"


@pytest.mark.asyncio
async def test_generate_response_with_openai_multimodal(multi_modal_agent, db_manager):
    with patch("llama_index.core.settings._Settings.llm", new=MagicMock(spec=OpenAIMultiModal)):
        chat_manager = ChatManager(multi_modal_agent, user_id="123", session_id="abc", enable_multi_modal=True)
        user_message = ChatMessage(role=MessageRole.USER, content="Hello!")
        files = ["image1.png", "image2.png"]

        response = ""
        async for chunk in chat_manager.generate_response(db_manager, user_message, files):
            if chunk is not None:
                response += chunk

        response = response.replace("END_OF_STREAM", "")
        assert response == "multimodal response"

        messages = await chat_manager.get_messages(db_manager)
        assert len(messages) == 2
        assert messages[0].content == "Hello!"
        assert messages[1].content == "multimodal response"


@pytest.mark.asyncio
async def test_execute_task_success(multi_modal_agent):
    chat_manager = ChatManager(multi_modal_agent, user_id="123", session_id="abc")

    result = ""
    async for chunk in chat_manager._execute_task("task_id_123", event_handler=None):
        if chunk is not None:
            result += chunk

    result = result.replace("END_OF_STREAM", "")
    assert result == "multimodal response"
    multi_modal_agent._arun_step.assert_called_once_with("task_id_123")


@pytest.mark.asyncio
async def test_execute_task_with_exception(multi_modal_agent):
    async def mock_arun_step(task_id):
        raise ValueError(f"Could not find step_id: {task_id}")

    multi_modal_agent._arun_step = MagicMock(side_effect=mock_arun_step)

    chat_manager = ChatManager(multi_modal_agent, user_id="123", session_id="abc")

    result = ""
    async for chunk in chat_manager._execute_task("task_id_123", event_handler=None):
        if chunk is not None:
            result += chunk

    assert result == "error during step execution: Could not find step_id: task_id_123"
    multi_modal_agent._arun_step.assert_called_once_with("task_id_123")


@pytest.mark.asyncio
async def test_generate_response_with_openai_agent(agent, db_manager):
    with patch("llama_index.core.settings._Settings.llm", new=MagicMock(spec=OpenAIAgent)):
        chat_manager = ChatManager(agent, user_id="123", session_id="abc")
        user_message = ChatMessage(role=MessageRole.USER, content="Hello!")

        response = ""
        async for chunk in chat_manager.generate_response(db_manager, user_message, []):
            if chunk is not None:
                response += chunk

        response = response.replace("END_OF_STREAM", "")
        assert response == "chat response"

        messages = await chat_manager.get_messages(db_manager)
        assert len(messages) == 2
        assert messages[0].content == "Hello!"
        assert messages[1].content == "chat response"
