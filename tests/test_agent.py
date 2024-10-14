import os
import signal
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import pytest
from llama_index.core.agent.runner.base import AgentRunner
from llama_index.core.llms import ChatMessage, MessageRole

from swarmzero.agent import Agent
from swarmzero.tools.retriever.base_retrieve import IndexStore


@pytest.fixture
def agent():
    with (
        patch.object(IndexStore, "get_instance", return_value=IndexStore()),
        patch("swarmzero.agent.OpenAILLM"),
        patch("swarmzero.agent.ClaudeLLM"),
        patch("swarmzero.agent.MistralLLM"),
        patch("swarmzero.agent.OllamaLLM"),
        patch("swarmzero.agent.setup_routes"),
        patch("uvicorn.Server.serve", new_callable=MagicMock),
        patch("llama_index.core.VectorStoreIndex.from_documents"),
        patch("llama_index.core.objects.ObjectIndex.from_objects"),
        patch.object(IndexStore, "save_to_file", MagicMock()),
    ):
        os.environ['ANTHROPIC_API_KEY'] = "anthropic_api_key"
        os.environ['MISTRAL_API_KEY'] = "mistral_api_key"

        test_agent = Agent(
            name="TestAgent",
            functions=[lambda x: x],
            config_path="./swarmzero_config_test.toml",
            host="0.0.0.0",
            port=8000,
            instruction="Test instruction",
            role="leader",
            retrieve=True,
            required_exts=[".txt"],
            retrieval_tool="basic",
            load_index_file=False,
        )
    return test_agent


@pytest.mark.asyncio
async def test_agent_initialization(agent):
    assert agent.name == "TestAgent"
    assert agent.config_path == "./swarmzero_config_test.toml"
    assert agent.instruction == "Test instruction"
    assert agent.role == "leader"
    assert agent.retrieve is True
    assert agent.required_exts == [".txt"]
    assert agent.retrieval_tool == "basic"
    assert agent.load_index_file is False


def test_server_setup(agent):
    with patch("swarmzero.agent.setup_routes") as mock_setup_routes:
        agent._Agent__setup_server()
        mock_setup_routes.assert_called_once()


@pytest.mark.asyncio
async def test_run_server(agent):
    with patch("uvicorn.Server.serve", new_callable=MagicMock) as mock_serve:
        await agent.run_server()
        mock_serve.assert_called_once()


def test_signal_handler(agent):
    agent.shutdown_event = MagicMock()
    agent.shutdown_procedures = MagicMock()
    with patch("asyncio.create_task") as mock_create_task:
        agent._Agent__signal_handler(signal.SIGINT, None)
        mock_create_task.assert_called_once_with(agent.shutdown_procedures())


def test_server_setup_exception(agent):
    with patch("swarmzero.agent.setup_routes") as mock_setup_routes:
        mock_setup_routes.side_effect = Exception("Failed to setup routes")
        with pytest.raises(Exception):
            agent._Agent__setup_server()


def test_openai_agent_initialization_exception(agent):
    with patch("llama_index.agent.openai.OpenAIAgent.from_tools") as mock_from_tools:
        mock_from_tools.side_effect = Exception("Failed to initialize OpenAI agent")
        with pytest.raises(Exception):
            agent._Agent__setup()


@pytest.mark.asyncio
async def test_shutdown_procedures_exception(agent):
    with patch("asyncio.gather") as mock_gather:
        mock_gather.side_effect = Exception("Failed to gather tasks")
        with pytest.raises(Exception):
            await agent.shutdown_procedures()


@pytest.mark.asyncio
async def test_cleanup(agent):
    agent.db_session = MagicMock()
    await agent._Agent__cleanup()
    agent.db_session.close.assert_called_once()


def test_recreate_agent(agent):
    """
    with patch("swarmzero.utils.tools_from_funcs") as mock_tools_from_funcs, patch.object(
        IndexStore, "get_instance"
    ) as mock_get_instance, patch.object(ObjectIndex, "from_objects") as mock_from_objects, patch.object(
        agent, "_assign_agent"
    ) as mock_assign_agent:

        mock_custom_tools = [MagicMock(name="custom_tool")]
        mock_system_tools = [MagicMock(name="system_tool")]
        mock_tools_from_funcs.side_effect = [mock_custom_tools, mock_system_tools]

        mock_index_store = MagicMock()
        mock_get_instance.return_value = mock_index_store

        mock_vectorstore_object = MagicMock()
        mock_from_objects.return_value = mock_vectorstore_object

        agent.recreate_agent()

        mock_tools_from_funcs.assert_any_call(agent.functions)
        mock_tools_from_funcs.assert_any_call([get_db_schemas, text_2_sql])

        mock_get_instance.assert_called_once()

        mock_from_objects.assert_called_once_with(
            mock_custom_tools + mock_system_tools,
            index=mock_index_store.get_all_indexes(),
        )

        mock_vectorstore_object.as_retriever.assert_called_once_with(similarity_top_k=3)

        mock_assign_agent.assert_called_once_with([], mock_vectorstore_object.as_retriever.return_value)
    """
    pass


def test_assign_agent(agent):
    with (
        patch("swarmzero.llms.openai.OpenAIMultiModalLLM") as mock_openai_multimodal,
        patch("swarmzero.llms.openai.OpenAILLM") as mock_openai_llm,
        patch("swarmzero.llms.claude.ClaudeLLM") as mock_claude_llm,
        patch("swarmzero.llms.ollama.OllamaLLM") as mock_ollama_llm,
        patch("swarmzero.llms.mistral.MistralLLM") as mock_mistral_llm,
    ):
        models = [
            ("gpt-4o", mock_openai_multimodal),
            ("gpt-3.5-turbo", mock_openai_llm),
            ("claude-3-opus-20240229", mock_claude_llm),
            ("llama-2", mock_ollama_llm),
            ("mistral-large-latest", mock_mistral_llm),
            ("gpt-4", mock_openai_llm),
        ]

        tools = MagicMock()
        tool_retriever = MagicMock()

        for model_name, expected_mock_class in models:
            with patch("swarmzero.config.Config.get", return_value=model_name):
                agent._assign_agent(tools, tool_retriever)

                # expected_mock_class.assert_called_once()  # todo not working

                assert isinstance(agent._Agent__agent, AgentRunner)

                # expected_mock_class.reset_mock()


@pytest.mark.asyncio
async def test_chat_method(agent):
    """Test the chat method with different scenarios including default and custom parameters."""
    agent.sdk_context.get_utility = MagicMock()
    mock_db_manager = MagicMock()
    agent.sdk_context.get_utility.return_value = mock_db_manager

    agent._ensure_utilities_loaded = AsyncMock()

    test_cases = [
        {"prompt": "Hello", "user_id": "default_user", "session_id": "default_chat", "image_paths": []},
        {
            "prompt": "Analyze this image",
            "user_id": "custom_user",
            "session_id": "custom_session",
            "image_paths": ["path/to/image.jpg"],
        },
    ]

    with patch("swarmzero.agent.ChatManager", autospec=True) as mock_chat_manager_class:
        mock_chat_manager_instance = mock_chat_manager_class.return_value
        mock_chat_manager_instance.generate_response = AsyncMock(side_effect=["Response 1", "Response 2"])

        for i, test_case in enumerate(test_cases):
            response = await agent.chat(
                prompt=test_case["prompt"],
                user_id=test_case["user_id"],
                session_id=test_case["session_id"],
                image_document_paths=test_case["image_paths"],
            )

            assert response == f"Response {i + 1}"

            agent._ensure_utilities_loaded.assert_called()

            agent.sdk_context.get_utility.assert_called_with("db_manager")

            mock_chat_manager_class.assert_called_with(
                agent._Agent__agent, user_id=test_case["user_id"], session_id=test_case["session_id"]
            )

            expected_message = ChatMessage(role=MessageRole.USER, content=test_case["prompt"])
            mock_chat_manager_instance.generate_response.assert_called_with(
                mock_db_manager, expected_message, test_case["image_paths"]
            )


@pytest.mark.asyncio
async def test_chat_method_error_handling(agent):
    """Test error handling in the chat method."""
    agent.sdk_context.get_utility = MagicMock(return_value=MagicMock())
    agent._ensure_utilities_loaded = AsyncMock()

    with patch("swarmzero.agent.ChatManager", autospec=True) as mock_chat_manager_class:
        mock_chat_manager_instance = mock_chat_manager_class.return_value
        mock_chat_manager_instance.generate_response = AsyncMock(side_effect=Exception("Test error"))

        with pytest.raises(Exception) as exc_info:
            await agent.chat("Hello")

        assert str(exc_info.value) == "Test error"


@pytest.mark.asyncio
async def test_chat_history_method(agent):
    agent.sdk_context.get_utility = MagicMock()
    mock_db_manager = MagicMock()

    agent.sdk_context.get_utility.return_value = mock_db_manager

    with patch("swarmzero.agent.ChatManager") as mock_chat_manager_class:
        mock_chat_manager_instance = mock_chat_manager_class.return_value

        mock_chat_manager_instance.get_all_chats_for_user = AsyncMock(
            return_value={
                "default_chat": [
                    {"message": "what's the capital of Nigeria?", "role": "user", "timestamp": "2024-01-01T12:00:00Z"},
                    {
                        "message": "The capital of Nigeria is Abuja.",
                        "role": "assistant",
                        "timestamp": "2024-01-01T12:01:00Z",
                    },
                    {"message": "what's the population?", "role": "user", "timestamp": "2024-01-01T12:02:00Z"},
                    {
                        "message": "Nigeria has a population of over 200 million.",
                        "role": "assistant",
                        "timestamp": "2024-01-01T12:03:00Z",
                    },
                ]
            }
        )

        chats = await agent.chat_history(user_id="default_user", session_id="default_chat")

        agent.sdk_context.get_utility.assert_called_once_with("db_manager")

        mock_chat_manager_instance.get_all_chats_for_user.assert_awaited_once_with(mock_db_manager)

        expected_chat_history = {
            "default_chat": [
                {"message": "what's the capital of Nigeria?", "role": "user", "timestamp": "2024-01-01T12:00:00Z"},
                {
                    "message": "The capital of Nigeria is Abuja.",
                    "role": "assistant",
                    "timestamp": "2024-01-01T12:01:00Z",
                },
                {"message": "what's the population?", "role": "user", "timestamp": "2024-01-01T12:02:00Z"},
                {
                    "message": "Nigeria has a population of over 200 million.",
                    "role": "assistant",
                    "timestamp": "2024-01-01T12:03:00Z",
                },
            ]
        }
        assert chats == expected_chat_history
