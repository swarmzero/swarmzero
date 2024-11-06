import json
import uuid
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import ToolMetadata

from swarmzero.config import Config
from swarmzero.core.agent import Agent
from swarmzero.core.sdk_context import SDKContext
from swarmzero.core.services.chat import ChatManager
from swarmzero.core.services.llms.llm import LLM
from swarmzero.core.swarm import Swarm


@pytest.fixture
def mock_sdk_context():
    mock = Mock(spec=SDKContext)
    mock.generate_agents_from_config = Mock(return_value=[])
    mock.add_resource = Mock()
    mock.get_config = Mock(return_value=Mock(spec=Config))
    mock.load_default_utility = AsyncMock()
    mock.get_utility = Mock()
    return mock


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLM)
    # Add any necessary LLM attributes/methods here
    return llm


@pytest.fixture
def mock_functions():
    def dummy_function():
        pass

    return [dummy_function]


@pytest.fixture
def mock_agent():
    agent = MagicMock(spec=Agent)
    agent.name = "an_agent"
    agent.id = str(uuid.uuid4())
    agent.role = "test_role"
    agent.description = "Test agent description"
    agent.metadata = ToolMetadata(name="test_agent", description="Test agent description")
    return agent


@pytest.fixture
def mock_react_agent():
    agent = MagicMock(spec=ReActAgent)
    return agent


@pytest.fixture
def basic_swarm(mock_sdk_context, mock_llm, mock_functions, mock_react_agent):
    # Mocking the SDK context to return a list of mock agents
    mock_agent = MagicMock(spec=Agent)
    mock_agent.name = "test_agent"
    mock_agent.id = "agent_id_1"
    mock_agent.role = "test_role"
    mock_agent.description = "Test agent description"

    # Mock the SDK context's generate_agents_from_config method
    mock_sdk_context.generate_agents_from_config.return_value = [mock_agent]

    with patch('swarmzero.swarm.llm_from_config_without_agent', return_value=mock_llm):
        with patch('swarmzero.swarm.llm_from_wrapper', return_value=mock_llm):
            with patch('swarmzero.swarm.ReActAgent.from_tools', return_value=mock_react_agent):
                swarm = Swarm(
                    name="test_swarm",
                    description="Test swarm",
                    instruction="Test instruction",
                    functions=mock_functions,
                    llm=mock_llm,
                    sdk_context=mock_sdk_context,
                )
    return swarm


@pytest.mark.asyncio
async def test_add_agent(basic_swarm, mock_agent, mock_react_agent, mock_llm):
    with patch('swarmzero.swarm.llm_from_wrapper', return_value=mock_llm) as mock_llm_wrapper:
        with patch('swarmzero.swarm.ReActAgent.from_tools', return_value=mock_react_agent):
            mock_llm_wrapper.return_value = mock_llm
            basic_swarm.add_agent(mock_agent)
            assert mock_agent.name in basic_swarm._Swarm__agents
            assert basic_swarm._Swarm__agents[mock_agent.name]["agent"] == mock_agent


@pytest.mark.asyncio
async def test_add_duplicate_agent(basic_swarm, mock_agent, mock_react_agent, mock_llm):
    with patch('swarmzero.swarm.llm_from_wrapper', return_value=mock_llm) as mock_llm_wrapper:
        with patch('swarmzero.swarm.ReActAgent.from_tools', return_value=mock_react_agent):
            mock_llm_wrapper.return_value = mock_llm
            basic_swarm.add_agent(mock_agent)
            with pytest.raises(ValueError, match=f"Agent `{mock_agent.name}` already exists in the swarm."):
                basic_swarm.add_agent(mock_agent)


@pytest.mark.asyncio
async def test_remove_agent(basic_swarm, mock_agent, mock_react_agent, mock_llm):
    with patch('swarmzero.swarm.llm_from_wrapper', return_value=mock_llm) as mock_llm_wrapper:
        with patch('swarmzero.swarm.ReActAgent.from_tools', return_value=mock_react_agent):
            mock_llm_wrapper.return_value = mock_llm
            basic_swarm.add_agent(mock_agent)
            basic_swarm.remove_agent(mock_agent.name)
            assert mock_agent.name not in basic_swarm._Swarm__agents


@pytest.mark.asyncio
async def test_remove_nonexistent_agent(basic_swarm):
    with pytest.raises(ValueError):
        basic_swarm.remove_agent("nonexistent_agent")


@pytest.mark.asyncio
async def test_chat(basic_swarm, mock_sdk_context):
    mock_chat_manager = AsyncMock(spec=ChatManager)

    async def mock_generate_response(*args, **kwargs):
        yield "Test response"
        yield "END_OF_STREAM"

    mock_chat_manager.generate_response = mock_generate_response

    with patch('swarmzero.swarm.ChatManager', return_value=mock_chat_manager):
        response = await basic_swarm.chat(prompt="Test prompt", user_id="test_user", session_id="test_session")

        # The response will contain both the test response and END_OF_STREAM markers
        assert "Test response" in response
        # Verify both chunks are present in the expected format
        assert '1:"Test response"' in response
        assert '1:"END_OF_STREAM"' in response


@pytest.mark.asyncio
async def test_chat_stream(basic_swarm, mock_sdk_context):
    mock_chat_manager = AsyncMock(spec=ChatManager)

    async def mock_generate_response(*args, **kwargs):
        yield "Test response"

    mock_chat_manager.generate_response = mock_generate_response

    with patch('swarmzero.swarm.ChatManager', return_value=mock_chat_manager):
        response = await basic_swarm.chat_stream(prompt="Test prompt", user_id="test_user", session_id="test_session")

        # Get the response content from the StreamingResponse
        response_content = ""
        async for chunk in response.body_iterator:
            if isinstance(chunk, bytes):
                chunk = chunk.decode()
            if chunk.startswith('1:'):  # Regular message chunk
                content = json.loads(chunk[2:])
                response_content += content

        assert response_content == "Test response"


@pytest.mark.asyncio
async def test_chat_history(basic_swarm, mock_sdk_context):
    mock_chat_manager = AsyncMock(spec=ChatManager)
    expected_history = {"messages": [{"role": "user", "content": "Test message"}]}
    mock_chat_manager.get_all_chats_for_user = AsyncMock(return_value=expected_history)

    with patch('swarmzero.swarm.ChatManager', return_value=mock_chat_manager):
        history = await basic_swarm.chat_history(user_id="test_user", session_id="test_session")

        assert history == expected_history
        mock_chat_manager.get_all_chats_for_user.assert_called_once()


@pytest.mark.asyncio
async def test_format_tool_name(basic_swarm):
    test_cases = [
        ("Test Agent", "test_agent"),
        ("test-agent", "test_agent"),
        ("test.agent!", "testagent"),
        ("Test Agent 123", "test_agent_123"),
    ]

    for input_name, expected_output in test_cases:
        assert basic_swarm._format_tool_name(input_name) == expected_output


@pytest.mark.asyncio
async def test_ensure_utilities_loaded(basic_swarm, mock_sdk_context):
    await basic_swarm._ensure_utilities_loaded()
    mock_sdk_context.load_default_utility.assert_called_once()

    mock_sdk_context.load_default_utility.reset_mock()

    await basic_swarm._ensure_utilities_loaded()
    mock_sdk_context.load_default_utility.assert_not_called()


@pytest.mark.skip(reason="Skipping this test for now")
@pytest.mark.asyncio
async def test_ensure_utilities_loaded_complete(mock_sdk_context, mock_llm, mock_functions, mock_react_agent):
    with patch('swarmzero.swarm.llm_from_config_without_agent', return_value=mock_llm):
        with patch('swarmzero.swarm.llm_from_wrapper', return_value=mock_llm):
            with patch('swarmzero.swarm.ReActAgent.from_tools', return_value=mock_react_agent):
                # Create a fresh swarm instance
                swarm = Swarm(
                    name="test_swarm",
                    description="Test swarm",
                    instruction="Test instruction",
                    functions=mock_functions,
                    llm=mock_llm,
                    sdk_context=mock_sdk_context,
                )

                # First call should load utilities
                await swarm._ensure_utilities_loaded()
                mock_sdk_context.load_default_utility.assert_called_once()

                # Reset the mock to clear the call history
                mock_sdk_context.load_default_utility.reset_mock()

                # Second call should not load utilities again
                await swarm._ensure_utilities_loaded()
                mock_sdk_context.load_default_utility.assert_not_called()


@pytest.mark.skip(reason="Skipping this test for now")
@pytest.mark.asyncio
async def test_swarm_with_custom_id(mock_sdk_context, mock_llm, mock_functions, mock_react_agent):
    custom_id = "custom-swarm-id"
    mock_sdk_context.generate_agents_from_config.return_value = []  # Explicitly return empty list

    with patch('swarmzero.swarm.llm_from_config_without_agent', return_value=mock_llm):
        with patch('swarmzero.swarm.llm_from_wrapper', return_value=mock_llm):
            with patch('swarmzero.swarm.ReActAgent.from_tools', return_value=mock_react_agent):
                swarm = Swarm(
                    name="test_swarm",
                    description="Test swarm",
                    instruction="Test instruction",
                    functions=mock_functions,
                    swarm_id=custom_id,
                    sdk_context=mock_sdk_context,
                )
                assert swarm.id == custom_id


@pytest.mark.skip(reason="Skipping this test for now")
@pytest.mark.asyncio
async def test_build_swarm_with_agents(mock_sdk_context, mock_llm, mock_agent):
    with patch('swarmzero.swarm.llm_from_config_without_agent', return_value=mock_llm):
        with patch('swarmzero.swarm.llm_from_wrapper', return_value=mock_llm):
            with patch('swarmzero.swarm.ReActAgent.from_tools') as mock_react_agent:
                swarm = Swarm(
                    name="test_swarm",
                    description="Test swarm",
                    instruction="Test instruction",
                    functions=[],
                    sdk_context=mock_sdk_context,
                    llm=mock_llm,
                )
                swarm.add_agent(mock_agent)

                mock_react_agent.assert_called()
                call_args = mock_react_agent.call_args[1]
                assert 'tools' in call_args
                assert 'context' in call_args
                assert call_args['context'] == "Test instruction"
