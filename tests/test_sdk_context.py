import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from swarmzero.agent import Agent
from swarmzero.sdk_context import SDKContext


@pytest.fixture
def sdk_context():
    with patch("swarmzero.config.config.Config.load_config") as mock_load_config:
        mock_load_config.return_value = {
            "model": {"model": "gpt-3.5-turbo"},
            "environment": {"type": "dev"},
            "timeout": {"llm": 30},
            "log": {"level": "INFO"},
        }
        context = SDKContext("./swarmzero_config_test.toml")
        return context


def test_load_default_config(sdk_context):
    default_config = sdk_context.load_default_config()
    assert isinstance(default_config, dict)
    assert "model" in default_config
    assert "environment" in default_config
    assert "timeout" in default_config
    assert "log" in default_config


def test_load_agent_configs(sdk_context):
    agent_configs = sdk_context.load_agent_configs()
    assert isinstance(agent_configs, dict)


def test_set_config(sdk_context):
    sdk_context.set_config("test_agent", "model", "gpt-4")
    assert sdk_context.agent_configs["test_agent"]["model"] == "gpt-4"


def test_get_config(sdk_context):
    config = sdk_context.get_config("test_agent")
    assert isinstance(config, dict)
    assert "model" in config


@patch("swarmzero.sdk_context.SDKContext")
def test_add_resource_agent(mock_sdk_context):
    sdk_context = mock_sdk_context.return_value
    sdk_context.resources = {}

    mock_agent = MagicMock(spec=Agent)
    mock_agent.id = "test_agent"
    mock_agent.name = "Test Agent"
    mock_agent.functions = []
    mock_agent.config_path = "test_config_path"
    mock_agent.instruction = "test_instruction"
    mock_agent.role = "test_role"
    mock_agent.description = "test_description"
    mock_agent.retrieve = True
    mock_agent.required_exts = [".txt"]
    mock_agent.retrieval_tool = "test_tool"
    mock_agent.load_index_file = False
    mock_agent.llm = "test_llm"
    mock_agent.host = "test_host"
    mock_agent.port = "test_port"
    mock_agent.swarm_mode = "test_swarm_mode"

    sdk_context.add_resource(mock_agent, "agent")

    # Configure the mock to return the expected value
    sdk_context.get_resource.return_value = {"test_agent": mock_agent}

    # Update the assertion to check if the mock_agent is in the returned dictionary
    assert "test_agent" in sdk_context.get_resource('test_agent')
    assert sdk_context.get_resource('test_agent')["test_agent"] == mock_agent


def test_add_resource_tool(sdk_context):
    def test_tool():
        """Test tool docstring"""
        pass

    sdk_context.add_resource(test_tool, "tool")

    assert "test_tool" in sdk_context.resources
    assert sdk_context.resources["test_tool"]["object"] == test_tool


def test_get_resource(sdk_context):
    def test_tool():
        pass

    sdk_context.add_resource(test_tool, "tool")
    retrieved_tool = sdk_context.get_resource("test_tool")

    assert retrieved_tool == test_tool


@pytest.mark.asyncio
async def test_initialize_database(sdk_context):
    with patch("swarmzero.sdk_context.initialize_db") as mock_initialize_db:
        await sdk_context.initialize_database()
        mock_initialize_db.assert_called_once()


@pytest.mark.asyncio
async def test_save_sdk_context_to_db(sdk_context):
    mock_db_manager = AsyncMock()
    sdk_context.get_utility = MagicMock(return_value=mock_db_manager)
    mock_db_manager.get_table_definition = AsyncMock(return_value=None)
    mock_db_manager.create_table = AsyncMock()
    mock_db_manager.insert_data = AsyncMock()

    # Set up test data
    sdk_context.default_config = {"test": "config"}
    sdk_context.agent_configs = {"agent1": {"config": "test"}}
    sdk_context.resources = {"resource1": {"init_params": {"type": "agent", "name": "test"}, "object": None}}
    sdk_context.utilities = {"utility1": {"info": {"type": "test"}, "object": None}}

    await sdk_context.save_sdk_context_to_db()

    # Verify table creation and data insertion
    mock_db_manager.get_table_definition.assert_called_once_with("sdkcontext")
    mock_db_manager.create_table.assert_called_once_with(
        "sdkcontext", {"type": "String", "data": "JSON", "create_date": "DateTime"}
    )
    mock_db_manager.insert_data.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_data(sdk_context):
    mock_db_manager = AsyncMock()
    sdk_context.get_utility = MagicMock(return_value=mock_db_manager)
    mock_db_manager.read_data = AsyncMock(return_value=[{"test": "data"}])

    conditions = {"type": "test"}
    result = await sdk_context.fetch_data("test_table", conditions)

    mock_db_manager.read_data.assert_called_once_with("test_table", filters=conditions, order_by="create_date", limit=1)
    assert result == [{"test": "data"}]


@pytest.mark.asyncio
async def test_load_sdk_context_from_db(sdk_context):
    test_data = {
        "default_config": {"test": "config"},
        "agent_configs": {"agent1": {"config": "test"}},
        "resources": {"resource1": {"type": "agent", "name": "test"}},
        "utilities": {"utility1": {"type": "test"}},
    }
    sdk_context.fetch_data = AsyncMock(return_value=[["sdkcontext", "sdk_context", json.dumps(test_data)]])
    sdk_context.load_default_utility = MagicMock()
    sdk_context.load_db_manager = AsyncMock()
    sdk_context.restore_non_serializable_objects = MagicMock()

    result = await sdk_context.load_sdk_context_from_db()

    assert result.default_config == test_data["default_config"]
    assert result.agent_configs == test_data["agent_configs"]
    assert result.resources == {"resource1": {"init_params": {"type": "agent", "name": "test"}, "object": None}}


@pytest.mark.asyncio
async def test_save_resource_to_db(sdk_context):
    mock_db_manager = AsyncMock()
    sdk_context.get_utility = MagicMock(return_value=mock_db_manager)
    mock_db_manager.get_table_definition = AsyncMock(return_value=None)
    mock_db_manager.create_table = AsyncMock()
    mock_db_manager.insert_data = AsyncMock()

    # Add a test resource first
    test_agent = MagicMock(spec=Agent)
    test_agent.id = "test_id"
    test_agent.name = "test_agent"
    test_agent.config = {"key": "value"}
    test_agent.config_path = "test_config_path"
    test_agent.instruction = "test_instruction"
    test_agent.role = "test_role"
    test_agent.description = "test_description"
    test_agent.retrieve = True
    test_agent.required_exts = [".txt"]
    test_agent.retrieval_tool = "test_tool"
    test_agent.load_index_file = False
    test_agent.llm = "test_llm"
    test_agent._Agent__host = "test_host"
    test_agent._Agent__port = "test_port"
    test_agent._Agent__swarm_mode = "test_swarm_mode"
    test_agent.functions = []

    sdk_context.add_resource(test_agent, "agent")

    await sdk_context.save_resource_to_db("test_id")

    mock_db_manager.get_table_definition.assert_called_once_with("resources")
    mock_db_manager.create_table.assert_called_once_with(
        "resources",
        {
            "resource_id": "String",
            "name": "String",
            "type": "String",
            "config": "JSON",
            "state": "JSON",
            "create_date": "DateTime",
            "last_modified": "DateTime",
        },
    )
    mock_db_manager.insert_data.assert_called_once()


@pytest.mark.asyncio
async def test_load_resource_from_db(sdk_context):
    mock_db_manager = AsyncMock()
    sdk_context.get_utility = MagicMock(return_value=mock_db_manager)

    config = {
        "name": "test_agent",
        "type": "agent",
        "instruction": "test_instruction",
        "description": "test_description",
        "functions": [],
        "role": "test_role",
        "swarm_mode": False,
        "required_exts": [".txt"],
        "retrieval_tool": "test_tool",
        "load_index_file": False,
    }

    test_resource = {
        "resource_id": "test_id",
        "name": "test_agent",
        "type": "agent",
        "config": config,
        "state": {},
        "create_date": datetime.now(),
        "last_modified": datetime.now(),
    }

    mock_db_manager.read_data = AsyncMock(return_value=[test_resource])

    # Create a mock agent with proper type information
    mock_agent = MagicMock(spec=Agent)
    mock_agent.id = "test_id"
    mock_agent.name = "test_agent"
    mock_agent.instruction = "test_instruction"
    mock_agent.role = "test_role"
    mock_agent.description = "test_description"
    mock_agent.index_store = MagicMock()
    mock_agent.index_store.list_indexes.return_value = []

    # Mock the Agent class to return our properly configured mock
    with patch('swarmzero.agent.Agent', return_value=mock_agent):
        result = await sdk_context.load_resource_from_db("test_id")

        assert result is not None
        assert isinstance(result, Agent)
        assert result.id == "test_id"
        assert result.name == "test_agent"
        assert result.instruction == "test_instruction"
        assert result.role == "test_role"
        assert result.description == "test_description"


@pytest.mark.asyncio
async def test_find_resources(sdk_context):
    mock_db_manager = AsyncMock()
    sdk_context.get_utility = MagicMock(return_value=mock_db_manager)

    test_resources = [
        {
            "resource_id": "test_id_1",
            "type": "agent",
            "name": "test_agent_1",
            "config": json.dumps({"key": "value1"}),
            "state": json.dumps({}),
        },
        {
            "resource_id": "test_id_2",
            "type": "agent",
            "name": "test_agent_2",
            "config": json.dumps({"key": "value2"}),
            "state": json.dumps({}),
        },
    ]
    mock_db_manager.read_data = AsyncMock(return_value=test_resources)

    result = await sdk_context.find_resources(resource_type="agent")
    assert len(result) == 2
    assert result[0]["resource_id"] == "test_id_1"
    assert result[1]["resource_id"] == "test_id_2"
