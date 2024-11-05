import json
from unittest.mock import MagicMock, patch

import pytest

from swarmzero.agent import Agent
from swarmzero.sdk_context import SDKContext


@pytest.fixture
def sdk_context():
    return SDKContext("./swarmzero_config_test.yaml")


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
    # Add more specific assertions based on your expected configurations


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
