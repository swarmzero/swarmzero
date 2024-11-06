from unittest.mock import MagicMock, patch

import llama_index
import pytest
from llama_index.agent.openai import OpenAIAgent  # type: ignore
from llama_index.core.agent.runner.base import AgentRunner
from llama_index.core.objects import ObjectIndex
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.multi_modal_llms.openai import OpenAIMultiModal

from swarmzero.core.sdk_context import SDKContext
from swarmzero.core.services.llms.claude import ClaudeLLM
from swarmzero.core.services.llms.gemini import GeminiLLM
from swarmzero.core.services.llms.mistral import MistralLLM
from swarmzero.core.services.llms.ollama import OllamaLLM
from swarmzero.core.services.llms.openai import OpenAILLM, OpenAIMultiModalLLM


@pytest.fixture
def tools():
    return ["tool1", "tool2"]


@pytest.fixture
def instruction():
    return """Act as if you are a financial advisor"""


@pytest.fixture
def empty_tools():
    return []


@pytest.fixture
def required_exts():
    return [".txt", ".md"]


@pytest.fixture
def sdk_context():
    mock_sdk_context = MagicMock(spec=SDKContext)
    mock_sdk_context.get_utility.return_value = MagicMock()
    return mock_sdk_context


@pytest.fixture
def tool_retriever(tools):
    with (
        patch("llama_index.core.VectorStoreIndex.from_documents"),
        patch("llama_index.core.objects.ObjectIndex.from_objects"),
        patch("swarmzero.tools.retriever.base_retrieve.RetrieverBase.create_basic_index") as mock_create_basic_index,
    ):
        vectorstore_object = ObjectIndex.from_objects(
            tools,
            index=mock_create_basic_index,
        )
        tool_retriever = vectorstore_object.as_retriever(similarity_top_k=3)
        return tool_retriever


def test_openai_llm_initialization(tools, instruction, sdk_context):
    openai_llm = OpenAILLM(OpenAI(model="gpt-3.5-turbo"), tools, instruction, sdk_context=sdk_context)
    print(f"Agent: {openai_llm.agent}")
    print(f"Tools: {openai_llm.tools}")
    print(f"System Prompt: {openai_llm.system_prompt}")

    assert openai_llm.agent is not None
    assert isinstance(openai_llm.agent, AgentRunner)
    assert openai_llm.tools == tools
    assert instruction in openai_llm.system_prompt


def test_openai_multimodal_llm_initialization(tools, instruction, sdk_context):
    openai_llm = OpenAIMultiModalLLM(OpenAIMultiModal(model="gpt-4"), tools, instruction, sdk_context=sdk_context)
    assert openai_llm.agent is not None
    assert isinstance(openai_llm.agent, llama_index.core.agent.runner.base.AgentRunner)
    assert openai_llm.tools == tools
    assert instruction in openai_llm.system_prompt


def test_claude_llm_initialization(tools, instruction, sdk_context):
    claude_llm = ClaudeLLM(Anthropic(model="claude-3-opus-20240229"), tools, instruction, sdk_context=sdk_context)
    assert claude_llm.agent is not None
    assert isinstance(claude_llm.agent, llama_index.core.agent.runner.base.AgentRunner)
    assert claude_llm.tools == tools
    assert instruction in claude_llm.system_prompt


def test_llama_llm_initialization(tools, instruction, sdk_context):
    llama_llm = OllamaLLM(Ollama(model="llama3"), tools, instruction, sdk_context=sdk_context)
    assert llama_llm.agent is not None
    assert isinstance(llama_llm.agent, llama_index.core.agent.runner.base.AgentRunner)
    assert llama_llm.tools == tools
    assert instruction in llama_llm.system_prompt


def test_mistral_llm_initialization(tools, instruction, sdk_context):
    mistral_llm = MistralLLM(
        MistralAI(model="mistral-large-latest", api_key="mistral_api_key"), tools, instruction, sdk_context=sdk_context
    )
    assert mistral_llm.agent is not None
    assert isinstance(mistral_llm.agent, llama_index.core.agent.runner.base.AgentRunner)
    assert mistral_llm.tools == tools
    assert instruction in mistral_llm.system_prompt


def test_gemini_llm_initialization(tools, instruction, sdk_context):
    with patch('llama_index.llms.gemini.Gemini') as mock_gemini:
        gemini_llm = GeminiLLM(mock_gemini.return_value, tools, instruction, sdk_context=sdk_context)
        assert gemini_llm.agent is not None
        assert isinstance(gemini_llm.agent, llama_index.core.agent.runner.base.AgentRunner)
        assert gemini_llm.tools == tools
        assert instruction in gemini_llm.system_prompt


def test_retrieval_openai_llm_initialization(empty_tools, instruction, tool_retriever, sdk_context):
    openai_llm = OpenAILLM(
        OpenAI(model="gpt-3.5-turbo"), empty_tools, instruction, tool_retriever=tool_retriever, sdk_context=sdk_context
    )
    assert openai_llm.agent is not None
    assert isinstance(openai_llm.agent, AgentRunner)
    assert openai_llm.tools == empty_tools
    assert instruction in openai_llm.system_prompt
    assert openai_llm.tool_retriever == tool_retriever


def test_retrieval_ollamallm_initialization(empty_tools, instruction, tool_retriever, sdk_context):
    ollamallm = OllamaLLM(
        Ollama(model="llama3"), empty_tools, instruction, tool_retriever=tool_retriever, sdk_context=sdk_context
    )
    assert ollamallm.agent is not None
    assert isinstance(ollamallm.agent, llama_index.core.agent.runner.base.AgentRunner)
    assert ollamallm.tools == empty_tools
    assert instruction in ollamallm.system_prompt
    assert ollamallm.tool_retriever == tool_retriever
