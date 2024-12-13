from llama_index.agent.openai import OpenAIAgent  # type: ignore
from llama_index.core.agent import ReActAgentWorker
from llama_index.core.agent.react_multimodal.step import MultimodalReActAgentWorker

from swarmzero.llms.llm import LLM
from swarmzero.sdk_context import SDKContext


class OpenAILLM(LLM):
    def __init__(self, llm=None, tools=None, instruction="", tool_retriever=None, sdk_context: SDKContext = None):
        super().__init__(llm, tools, instruction, tool_retriever)
        # self.agent = OpenAIAgent.from_tools(
        #     tools=self.tools,
        #     system_prompt=self.system_prompt,
        #     tool_retriever=tool_retriever,
        #     llm=llm,
        #     callback_manager=sdk_context.get_utility("callback_manager")
        # )
        self.agent = ReActAgentWorker.from_tools(
            tools=self.tools,
            system_prompt=self.system_prompt,
            llm=llm,
            verbose=True,
            tool_retriever=self.tool_retriever,
            callback_manager=sdk_context.get_utility("callback_manager"),
        ).as_agent()


class AzureOpenAILLM(LLM):
    def __init__(self, llm=None, tools=None, instruction="", tool_retriever=None, sdk_context: SDKContext = None):
        super().__init__(llm, tools, instruction, tool_retriever)
        self.agent = ReActAgentWorker.from_tools(
            tools=self.tools,
            system_prompt=self.system_prompt,
            llm=llm,
            verbose=True,
            tool_retriever=self.tool_retriever,
            callback_manager=sdk_context.get_utility("callback_manager"),
        ).as_agent()


class OpenAIMultiModalLLM(LLM):
    def __init__(
        self,
        llm=None,
        tools=None,
        instruction="",
        tool_retriever=None,
        max_iterations=10,
        sdk_context: SDKContext = None,
    ):
        super().__init__(llm, tools, instruction, tool_retriever)
        self.agent = MultimodalReActAgentWorker.from_tools(
            tools=self.tools,
            system_prompt=self.system_prompt,
            tool_retriever=self.tool_retriever,
            # llm=self.llm,
            multi_modal_llm=self.llm,
            max_iterations=max_iterations,
            callback_manager=sdk_context.get_utility("callback_manager"),
        ).as_agent()
