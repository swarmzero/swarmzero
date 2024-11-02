from llama_index.core.agent import ReActAgentWorker
from llama_index.core.settings import Settings

from swarmzero.llms.llm import LLM
from swarmzero.sdk_context import SDKContext


class OllamaLLM(LLM):
    def __init__(self, llm=None, tools=None, instruction="", tool_retriever=None, sdk_context: SDKContext = None):
        super().__init__(llm, tools, instruction, tool_retriever)
        self.agent = ReActAgentWorker.from_tools(
            tools=self.tools,
            system_prompt=self.system_prompt,
            llm=llm,
            # verbose=True,
            tool_retriever=self.tool_retriever,
            callback_manager=sdk_context.get_utility("callback_manager"),
        ).as_agent()
