from llama_index.agent.openai import OpenAIAgent  # type: ignore
from llama_index.core.agent import ReActAgentWorker
from llama_index.core.agent.react_multimodal.step import MultimodalReActAgentWorker
from llama_index.core.settings import Settings

from swarmzero.llms.llm import LLM


class OpenAILLM(LLM):
    def __init__(self, llm=None, tools=None, instruction="", tool_retriever=None):
        super().__init__(llm, tools, instruction, tool_retriever)
        self.agent = OpenAIAgent.from_tools(
            tools=self.tools,
            system_prompt=self.system_prompt,
            tool_retriever=tool_retriever,
            llm=llm,
        )
        # self.agent = ReActAgentWorker.from_tools(
        #     tools=self.tools,
        #     system_prompt=self.system_prompt,
        #     llm=llm,
        #     verbose=True,
        #     tool_retriever=self.tool_retriever,
        # ).as_agent()


class OpenAIMultiModalLLM(LLM):
    def __init__(self, llm=None, tools=None, instruction="", tool_retriever=None, max_iterations=10):
        super().__init__(llm, tools, instruction, tool_retriever)
        self.agent = MultimodalReActAgentWorker.from_tools(
            tools=self.tools,
            system_prompt=self.system_prompt,
            tool_retriever=self.tool_retriever,
            # llm=self.llm,
            multi_modal_llm=self.llm,
            max_iterations=max_iterations,
        ).as_agent()
