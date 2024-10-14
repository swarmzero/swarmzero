import os
import string
import uuid
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from langtrace_python_sdk import inject_additional_attributes
from llama_index.core.agent import AgentRunner, ReActAgent
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from swarmzero.agent import Agent
from swarmzero.chat import ChatManager
from swarmzero.llms.llm import LLM
from swarmzero.llms.utils import llm_from_config_without_agent, llm_from_wrapper
from swarmzero.sdk_context import SDKContext
from swarmzero.utils import tools_from_funcs

load_dotenv()


class AgentMap(Dict[str, Dict[str, Any]]):
    def __init__(self):
        super().__init__()


class Swarm:
    id: str
    name: str
    instruction: str
    description: str
    __llm: LLM
    __agents: AgentMap
    __swarm: AgentRunner

    def __init__(
        self,
        name: str,
        description: str,
        instruction: str,
        functions: List[Callable],
        agents: Optional[List[Agent]] = None,
        llm: Optional[LLM] = None,
        config_path="./swarmzero_config_example.toml",
        swarm_id=os.getenv("SWARM_ID", ""),
        sdk_context: Optional[SDKContext] = None,
        max_iterations: Optional[int] = 10,
    ):
        self.id = swarm_id if swarm_id != "" else str(uuid.uuid4())
        self.name = name
        self.description = description
        self.instruction = instruction
        self.__agents = AgentMap()
        self.functions = functions
        self.sdk_context = sdk_context if sdk_context is not None else SDKContext(config_path=config_path)
        self.__config = self.sdk_context.get_config(self.name)
        self.__llm = llm if llm is not None else llm_from_config_without_agent(self.__config)
        self.max_iterations = max_iterations
        self.__utilities_loaded = False

        if agents is None:
            agents = self.sdk_context.generate_agents_from_config()

        if agents:
            for agent in agents:
                self.__agents[agent.name] = {
                    "id": agent.id,
                    "agent": agent,
                    "role": agent.role,
                    "description": agent.description,
                    "sdk_context": self.sdk_context,
                }

            self.sdk_context.add_resource(self, resource_type="swarm")
            self._build_swarm()
        else:
            raise ValueError("no agents provided in params or config file")

    def _build_swarm(self):
        query_engine_tools = (
            [
                QueryEngineTool(
                    query_engine=agent_data["agent"],
                    metadata=ToolMetadata(
                        name=self._format_tool_name(agent_name),
                        description=agent_data["description"],
                    ),
                )
                for agent_name, agent_data in self.__agents.items()
            ]
            if self.__agents
            else []
        )

        custom_tools = tools_from_funcs(funcs=self.functions)
        tools = custom_tools + query_engine_tools

        self.__swarm = ReActAgent.from_tools(
            tools=tools,
            llm=llm_from_wrapper(self.__llm, self.__config),
            verbose=True,
            context=self.instruction,
            max_iterations=self.max_iterations,
        )

    def add_agent(self, agent: Agent):
        if agent.name in self.__agents:
            raise ValueError(f"Agent `{agent.name}` already exists in the swarm.")
        self.__agents[agent.name] = {
            "id": agent.id,
            "agent": agent,
            "role": agent.role,
            "description": agent.description,
        }
        self.sdk_context.add_resource(agent, resource_type="agent")
        self._build_swarm()

    def remove_agent(self, name: str):
        if name not in self.__agents:
            raise ValueError(f"Agent `{name}` does not exist in the swarm.")
        del self.__agents[name]
        self._build_swarm()

    async def chat(
        self,
        prompt: str,
        user_id="default_user",
        session_id="default_chat",
        image_document_paths: Optional[List[str]] = [],
    ):
        await self._ensure_utilities_loaded()
        db_manager = self.sdk_context.get_utility("db_manager")

        chat_manager = ChatManager(self.__swarm, user_id=user_id, session_id=session_id)
        last_message = ChatMessage(role=MessageRole.USER, content=prompt)

        response = await inject_additional_attributes(
            lambda: chat_manager.generate_response(db_manager, last_message, image_document_paths), {"user_id": user_id}
        )
        return response

    async def chat_history(self, user_id="default_user", session_id="default_chat") -> dict[str, list]:
        await self._ensure_utilities_loaded()
        db_manager = self.sdk_context.get_utility("db_manager")

        chat_manager = ChatManager(self.__swarm, user_id=user_id, session_id=session_id)

        chats = await chat_manager.get_all_chats_for_user(db_manager)
        return chats

    def _format_tool_name(self, name: str) -> str:
        tmp = name.replace(" ", "_").replace("-", "_").lower()
        exclude = string.punctuation.replace("_", "")
        translation_table = str.maketrans("", "", exclude)
        result = tmp.translate(translation_table)

        return result

    async def _ensure_utilities_loaded(self):
        """Load utilities if they are not already loaded."""
        if not self.__utilities_loaded:
            await self.sdk_context.load_default_utility()
            self.__utilities_loaded = True
