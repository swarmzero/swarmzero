import json
import os
import string
import signal
import asyncio
import uvicorn
import uuid
import logging
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import UploadFile, FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langtrace_python_sdk import inject_additional_attributes
from llama_index.core.agent import AgentRunner, ReActAgent
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from swarmzero.agent import Agent
from swarmzero.chat import ChatManager
from swarmzero.llms.llm import LLM
from swarmzero.llms.utils import llm_from_config_without_agent, llm_from_wrapper
from swarmzero.sdk_context import SDKContext
from swarmzero.server.routes.files import insert_files_to_index
from swarmzero.utils import tools_from_funcs
from swarmzero.server.routes.swarm_chat import setup_swarm_chat_routes


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
    __host: str
    __port: int
    __app: FastAPI
    __shutdown_event: Optional[asyncio.Event] = None
    def __init__(self,
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
        host: Optional[str] = None,
        port: Optional[int] = None,
        ):
        self.id = swarm_id if swarm_id != "" else str(uuid.uuid4())
        self.name = name
        self.description = description
        self.instruction = instruction
        self.functions = functions
        self.__host = host if host else "0.0.0.0"
        self.__port = port if port else 8000
        self.__shutdown_event = asyncio.Event()
        self.__agents = AgentMap()
        self.sdk_context = sdk_context if sdk_context is not None else SDKContext(config_path=config_path)
        self.__config = self.sdk_context.get_config(self.name)
        self.__llm = llm if llm is not None else llm_from_config_without_agent(self.__config, self.sdk_context)
        
        # resolve max_iterations: constructor override > per-agent config
        if max_iterations is not None:
            self.max_iterations = max_iterations
        elif isinstance(self.__config.get("max_iterations", None), int):
            self.max_iterations = self.__config.get("max_iterations", None) 
        
        self.__utilities_loaded = False
        self.sdk_context.load_default_utility()

        if agents is None:
            agents = self.sdk_context.generate_agents_from_config()
        if agents:
            for agent in agents:
                agent.swarm_id = self.id
                self.sdk_context.add_resource(agent, resource_type="agent")
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
            raise ValueError("No agents provided in params or config file")

        self.__app = FastAPI()
        self.__setup_server()

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
            callback_manager=self.sdk_context.get_utility("callback_manager"),
        )

    

    def add_agent(self, agent: Agent):
        if agent.name in self.__agents:
            raise ValueError(f"Agent `{agent.name}` already exists in the swarm.")
        agent.swarm_id = self.id
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
        files: Optional[List[UploadFile]] = [],
        verbose: bool = False,
    ):
        await self._ensure_utilities_loaded()
        await self.sdk_context.save_sdk_context_to_db()
        db_manager = self.sdk_context.get_utility("db_manager")

        chat_manager = ChatManager(self.__swarm, user_id=user_id, session_id=session_id, swarm_id=self.id)
        last_message = ChatMessage(role=MessageRole.USER, content=prompt)
        event_handler = self.sdk_context.get_utility("reasoning_callback") if verbose else None

        stored_files = []
        if files and len(files) > 0:
            stored_files = await insert_files_to_index(files, self.id, self.sdk_context)

        response = ""
        try:
            async for chunk in inject_additional_attributes(
                lambda: chat_manager.generate_response(
                    db_manager, last_message, stored_files, event_handler, stream_mode=False, verbose=verbose
                ),
                {"user_id": user_id},
            ):
                try:
                    if verbose:
                        if isinstance(chunk, dict):
                            response += f"0:{json.dumps(chunk)}\n"
                        elif isinstance(chunk, list):
                            response += f"2:{json.dumps(chunk)}\n"
                        else:
                            response += f"1:{json.dumps(chunk)}\n"
                    else:
                        if isinstance(chunk, str):
                            response += chunk
                except Exception as e:
                    print(f"Error processing chunk: {e}")
        finally:
            if hasattr(db_manager, "db"):
                await db_manager.db.close()

        return response

    async def chat_stream(
        self,
        prompt: str,
        user_id="default_user",
        session_id="default_chat",
        files: Optional[List[UploadFile]] = [],
        verbose: bool = False,
    ) -> StreamingResponse:
        await self._ensure_utilities_loaded()
        await self.sdk_context.save_sdk_context_to_db()
        db_manager = self.sdk_context.get_utility("db_manager")

        chat_manager = ChatManager(self.__swarm, user_id=user_id, session_id=session_id, swarm_id=self.id)
        last_message = ChatMessage(role=MessageRole.USER, content=prompt)
        event_handler = self.sdk_context.get_utility("reasoning_callback") if verbose else None
        stored_files = []
        if files and len(files) > 0:
            stored_files = await insert_files_to_index(files, self.id, self.sdk_context)

        async def stream_response():
            async for chunk in chat_manager.generate_response(
                db_manager, last_message, stored_files, event_handler, stream_mode=True, verbose=verbose
            ):
                if verbose:
                    if isinstance(chunk, dict):
                        yield f"0:{json.dumps(chunk)}\n"
                    else:
                        yield f"1:{json.dumps(chunk)}\n"
                else:
                    if isinstance(chunk, str):
                        yield f"data: {chunk}\n\n"

        return StreamingResponse(
            inject_additional_attributes(stream_response, {"user_id": user_id}),
            media_type="text/event-stream",
        )

    async def chat_history(self, user_id="", session_id=""):
        await self._ensure_utilities_loaded()
        db_manager = self.sdk_context.get_utility("db_manager")

        chat_manager = ChatManager(self.__swarm, user_id=user_id, session_id=session_id, swarm_id=self.id)

        if session_id:
            chats = await chat_manager.get_messages(db_manager)
        else:
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
            await self.sdk_context.load_db_manager()
            self.__utilities_loaded = True
    
    def __setup_server(self):
        self.__configure_cors()
        self.__setup_routes()
    
    def __configure_cors(self):
        self.__app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    def __setup_routes(self):
        router = APIRouter()
        setup_swarm_chat_routes(router, self, self.sdk_context)
        self.__app.include_router(router)

        @self.__app.get("/health")
        async def health_check():
            return {"status": "healthy"}

    async def run_server(self):
        try:
            self.__setup_signal_handlers()
            config = uvicorn.Config(app=self.__app,
                                    host=self.__host,
                                    port=self.__port,
                                    loop="asyncio")
            server = uvicorn.Server(config)
            await server.serve()
        except Exception as e:
            print(f"Error while running the server: {e}")
    
    def __setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self.__signal_handler)
    def __signal_handler(self, signum, _):
        asyncio.create_task(self.shutdown_procedures())
    
    async def shutdown_procedures(self):
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        [task.cancel() for task in tasks]
        await asyncio.gather(*tasks, return_exceptions=True)
        self.__shutdown_event.set()
