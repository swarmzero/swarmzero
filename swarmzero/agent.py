import asyncio
import importlib.util
import logging
import os
import signal
import subprocess
import sys
import uuid
from typing import Callable, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langtrace_python_sdk import inject_additional_attributes  # type: ignore   # noqa
from llama_index.core.agent import AgentRunner  # noqa
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.objects import ObjectIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from swarmzero.chat import ChatManager
from swarmzero.llms.claude import ClaudeLLM
from swarmzero.llms.llm import LLM
from swarmzero.llms.mistral import MistralLLM
from swarmzero.llms.ollama import OllamaLLM
from swarmzero.llms.openai import OpenAILLM, OpenAIMultiModalLLM
from swarmzero.llms.utils import llm_from_config
from swarmzero.sdk_context import SDKContext
from swarmzero.server.models import ToolInstallRequest
from swarmzero.server.routes import files, setup_routes
from swarmzero.tools.retriever.base_retrieve import (
    IndexStore,
    RetrieverBase,
    index_base_dir,
    supported_exts,
)
from swarmzero.tools.retriever.chroma_retrieve import ChromaRetriever
from swarmzero.tools.retriever.pinecone_retrieve import PineconeRetriever
from swarmzero.utils import tools_from_funcs

load_dotenv()


class Agent:
    id: str
    name: str

    __llm: LLM
    __agent: AgentRunner

    def __init__(
        self,
        name: str,
        functions: List[Callable],
        llm: Optional[LLM] = None,
        config_path="./swarmzero_config_example.toml",
        host="0.0.0.0",
        port=8000,
        instruction="",
        role="",
        description="",
        agent_id=os.getenv("AGENT_ID", ""),
        retrieve=False,
        required_exts=supported_exts,
        retrieval_tool="basic",
        index_name: Optional[str] = None,
        load_index_file=False,
        swarm_mode=False,
        chat_only_mode=False,
        sdk_context: Optional[SDKContext] = None,
        max_iterations: Optional[int] = 10,
    ):
        self.id = agent_id if agent_id != "" else str(uuid.uuid4())
        self.name = name
        self.functions = functions
        self.config_path = config_path
        self.__host = host
        self.__port = port
        self.__app = FastAPI()
        self.shutdown_event = asyncio.Event()
        self.instruction = instruction
        self.role = role
        self.description = description
        self.sdk_context = sdk_context if sdk_context is not None else SDKContext(config_path=config_path)
        self.__config = self.sdk_context.get_config(self.name)
        self.__llm = llm if llm is not None else None
        self.max_iterations = max_iterations
        self.__optional_dependencies: dict[str, bool] = {}
        self.__swarm_mode = swarm_mode
        self.__chat_only_mode = chat_only_mode
        self.retrieve = retrieve
        self.required_exts = required_exts
        self.retrieval_tool = retrieval_tool
        self.index_name = index_name
        self.load_index_file = load_index_file
        logging.basicConfig(stream=sys.stdout, level=self.__config.get("log"))
        logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

        self.logger = logging.getLogger()
        self.logger.setLevel(self.__config.get("log"))

        self._check_optional_dependencies()
        self.__setup()

        self.sdk_context.add_resource(self, resource_type="agent")
        [self.sdk_context.add_resource(func, resource_type="tool") for func in self.functions]

        self.__utilities_loaded = False

    async def _ensure_utilities_loaded(self):
        """Load utilities if they are not already loaded."""
        if not self.__utilities_loaded and self.__chat_only_mode:
            await self.sdk_context.load_default_utility()
            self.__utilities_loaded = True

    def _check_optional_dependencies(self):
        try:
            from web3 import Web3  # noqa

            self.__optional_dependencies["web3"] = True
        except ImportError:
            self.__optional_dependencies["web3"] = False

    def is_dir_not_empty(self, path):
        if os.path.exists(path):
            if os.path.isfile(path):
                return os.path.getsize(path) > 0
            elif os.path.isdir(path):
                return bool(os.listdir(path))
        return False

    def __setup(self):

        self.get_indexstore()
        self.init_agent()

        if self.__swarm_mode is False and self.__chat_only_mode is False:
            self.__setup_server()

    def __setup_server(self):

        self.__configure_cors()
        setup_routes(self.__app, self.id, self.sdk_context)

        @self.__app.get("/health")
        def health():
            return {"status": "healthy"}

        @self.__app.post("/api/v1/install_tools")
        async def install_tool(tools: List[ToolInstallRequest]):
            try:
                print(f"now installing tools:\n{tools}")
                self.install_tools(tools)
                return {"status": "Tools installed successfully"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.__app.get("/api/v1/sample_prompts")
        async def sample_prompts():
            default_config = self.sdk_context.load_default_config()
            return {"sample_prompts": default_config["sample_prompts"]}

        signal.signal(signal.SIGINT, self.__signal_handler)
        signal.signal(signal.SIGTERM, self.__signal_handler)

    def __configure_cors(self):
        environment = self.__config.get("environment")  # default to 'development' if not set

        if environment == "dev":
            logger = logging.getLogger("uvicorn")
            logger.warning("Running in development mode - allowing CORS for all origins")
            self.__app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

    async def run_server(self):
        try:
            config = uvicorn.Config(app=self.__app, host=self.__host, port=self.__port, loop="asyncio")
            server = uvicorn.Server(config)
            await server.serve()
        except Exception as e:
            logging.error(f"unexpected error while running the server: {e}", exc_info=True)
        finally:
            await self.__cleanup()

    def run(self):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                logging.warning("Event loop already running, creating a new one.")
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                new_loop.run_until_complete(self.run_server())
            else:
                loop.run_until_complete(self.run_server())
        except Exception as e:
            logging.error(f"An error occurred in the main event loop: {e}", exc_info=True)

    async def chat(
        self,
        prompt: str,
        user_id="default_user",
        session_id="default_chat",
        image_document_paths: Optional[List[str]] = [],
    ):
        await self._ensure_utilities_loaded()
        db_manager = self.sdk_context.get_utility("db_manager")

        chat_manager = ChatManager(self.__agent, user_id=user_id, session_id=session_id)
        last_message = ChatMessage(role=MessageRole.USER, content=prompt)

        response = await inject_additional_attributes(
            lambda: chat_manager.generate_response(db_manager, last_message, image_document_paths), {"user_id": user_id}
        )
        return response

    async def chat_history(self, user_id="default_user", session_id="default_chat") -> dict[str, list]:
        await self._ensure_utilities_loaded()
        db_manager = self.sdk_context.get_utility("db_manager")

        chat_manager = ChatManager(self.__agent, user_id=user_id, session_id=session_id)

        chats = await chat_manager.get_all_chats_for_user(db_manager)
        return chats

    def query(self, *args, **kwargs):
        return self.__agent.query(*args, **kwargs)

    def aquery(self, *args, **kwargs):
        return self.__agent.aquery(*args, **kwargs)

    def __signal_handler(self, signum, frame):
        logging.info(f"signal {signum} received, initiating graceful shutdown...")
        asyncio.create_task(self.shutdown_procedures())

    async def shutdown_procedures(self):
        # attempt to complete or cancel all running tasks
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        [task.cancel() for task in tasks]

        await asyncio.gather(*tasks, return_exceptions=True)
        self.shutdown_event.set()
        logging.info("all tasks have been cancelled or completed")

    async def __cleanup(self):
        try:
            if hasattr(self, "db_session"):
                await self.db_session.close()
                logging.debug("database connection closed")
        except Exception as e:
            logging.error(f"error during cleanup: {e}", exc_info=True)
        finally:
            logging.info("cleanup process completed")

    def get_indexstore(self):
        is_base_dir_not_empty = self.is_dir_not_empty(files.BASE_DIR)
        is_index_dir_not_empty = self.is_dir_not_empty(index_base_dir)

        if is_index_dir_not_empty and self.load_index_file:
            self.index_store = IndexStore.load_from_file()
        else:
            self.index_store = IndexStore.get_instance()

        if is_base_dir_not_empty and self.retrieve:
            self.add_batch_indexes()

    def add_batch_indexes(self):

        if "basic" in self.retrieval_tool:
            retriever = RetrieverBase()
            index, file_names = retriever.create_basic_index()
            self.index_store.add_index(retriever.name, index, file_names)

        if "chroma" in self.retrieval_tool:
            chroma_retriever = ChromaRetriever()
            if self.index_name is not None:
                index, file_names = chroma_retriever.create_index(collection_name=self.index_name)
            else:
                index, file_names = chroma_retriever.create_index()
            self.index_store.add_index(chroma_retriever.name, index, file_names)

        if "pinecone-serverless" in self.retrieval_tool:
            pinecone_retriever = PineconeRetriever()
            if self.index_name is not None:
                index, file_names = pinecone_retriever.create_serverless_index(collection_name=self.index_name)
            else:
                index, file_names = pinecone_retriever.create_serverless_index()
            self.index_store.add_index(pinecone_retriever.name, index, file_names)

        if "pinecone-pod" in self.retrieval_tool:
            pinecone_retriever = PineconeRetriever()
            if self.index_name is not None:
                index, file_names = pinecone_retriever.create_pod_index(collection_name=self.index_name)
            else:
                index, file_names = pinecone_retriever.create_pod_index()
            self.index_store.add_index(pinecone_retriever.name, index, file_names)

            self.index_store.save_to_file()

    def get_tools(self):

        tools = tools_from_funcs(self.functions)

        return tools

    def init_agent(self):

        tools = self.get_tools()
        tool_retriever = None

        if self.load_index_file or self.retrieve or len(self.index_store.list_indexes()) > 0:
            index_store = IndexStore.get_instance()
            query_engine_tools = []
            for index_name in index_store.get_all_index_names():
                index_files = index_store.get_index_files(index_name)
                query_engine_tools.append(
                    QueryEngineTool(
                        query_engine=index_store.get_index(index_name).as_query_engine(),
                        metadata=ToolMetadata(
                            name=index_name + "_tool",
                            description=(
                                "Useful for questions related to specific aspects of " "documents" f" {index_files}"
                            ),
                        ),
                    )
                )

            tools = tools + query_engine_tools

            vectorstore_object = ObjectIndex.from_objects(
                tools,
                index=index_store.get_all_indexes(),
            )
            tool_retriever = vectorstore_object.as_retriever(similarity_top_k=3)
            tools = []  # Cannot specify both tools and tool_retriever
            self._assign_agent(tools, tool_retriever)
        else:
            self._assign_agent(tools, tool_retriever)

    def recreate_agent(self):
        return self.init_agent()

    def _assign_agent(self, tools, tool_retriever):
        if self.__llm is not None:
            print(f"using provided llm: {type(self.__llm)}")
            agent_class = type(self.__llm)
            llm = self.__llm

            self.sdk_context.set_attributes(
                id=self.id,
                llm=llm,
                tools=tools,
                tool_retriever=tool_retriever,
                agent_class=agent_class,
                instruction=self.instruction,
                max_iterations=self.max_iterations,
            )
            if agent_class == OpenAIMultiModalLLM:
                self.__agent = agent_class(
                    llm, tools, self.instruction, tool_retriever, max_iterations=self.max_iterations
                ).agent
            else:
                self.__agent = agent_class(llm, tools, self.instruction, tool_retriever).agent

        else:
            model = self.__config.get("model")
            enable_multi_modal = self.__config.get("enable_multi_modal")
            llm = llm_from_config(self.__config)

            if model.startswith("gpt-4") and enable_multi_modal is True:
                agent_class = OpenAIMultiModalLLM
            elif "gpt" in model:
                agent_class = OpenAILLM
            elif "claude" in model:
                agent_class = ClaudeLLM
            elif "llama" in model:
                agent_class = OllamaLLM
            elif "mixtral" in model or "mistral" in model or "codestral" in model:
                agent_class = MistralLLM
            else:
                agent_class = OpenAILLM

            self.sdk_context.set_attributes(
                id=self.id,
                llm=llm,
                tools=tools,
                tool_retriever=tool_retriever,
                agent_class=agent_class,
                instruction=self.instruction,
                enable_multi_modal=enable_multi_modal,
                max_iterations=self.max_iterations,
            )
            if agent_class == OpenAIMultiModalLLM:
                self.__agent = agent_class(
                    llm, tools, self.instruction, tool_retriever, max_iterations=self.max_iterations
                ).agent
            else:
                self.__agent = agent_class(llm, tools, self.instruction, tool_retriever).agent

    def add_tool(self, function_tool):
        self.functions.append(function_tool)
        self.recreate_agent()

    def install_tools(self, tools: List[ToolInstallRequest], install_path="swarmzero-data/tools"):
        """
        Install tools from a list of tool configurations.

        :param install_path: Path to the folder where the tools are installed
        :param tools: List of ToolInstallRequest objects where each contains:
                      - 'github_url': the GitHub URL of the tool repository.
                      - 'functions': list of paths to the functions to import.
                      - 'install_path': optional path where to install the tools.
                      - 'github_token': optional GitHub token for private repositories.
                      - 'env_vars': optional environment variables required for the tool to run.
        """
        os.makedirs(install_path, exist_ok=True)

        for tool in tools:
            if tool.env_vars is not None:
                for key, value in tool.env_vars:
                    os.environ[key] = value

            github_url = tool.github_url
            functions = tool.functions
            tool_install_path = install_path
            if tool.install_path is not None:
                tool_install_path = tool.install_path

            if tool.github_token:
                url_with_token = tool.url.replace("https://", f"https://{tool.github_token}@")
                github_url = url_with_token

            repo_dir = os.path.join(tool_install_path, os.path.basename(github_url))
            if not os.path.exists(repo_dir):
                subprocess.run(["git", "clone", github_url, repo_dir], check=True)

            for func_path in functions:
                module_name, func_name = func_path.rsplit(".", 1)
                module_path = os.path.join(repo_dir, *module_name.split(".")) + ".py"

                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                func = getattr(module, func_name)
                self.functions.append(func)
                print(f"Installed function: {func_name} from {module_name}")

        self.recreate_agent()
