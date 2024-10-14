import importlib
import json
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import MetaData, create_engine
from sqlalchemy.sql import text as sql_text

from swarmzero.config import Config
from swarmzero.database.database import (
    DatabaseManager,
    db_url,
    get_db,
    initialize_db,
    setup_chats_table,
)


class SDKContext:

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    """
    A context class to provide the necessary environment for task execution.
    This includes configuration settings, resources, and utilities.
    """

    def __init__(self, config_path: Optional[str] = "./swarmzero_config_example.toml"):
        """
        Initialize the SDKContext with a path to a TOML configuration file.

        :param config_path: Path to the TOML configuration file.
        """
        self.config = Config(config_path)
        self.default_config = self.load_default_config()
        self.agent_configs = self.load_agent_configs()
        self.resources = {}
        self.resources_info = {}
        self.utilities = {}
        self.attributes = {}

    def load_default_config(self):
        """
        Load the default configuration settings.

        :return: A dictionary with default configuration settings.
        """
        return {
            "model": self.config.get("model", "model", "gpt-3.5-turbo"),
            "environment": self.config.get("environment", "type", "dev"),
            "timeout": self.config.get("timeout", "llm", 30),
            "log": self.config.get("log", "level", "INFO"),
            "ollama_server_url": self.config.get("model", "ollama_server_url", "http://localhost:11434"),
            "enable_multi_modal": self.config.get("model", "enable_multi_modal", False),
            "sample_prompts": self.config.get("sample_prompts", "prompts", []),
        }

    def load_agent_configs(self):
        """
        Load configurations for each agent from the configuration file.

        :return: A dictionary of agent configurations.
        """
        agent_configs = {}
        for section in self.config.config:
            if section not in ["model", "environment", "timeout", "log"]:
                agent_configs[section] = {
                    "model": self.config.get(section, "model", self.default_config["model"]),
                    "environment": self.config.get(section, "environment", self.default_config["environment"]),
                    "timeout": self.config.get(section, "timeout", self.default_config["timeout"]),
                    "log": self.config.get(section, "log", self.default_config["log"]),
                    "ollama_server_url": self.config.get(
                        section, "ollama_server_url", self.default_config["ollama_server_url"]
                    ),
                    "enable_multi_modal": self.config.get(
                        section, "enable_multi_modal", self.default_config["enable_multi_modal"]
                    ),
                    "sample_prompts": self.config.get(section, "sample_prompts", self.default_config["sample_prompts"]),
                    "tools": self.config.get(section, "tools", []),
                    "instruction": self.config.get(section, "instruction", ""),
                }
        return agent_configs

    def add_agent_config(self, file_path):
        agent_config = Config(file_path)
        for section in agent_config.config:
            if section not in ["model", "environment", "timeout", "log"]:
                agent_config = {
                    "model": self.config.get(section, "model", self.default_config["model"]),
                    "environment": self.config.get(section, "environment", self.default_config["environment"]),
                    "timeout": self.config.get(section, "timeout", self.default_config["timeout"]),
                    "log": self.config.get(section, "log", self.default_config["log"]),
                    "ollama_server_url": self.config.get(
                        section, "ollama_server_url", self.default_config["ollama_server_url"]
                    ),
                    "enable_multi_modal": self.config.get(
                        section, "enable_multi_modal", self.default_config["enable_multi_modal"]
                    ),
                    "sample_prompts": self.config.get(section, "sample_prompts", self.default_config["sample_prompts"]),
                    "tools": self.config.get(section, "tools", []),
                    "instruction": self.config.get(section, "instruction", ""),
                }

        self.agent_configs[section] = agent_config

    def generate_agents_from_config(self):
        from swarmzero.agent import Agent

        """
        Generate agents from the configuration file.
        """
        agents = []
        for section in self.agent_configs:
            if section not in ["model", "environment", "timeout", "log"]:
                agent_config = self.agent_configs[section]
                agent = Agent(
                    name=section,
                    functions=[
                        getattr(importlib.import_module(func["module"]), func["name"])
                        for func in agent_config.get("tools")
                    ],
                    instruction=agent_config.get("instruction"),
                    role=agent_config.get("role"),
                    description=agent_config.get("description"),
                    swarm_mode=True,
                    sdk_context=self,
                )
                agents.append(agent)
                self.add_resource(agent, resource_type="agent")

        return agents

    def get_config(self, agent: str):
        """
        Retrieve a configuration object for a specific agent.

        :param agent: Name of the agent.
        :return: A configuration dictionary for the requested agent.
        """
        return self.agent_configs.get(agent, self.default_config)

    def set_config(self, agent: str, key: str, value: Any):
        """
        Set a configuration setting in the context for a specific agent.

        :param agent: Name of the agent.
        :param key: Key of the configuration setting.
        :param value: Value to be set.
        """
        if agent not in self.agent_configs:
            self.agent_configs[agent] = {}
        self.agent_configs[agent][key] = value
        self.config.set(agent, key, value)

    def save_config(self):
        """
        Save the current configuration to the file.
        """
        self.config.save_config()

    def load_config(self):
        """
        Load configuration from the file.
        """
        self.config.load_config()
        self.default_config = self.load_default_config()
        self.agent_configs = self.load_agent_configs()

    def add_resource(self, resource: Any, resource_type: str = "agent"):
        """
        Add a resource to the context. Automatically extracts fields from the resource.

        :param resource: The resource to be added.
        :param resource_type: Type of the resource ("agent", "swarm", or "tool").
        """
        from swarmzero.agent import Agent
        from swarmzero.swarm import Swarm

        if isinstance(resource, Agent) and resource_type == "agent":
            resource_info = {
                "id": resource.id,
                "name": resource.name,
                "config_path": resource.config_path,
                "type": resource_type,
                "host": resource._Agent__host,
                "port": resource._Agent__port,
                "instruction": resource.instruction,
                "role": resource.role,
                "description": resource.description,
                "swarm_mode": resource._Agent__swarm_mode,
                "retrieve": resource.retrieve,
                "required_exts": resource.required_exts,
                "retrieval_tool": resource.retrieval_tool,
                "load_index_file": resource.load_index_file,
                "functions": [{"module": func.__module__, "name": func.__name__} for func in resource.functions],
            }
            self.resources[resource.id] = {"init_params": resource_info, "object": resource}
            self.add_resource_info(resource_info)
            for function in resource.functions:
                self.add_resource(function, resource_type="tool")
        elif isinstance(resource, Swarm) and resource_type == "swarm":
            resource_info = {
                "id": resource.id,
                "name": resource.name,
                "type": resource_type,
                "instruction": resource.instruction,
                "description": resource.description,
                "agents": [agent["id"] for agent in resource._Swarm__agents.values()],
                "functions": [{"module": func.__module__, "name": func.__name__} for func in resource.functions],
            }
            self.resources[resource.id] = {"init_params": resource_info, "object": resource}
            self.add_resource_info(resource_info)
            for function in resource.functions:
                self.add_resource(function, resource_type="tool")
        elif resource_type == "tool" and callable(resource):
            resource_info = {"name": resource.__name__, "type": resource_type, "doc": resource.__doc__}
            self.resources[resource.__name__] = {"init_params": resource_info, "object": resource}
            self.add_resource_info(resource_info)
        else:
            raise ValueError("Unsupported resource type")

    def get_resource(self, key: str):
        """
        Retrieve a resource from the context.

        :param name: Name of the resource.
        :return: The requested resource.
        """
        resource = self.resources.get(key)
        return resource.get("object") if isinstance(resource, dict) else resource

    def add_resource_info(self, resource_info: dict):
        """
        Add resource information to the context.

        :param resource_info: Information about the resource to be added.
        """
        self.resources_info[resource_info["name"]] = resource_info

    def get_resource_info(self, name: str):
        """
        Retrieve resource information from the context.

        :param name: Name of the resource.
        :return: Information about the requested resource.
        """
        return self.resources_info.get(name)

    def save_sdk_context_json(self, file_path="sdk_context.json"):
        """
        Save the current SDK context to a file, excluding non-serializable objects.

        :param file_path: Path to the file where the context should be saved.
        """
        state = self.__dict__.copy()
        # Exclude non-serializable objects
        state["resources"] = {k: v["init_params"] if isinstance(v, dict) else v for k, v in state["resources"].items()}
        state["utilities"] = {k: v["info"] if isinstance(v, dict) else v for k, v in state["utilities"].items()}
        state.pop("config", None)
        state.pop("resources_info", None)

        with open(file_path, "w") as f:
            json.dump(state, f, default=str, indent=4)

    async def load_sdk_context_json(self, file_path="sdk_context.json"):
        """
        Load the SDK context from a file and restore non-serializable objects.

        :param file_path: Path to the file from which the context should be loaded.
        """
        with open(file_path, "r") as f:
            state = json.load(f)

        self.__dict__.update(state)
        self.default_config = state["default_config"]
        self.agent_configs = state["agent_configs"]
        self.resources = {
            k: {"init_params": v, "object": None} if isinstance(v, dict) else v for k, v in self.resources.items()
        }
        self.utilities = {}
        await self.load_default_utility()
        self.restore_non_serializable_objects()
        return self

    def restore_non_serializable_objects(self):
        """
        Restore non-serializable objects after loading the context.
        """
        from swarmzero.agent import Agent

        for name, resource in self.resources.items():
            if isinstance(resource, dict) and resource["init_params"]["type"] == "agent":
                params = resource["init_params"]
                functions = [
                    getattr(importlib.import_module(func["module"]), func["name"]) for func in params["functions"]
                ]

                resource_obj = Agent(
                    agent_id=params["id"],
                    name=params["name"],
                    sdk_context=self,
                    host=params["host"],
                    port=params["port"],
                    instruction=params["instruction"],
                    role=params["role"],
                    description=params["description"],
                    swarm_mode=params["swarm_mode"],
                    required_exts=params["required_exts"],
                    retrieval_tool=params["retrieval_tool"],
                    load_index_file=params["load_index_file"],
                    functions=functions,
                )
                self.resources[name]["object"] = resource_obj

    async def initialize_database(self):
        await initialize_db()

    async def save_sdk_context_to_db(self):
        await self.initialize_database()

        async for db in get_db():
            db_manager = DatabaseManager(db)

            table_exists = await db_manager.get_table_definition("sdkcontext")
            if table_exists == None:
                # Create a table to store configuration and resource details as JSON
                await db_manager.create_table(
                    "sdkcontext", {"type": "String", "data": "JSON", "create_date": "DateTime"}
                )

            # Prepare the data to be inserted
            config_data = {
                "default_config": self.default_config,
                "agent_configs": self.agent_configs,
                "resources": {k: v["init_params"] for k, v in self.resources.items()},
                "utilities": {k: v["info"] for k, v in self.utilities.items()},
            }

            # Insert the data into the table
            await db_manager.insert_data(
                "sdkcontext", {"type": "sdk_context", "data": config_data, "create_date": datetime.now()}
            )

    async def fetch_data(self, table_name: str, conditions: dict, order_by: str = "create_date", limit: int = 1):
        """
        Fetch data from the specified table based on conditions, with optional ordering and limiting.

        :param table_name: Name of the table.
        :param conditions: A dictionary of conditions to filter the records.
        :param order_by: Column name to order the results by create_date.
        :param limit: Maximum number of records to returns default is 1.
        :return: List of records matching the conditions.
        """
        query = f"SELECT * FROM {table_name} WHERE " + " AND ".join(
            [f"{k} = '{conditions[k]}'" for k in conditions.keys()]
        )
        if order_by:
            query += f" ORDER BY {order_by} DESC"
        if limit:
            query += f" LIMIT {limit}"

        print(query)
        result = await self.db_manager.execute(sql_text(query), conditions)
        return result.fetchall()

    async def load_sdk_context_from_db(self):
        """
        Load the SDK context from the database and restore non-serializable objects.
        """
        async for db in get_db():
            self.db_manager = DatabaseManager(db)

            # Fetch the configuration data from the database
            config_record = await self.fetch_data("sdkcontext", {"type": "sdk_context"})
            if config_record:
                print(config_record)
                state = json.loads(config_record[0][2])

                self.__dict__.update(state)
                self.default_config = state["default_config"]
                self.agent_configs = state["agent_configs"]
                self.resources = {
                    k: {"init_params": v, "object": None} if isinstance(v, dict) else v
                    for k, v in self.resources.items()
                }
                self.utilities = {}
                await self.load_default_utility()
                self.restore_non_serializable_objects()
                return self

    def set_attributes(self, id, **kwargs):
        """
        Set multiple attributes of the SDKContext at once.

        :param kwargs: Keyword arguments for attributes to set.
        """
        valid_attributes = [
            "llm",
            "tools",
            "tool_retriever",
            "agent_class",
            "instruction",
            "enable_multi_modal",
            "max_iterations",
        ]
        if id not in self.attributes:
            self.attributes[id] = {}
        for attr, value in kwargs.items():
            if attr in valid_attributes:
                self.attributes[id][attr] = value
            else:
                print(f"Warning: '{attr}' is not a valid attribute and was ignored.")

    def get_attributes(self, id, *args):
        """
        Get multiple attributes of the SDKContext at once.

        :param args: Names of attributes to retrieve.
        :return: A dictionary of requested attributes and their values.
        """
        valid_attributes = [
            "llm",
            "tools",
            "tool_retriever",
            "agent_class",
            "instruction",
            "enable_multi_modal",
            "max_iterations",
        ]
        return {attr: self.attributes[id].get(attr) for attr in args if attr in valid_attributes}

    def add_utility(self, utility, utility_type: str, name: str):
        """
        Add a utility to the context.

        :param utility: The utility to be added.
        :param util_type: The type of the utility.
        :param name: The name of the utility.
        """
        utility_info = {"name": name, "type": utility_type}

        # Add type-specific information
        if hasattr(utility, "url"):
            utility_info["url"] = utility.url

        self.utilities[name] = {"info": utility_info, "object": utility}

    def get_utility(self, name: str):
        """
        Retrieve a utility function from the context.

        :param name: Name of the utility function.
        :return: The requested utility function.
        """
        utility = self.utilities.get(name)
        if utility is None:
            return None
        return utility["object"]

    async def load_default_utility(self):
        """
        Load the default utility function from the context.

        :return: The requested utility function.
        """

        # if self.get_utility("text2sql_engine") is None:
        #     engine = create_engine(db_url.replace("+aiosqlite", "").replace("+asyncpg", ""))
        #     metadata = MetaData()
        #     metadata.reflect(bind=engine)
        #     self.add_utility(engine, utility_type="DBEngine", name="text2sql_engine")
        #     self.add_utility(metadata, utility_type="MetaData", name="text2sql_metadata")
        if self.get_utility("db_manager") is None:
            await initialize_db()
            async for db in get_db():
                await setup_chats_table(db)
                db_manager = DatabaseManager(db)
                self.add_utility(db_manager, utility_type="DatabaseManager", name="db_manager")
                break  # Exit after getting the first session
