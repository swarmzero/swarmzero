import importlib
import json
import os
from datetime import datetime
from typing import Any, Optional

from llama_index.core.callbacks import CallbackManager

from swarmzero.config import Config
from swarmzero.database.database import (
    DatabaseManager,
    get_db,
    initialize_db,
    setup_chats_table,
)
from swarmzero.utils import EventCallbackHandler, IndexStore


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

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the SDKContext with a TOML configuration file.

        If ``config_path`` is not provided, the default example configuration
        located in the project root will be used regardless of the current
        working directory.

        :param config_path: Path to the TOML configuration file.
        """
        if config_path is None:
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            config_path = os.path.join(root_dir, "swarmzero_config_example.toml")

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
        from swarmzero.workflow import Workflow

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
        elif isinstance(resource, Workflow) and resource_type == "workflow":
            resource_info = {
                "id": resource.id,
                "name": resource.name,
                "type": resource_type,
                "instruction": resource.instruction,
                "description": resource.description,
                "steps": [{"name": step.name, "mode": step.mode.name} for step in resource.steps],
            }
            self.resources[resource.id] = {"init_params": resource_info, "object": resource}
            self.add_resource_info(resource_info)
        elif resource_type == "tool" and callable(resource):
            resource_info = {"name": resource.__name__, "type": resource_type, "doc": resource.__doc__}
            self.resources[resource.__name__] = {"init_params": resource_info, "object": resource}
            self.add_resource_info(resource_info)
        else:
            raise ValueError("Unsupported resource type")

    def get_resource(self, key: str):
        """
        Retrieve a resource from the context.

        :param key: Key/ID of the resource.
        :return: The requested resource object.
        """
        resource = self.resources.get(key)
        if not resource:
            return None
        if isinstance(resource, dict):
            return resource.get("object")
        return resource

    def add_resource_info(self, resource_info: dict):
        """
        Add resource information to the context.

        :param resource_info: Information about the resource to be added.
        """
        self.resources_info[resource_info["name"]] = resource_info

    def get_resource_info(self, key: str):
        """
        Retrieve resource information from the context.

        :param key: Key/ID of the resource.
        :return: Information about the requested resource.
        """
        resource = self.resources.get(key)
        if isinstance(resource, dict):
            return resource.get("init_params")
        return None

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
        First restore all agents, then restore swarms that depend on those agents.
        """
        from swarmzero.agent import Agent
        from swarmzero.swarm import Swarm

        # First pass: Restore agents
        for name, resource in list(self.resources.items()):
            if not isinstance(resource, dict) or resource.get("object") is not None:
                continue

            params = resource["init_params"]
            if params["type"] != "agent":
                continue

            functions = [getattr(importlib.import_module(func["module"]), func["name"]) for func in params["functions"]]

            resource_obj = Agent(
                agent_id=params["id"],
                name=params["name"],
                sdk_context=self,
                instruction=params["instruction"],
                role=params["role"],
                description=params["description"],
                swarm_mode=params["swarm_mode"],
                required_exts=params["required_exts"],
                retrieval_tool=params["retrieval_tool"],
                load_index_file=params.get("load_index_file", False),
                functions=functions,
            )
            self.resources[name]["object"] = resource_obj

        # Second pass: Restore swarms (now that all agents are available)
        for name, resource in list(self.resources.items()):
            if not isinstance(resource, dict) or resource.get("object") is not None:
                continue

            params = resource["init_params"]
            if params["type"] != "swarm":
                continue

            functions = [getattr(importlib.import_module(func["module"]), func["name"]) for func in params["functions"]]

            # Get the agent objects that belong to this swarm
            agents = []
            for agent_id in params["agents"]:
                agent_resource = self.get_resource(agent_id)
                if not agent_resource:
                    raise ValueError(f"Agent {agent_id} not found when restoring swarm {params['name']}")
                agents.append(agent_resource)

            if not agents:
                raise ValueError(f"No agents found for swarm {params['name']}")

            resource_obj = Swarm(
                name=params["name"],
                description=params["description"],
                instruction=params["instruction"],
                functions=functions,
                agents=agents,
                swarm_id=params["id"],
                sdk_context=self,
            )
            self.resources[name]["object"] = resource_obj

        # Third pass: Restore workflows
        from swarmzero.workflow import StepMode, Workflow, WorkflowStep

        for name, resource in list(self.resources.items()):
            if not isinstance(resource, dict) or resource.get("object") is not None:
                continue

            params = resource["init_params"]
            if params["type"] != "workflow":
                continue

            steps = [
                WorkflowStep(runner=None, mode=StepMode[step["mode"]], name=step["name"])
                for step in params.get("steps", [])
            ]
            resource_obj = Workflow(
                name=params["name"],
                steps=steps,
                instruction=params.get("instruction", ""),
                description=params.get("description", ""),
                default_llm=None,
                sdk_context=self,
                workflow_id=params["id"],
            )
            self.resources[name]["object"] = resource_obj

    async def initialize_database(self):
        await initialize_db()

    async def save_sdk_context_to_db(self):
        """Save the current SDK context to the database."""
        db_manager = self.get_utility("db_manager")
        table_exists = await db_manager.get_table_definition("sdkcontext")
        if table_exists is None:
            # Create a table to store configuration and resource details as JSON
            await db_manager.create_table("sdkcontext", {"type": "String", "data": "JSON", "create_date": "DateTime"})

        # Prepare the data to be inserted
        sdk_context_data = {
            "default_config": self.default_config,
            "agent_configs": self.agent_configs,
            "resources": {k: v["init_params"] for k, v in self.resources.items()},
            "utilities": {k: v["info"] for k, v in self.utilities.items()},
        }

        # Insert the data into the table
        await db_manager.insert_data(
            "sdkcontext", {"type": "sdk_context", "data": sdk_context_data, "create_date": datetime.now()}
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
        db_manager = self.get_utility("db_manager")
        return await db_manager.read_data(table_name, filters=conditions, order_by=order_by, limit=limit)

    async def load_sdk_context_from_db(self):
        """
        Load the SDK context from the database and restore non-serializable objects.
        """
        # Fetch the configuration data from the database
        config_record = await self.fetch_data("sdkcontext", {"type": "sdk_context"})
        if config_record:
            print(config_record)
            state = json.loads(config_record[0][2])

            self.__dict__.update(state)
            self.default_config = state["default_config"]
            self.agent_configs = state["agent_configs"]
            self.resources = {
                k: {"init_params": v, "object": None} if isinstance(v, dict) else v for k, v in self.resources.items()
            }
            self.utilities = {}
            self.load_default_utility()
            await self.load_db_manager()
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

    def load_default_utility(self):
        """
        Load the default utility function from the context.

        :return: The requested utility function.
        """

        if self.get_utility("indexstore") is None:
            indexstore = IndexStore()
            self.add_utility(indexstore, 'IndexStore', 'indexstore')

        if self.get_utility("callback_manager") is None:
            reasoning_callback = EventCallbackHandler()
            callback_manager = CallbackManager(handlers=[reasoning_callback])
            self.add_utility(reasoning_callback, 'EventCallbackHandler', 'reasoning_callback')
            self.add_utility(callback_manager, 'CallbackManager', 'callback_manager')

    async def load_db_manager(self):
        if self.get_utility("db_manager") is None:
            await initialize_db()
            async for db in get_db():
                await setup_chats_table(db)
                db_manager = DatabaseManager(db)
                self.add_utility(db_manager, utility_type="DatabaseManager", name="db_manager")
                break  # Exit after getting the first session

    async def save_resource_to_db(self, resource_id: str):
        """Save a specific resource (agent/swarm) to the database."""
        db_manager = self.get_utility("db_manager")
        table_exists = await db_manager.get_table_definition("resources")
        if table_exists is None:
            # Create a table to store individual resources
            await db_manager.create_table(
                "resources",
                {
                    "resource_id": "String",  # Agent or Swarm ID
                    "name": "String",  # Name of the resource
                    "type": "String",  # 'agent' or 'swarm'
                    "config": "JSON",  # Resource configuration
                    "state": "JSON",  # Current state/runtime data
                    "create_date": "DateTime",
                    "last_modified": "DateTime",
                },
            )

        resource = self.resources.get(resource_id)
        if resource is None:
            raise ValueError(f"Resource {resource_id} not found")

        params = resource["init_params"]

        # If this is a swarm, save its agents first
        if params["type"] == "swarm":
            for agent_id in params["agents"]:
                # Check if agent exists in resources table
                agent_exists = await db_manager.read_data("resources", {"resource_id": [agent_id]})
                if not agent_exists:
                    # Get agent from sdkcontext resources
                    agent_resource = self.resources.get(agent_id)
                    if agent_resource:
                        # Save agent to resources table
                        await self.save_resource_to_db(agent_id)
                    else:
                        raise ValueError(f"Agent {agent_id} not found when saving swarm {params['name']}")

        # Split configuration and state
        config_data = {
            "name": params["name"],
            "type": params["type"],
            "instruction": params["instruction"],
            "description": params.get("description"),
            "functions": params["functions"],
        }

        if params["type"] == "agent":
            config_data.update(
                {
                    "role": params["role"],
                    "swarm_mode": params["swarm_mode"],
                    "required_exts": params["required_exts"],
                    "retrieval_tool": params["retrieval_tool"],
                    "load_index_file": params.get("load_index_file", False),
                }
            )
        elif params["type"] == "swarm":
            config_data["agents"] = params["agents"]

        # Prepare the resource data
        resource_data = {
            "resource_id": resource_id,
            "name": params["name"],
            "type": params["type"],
            "config": config_data,
            "state": {},  # Current state would go here
            "create_date": datetime.now(),
            "last_modified": datetime.now(),
        }

        # Insert or update the resource in the database
        await db_manager.insert_data("resources", resource_data)

    async def load_resource_from_db(self, resource_id: str):
        """Load a resource and its dependencies from the database."""
        db_manager = self.get_utility("db_manager")
        results = await db_manager.read_data("resources", {"resource_id": [resource_id]})

        if not results:
            return None

        resource_data = results[0]
        config = resource_data["config"]
        config["id"] = resource_data["resource_id"]

        # If this is a swarm, load its agents first
        if config["type"] == "swarm":
            for agent_id in config["agents"]:
                if agent_id not in self.resources:
                    agent = await self.load_resource_from_db(agent_id)
                    if not agent:
                        raise ValueError(f"Failed to load agent {agent_id} for swarm {config['name']}")

        # Add the resource to the context
        self.resources[resource_id] = {"init_params": config, "object": None}
        self.restore_non_serializable_objects()
        return self.get_resource(resource_id)

    async def find_resources(
        self, resource_type: Optional[str] = None, name: Optional[str] = None, limit: int = 10, offset: int = 0
    ):
        """
        Find resources based on type and/or name with pagination.

        :param resource_type: Optional filter by resource type ('agent' or 'swarm')
        :param name: Optional filter by resource name (supports partial matches)
        :param limit: Maximum number of results to return
        :param offset: Number of results to skip
        :return: List of matching resources
        """
        db_manager = self.get_utility("db_manager")

        # Build the query conditions
        conditions = {}
        if resource_type:
            conditions["type"] = [resource_type]
        if name:
            conditions["name"] = [name]

        # Get paginated results
        resources = await db_manager.read_data(
            "resources", filters=conditions, order_by="last_modified", limit=limit, offset=offset
        )

        return resources

    async def save_config_to_db(self):
        """Save the current configuration to the database."""
        db_manager = self.get_utility("db_manager")
        table_exists = await db_manager.get_table_definition("config")
        if table_exists is None:
            # Create a table to store configuration details as JSON
            await db_manager.create_table("config", {"name": "String", "data": "JSON", "create_date": "DateTime"})

        # Prepare the config data to be inserted
        config_data = {
            "default_config": self.default_config,
            "agent_configs": self.agent_configs,
        }

        # Insert the data into the table
        await db_manager.insert_data("config", {"name": "default", "data": config_data, "create_date": datetime.now()})
