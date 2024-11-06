import asyncio
import json
import logging
from functools import partial
from typing import Any, Dict, Optional

import toml

from swarmzero.core.services.database.database import (
    DatabaseManager,
    get_db,
    initialize_db,
)

logger = logging.getLogger(__name__)


class Config:
    """
    Unified configuration handler supporting JSON, TOML, and Database sources.
    For the database source, configurations are stored as JSON blobs with unique IDs.
    """

    def __init__(
        self,
        source_type: str,
        config_path: Optional[str] = None,
        config_id: Optional[int] = None,
        db_manager: Optional["DatabaseManager"] = None,
    ):
        """
        Initialize the Config object.

        :param source_type: Type of the configuration source ('json', 'toml', 'db')
        :param config_path: Path to the config file (required for 'json' and 'toml')
        :param config_id: Unique ID for the configuration (required for 'db')
        :param db_manager: Existing DatabaseManager instance (required for 'db')
        """
        self.source_type = source_type.lower()
        self.config_path = config_path
        self.config_id = config_id
        self.config_cache: Dict[str, Dict[str, Any]] = {}
        self.db_manager = db_manager

        if self.source_type in ["json", "toml"]:
            if not self.config_path:
                raise ValueError("config_path is required for JSON and TOML sources.")
        elif self.source_type == "db":
            if self.config_id is None:
                raise ValueError("config_id is required for Database source.")
            if self.db_manager is None:
                raise ValueError("db_manager is required for Database source.")
        else:
            raise ValueError(f"Unsupported source_type: {self.source_type}")

    async def load_config(self):
        """
        Load configuration from the specified source.
        """
        if self.source_type in ["json", "toml"]:
            await self._load_config_file()
        elif self.source_type == "db":
            await self._load_config_db()

    async def _load_config_file(self):
        """
        Asynchronously load configuration from a JSON or TOML file.
        """
        try:
            if self.source_type == "json":
                await self._load_json()
            else:
                await self._load_toml()
            logger.info(f"Configuration loaded from {self.source_type.upper()} file.")
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}. Starting with empty config.")
            self.config_cache = {}
        except (json.JSONDecodeError, toml.TomlDecodeError) as e:
            logger.error(f"Error loading config file: {e}. Starting with empty config.")
            self.config_cache = {}
        except Exception as e:
            logger.error(f"Unexpected error loading config file: {e}.")
            self.config_cache = {}

    async def _load_json(self):
        """
        Load JSON configuration from a file asynchronously.
        """
        loop = asyncio.get_running_loop()
        with open(self.config_path, "r") as f:
            self.config_cache = await loop.run_in_executor(None, json.load, f)

    async def _load_toml(self):
        """
        Load TOML configuration from a file asynchronously.
        """
        loop = asyncio.get_running_loop()
        with open(self.config_path, "r") as f:
            self.config_cache = await loop.run_in_executor(None, toml.load, f)

    async def setup_configurations_table_if_missing(self):
        table_exists = await self.db_manager.get_table_definition("configurations")

        if table_exists:
            logger.info("Table 'configurations' already exists. Skipping creation.")
            return

        columns = {
            "config": "JSON",
        }

        await self.db_manager.create_table("configurations", columns)
        logger.info("Table 'configurations' created successfully.")

    async def _load_config_db(self):
        """
        Load configuration from the database using the provided DatabaseManager.
        """
        try:
            # Assuming configurations are stored in a table named 'configurations'
            # with columns 'id' and 'config'
            await self.setup_configurations_table_if_missing()
            records = await self.db_manager.read_data("configurations", {"id": [self.config_id]})
            if records:
                # Assuming the 'config' field contains the JSON configuration
                self.config_cache = records[0].get("config", {})
                logger.info(f"Configuration loaded from Database (ID={self.config_id}).")
            else:
                self.config_cache = {}
                logger.warning(
                    f"No configuration found in Database with ID={self.config_id}. Starting with empty config."
                )
        except Exception as e:
            logger.error(f"Error loading config from DB: {e}. Starting with empty config.")
            self.config_cache = {}

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value.

        :param section: Configuration section
        :param key: Configuration key
        :param default: Default value if key is not found
        :return: Configuration value
        """
        return self.config_cache.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set a configuration value.

        :param section: Configuration section
        :param key: Configuration key
        :param value: New value to set
        """
        if section not in self.config_cache:
            self.config_cache[section] = {}
        self.config_cache[section][key] = value
        logger.debug(f"Set [{section}][{key}] = {value}")

    async def save(self):
        """
        Save the current configuration state to the source (file or database).
        """
        if self.source_type in ["json", "toml"]:
            await self._save_config_file()
        elif self.source_type == "db":
            await self._save_config_db()
        else:
            logger.error(f"Unsupported source_type: {self.source_type}")

    async def _save_config_file(self):
        """
        Asynchronously save configuration to a JSON or TOML file.
        """
        try:
            if self.source_type == "json":
                await self._dump_json()
            else:
                await self._dump_toml()
            logger.info(f"Configuration saved to {self.source_type.upper()} file.")
        except Exception as e:
            logger.error(f"Error saving config file: {e}")

    async def _dump_json(self):
        """
        Dump JSON configuration to a file asynchronously.
        """
        loop = asyncio.get_running_loop()
        with open(self.config_path, "w") as f:
            await loop.run_in_executor(None, partial(json.dump, self.config_cache, f, indent=4))

    async def _dump_toml(self):
        """
        Dump TOML configuration to a file asynchronously.
        """
        loop = asyncio.get_running_loop()
        with open(self.config_path, "w") as f:
            await loop.run_in_executor(None, toml.dump, self.config_cache, f)

    async def _save_config_db(self):
        """
        Save configuration to the database using the provided DatabaseManager.
        """
        try:
            await self.setup_configurations_table_if_missing()
            # Check if configuration with config_id exists
            existing_records = await self.db_manager.read_data("configurations", {"id": [self.config_id]})
            if existing_records:
                # Update existing configuration
                await self.db_manager.update_data("configurations", self.config_id, {"config": self.config_cache})
            else:
                # Insert new configuration
                await self.db_manager.insert_data("configurations", {"id": self.config_id, "config": self.config_cache})
            logger.info(f"Configuration with ID={self.config_id} saved to Database.")
        except Exception as e:
            logger.error(f"Error saving config to DB: {e}")


async def main():
    await initialize_db()

    # Example: Using JSON as configuration source
    json_config_path = "config.json"
    json_config = Config(source_type="json", config_path=json_config_path)
    await json_config.load_config()
    json_config.set("agent_settings", "model", "gpt-6")
    json_config.set("agent_settings", "environment", "production")
    await json_config.save()
    print("JSON Config - Model:", json_config.get("agent_settings", "model"))

    # Example: Using TOML as configuration source
    toml_config_path = "config.toml"
    toml_config = Config(source_type="toml", config_path=toml_config_path)
    await toml_config.load_config()
    toml_config.set("agent_settings", "timeout", 60)
    toml_config.set("agent_settings", "log", "INFO")
    await toml_config.save()
    print("TOML Config - Timeout:", toml_config.get("agent_settings", "timeout"))

    # Example: Using Database as configuration source
    async with get_db() as db:
        db_manager = DatabaseManager(db)
        db_config = Config(source_type="db", config_id=1, db_manager=db_manager)
        db_config_2 = Config(source_type="db", config_id=2, db_manager=db_manager)
        db_config_3 = Config(source_type="db", config_id=3, db_manager=db_manager)
        await db_config.load_config()
        await db_config_2.load_config()
        await db_config_3.load_config()

        db_config.set("agent_settings", "ollama_server_url", "http://localhost:1234")
        db_config_2.set("agent_settings", "ollama_server_url", "http://localhost:4568")
        db_config.set("agent_settings", "enable_multi_modal", True)
        db_config_3.set("agent_settings", "enable_multi_modal", False)

        await db_config.save()
        print("DB Config - Ollama Server URL:", db_config.get("agent_settings", "ollama_server_url"))
        print("DB Config - Ollama Server URL:", db_config_2.get("agent_settings", "ollama_server_url"))
        print("DB Config - agent_settings:", db_config_3.get("agent_settings", "enable_multi_modal"))


if __name__ == "__main__":
    asyncio.run(main())
