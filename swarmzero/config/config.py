# config_manager.py

import json
import logging
import os
from typing import Any, Dict, Optional

import toml
from sqlalchemy import JSON as SQLAlchemyJSON
from sqlalchemy import Column, Integer, String, create_engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, sessionmaker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SQLAlchemy setup for synchronous operations
DATABASE_URL = os.getenv("SWARMZERO_DATABASE_URL", "sqlite:///config.db")
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Configuration(Base):
    """
    SQLAlchemy model for the configurations table.
    Each row represents a configuration stored as a JSON blob with a unique ID.
    """

    __tablename__ = "configurations"

    id = Column(Integer, primary_key=True, index=True)
    config = Column(SQLAlchemyJSON, nullable=False, default={})

    def __repr__(self):
        return f"<Configuration(id={self.id}, config={self.config})>"


# Create the configurations table if it doesn't exist
Base.metadata.create_all(bind=engine)


class DatabaseManager:
    """Handles database operations for configuration storage."""

    def __init__(self):
        self.session = SessionLocal()

    def get_config(self, config_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve the configuration dictionary for a given ID.

        :param config_id: Unique identifier for the configuration.
        :return: Configuration dictionary or None if not found.
        """
        try:
            stmt = select(Configuration).where(Configuration.id == config_id)
            result = self.session.execute(stmt).scalar_one_or_none()
            if result:
                logger.debug(f"Retrieved from DB (ID={config_id}): {result.config}")
                return result.config
            logger.debug(f"No DB entry found for ID={config_id}")
            return None
        except SQLAlchemyError as e:
            logger.error(f"Database error during get_config: {e}")
            return None

    def set_config(self, config_id: int, config_dict: Dict[str, Any]) -> None:
        """
        Insert or update the configuration dictionary for a given ID.

        :param config_id: Unique identifier for the configuration.
        :param config_dict: Configuration dictionary to store.
        """
        try:
            stmt = select(Configuration).where(Configuration.id == config_id)
            config_entry = self.session.execute(stmt).scalar_one_or_none()
            if config_entry:
                config_entry.config = config_dict
                logger.debug(f"Updated DB entry (ID={config_id}): {config_dict}")
            else:
                config_entry = Configuration(id=config_id, config=config_dict)
                self.session.add(config_entry)
                logger.debug(f"Added new DB entry (ID={config_id}): {config_dict}")
            self.session.commit()
            logger.info(f"Configuration with ID={config_id} saved to Database.")
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Database error during set_config: {e}")
            raise


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
    ):
        """
        Initialize the Config object.

        :param source_type: Type of the configuration source ('json', 'toml', 'db')
        :param config_path: Path to the config file (required for 'json' and 'toml')
        :param config_id: Unique ID for the configuration (required for 'db')
        """
        self.source_type = source_type.lower()
        self.config_path = config_path
        self.config_id = config_id
        self.db_manager = DatabaseManager() if self.source_type == "db" else None
        self.config_cache: Dict[str, Dict[str, Any]] = {}

        if self.source_type in ["json", "toml"]:
            if not self.config_path:
                raise ValueError("config_path is required for JSON and TOML sources.")
            self.load_config_file()
        elif self.source_type == "db":
            if self.config_id is None:
                raise ValueError("config_id is required for Database source.")
            self.load_config_db()
        else:
            raise ValueError(f"Unsupported source_type: {self.source_type}")

    def load_config_file(self):
        """Load configuration from a JSON or TOML file."""
        if self.source_type == "json":
            loader = self._load_json
            dumper = self._dump_json
        else:
            loader = self._load_toml
            dumper = self._dump_toml

        try:
            with open(self.config_path, "r") as f:
                self.config_cache = loader(f)
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

        self._dumper = dumper

    def _load_json(self, file) -> Dict[str, Any]:
        return json.load(file)

    def _dump_json(self, file, data: Dict[str, Any]) -> None:
        json.dump(data, file, indent=4)

    def _load_toml(self, file) -> Dict[str, Any]:
        return toml.load(file)

    def _dump_toml(self, file, data: Dict[str, Any]) -> None:
        toml.dump(data, file)

    def load_config_db(self):
        """Load configuration from the database."""
        try:
            db_config = self.db_manager.get_config(self.config_id)
            if db_config is not None:
                self.config_cache = db_config
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

    def save(self) -> None:
        """
        Save the current configuration state to the source (file or database).
        """
        if self.source_type in ["json", "toml"]:
            self.save_config_file()
        elif self.source_type == "db":
            self.save_config_db()
        else:
            logger.error(f"Unsupported source_type: {self.source_type}")

    def save_config_file(self):
        """Save configuration back to the JSON or TOML file."""
        try:
            with open(self.config_path, "w") as f:
                self._dumper(f, self.config_cache)
            logger.info(f"Configuration saved to {self.source_type.upper()} file.")
        except Exception as e:
            logger.error(f"Error saving config file: {e}")

    def save_config_db(self):
        """Save configuration back to the database."""
        try:
            self.db_manager.set_config(self.config_id, self.config_cache)
        except Exception as e:
            logger.error(f"Error saving config to DB: {e}")

    def get_log_level(self) -> int:
        """
        Retrieve the log level from environment variables or default to INFO.

        :return: Logging level
        """
        SWARMZERO_LOG_LEVEL = os.getenv("SWARMZERO_LOG_LEVEL", "INFO").upper()
        return getattr(logging, SWARMZERO_LOG_LEVEL, logging.INFO)


# Example Usage
if __name__ == "__main__":
    # Example for TOML
    toml_config_path = "./swarmzero_config_example.toml"
    toml_config = Config(source_type="toml", config_path=toml_config_path)
    toml_config.set("database", "url", "postgresql://user:password@localhost/dbname")
    toml_config.save()
    print("TOML Database URL:", toml_config.get("database", "url"))

    # Example for JSON
    json_config_path = "config.json"
    json_config = Config(source_type="json", config_path=json_config_path)
    json_config.set("api", "key", "new_api_key")
    json_config.save()
    print("JSON API Key:", json_config.get("api", "key"))

    # Example for Database
    db_config_id = 1  # Unique ID for the configuration in the database
    db_config = Config(source_type="db", config_id=db_config_id)
    db_config.set("service", "enabled", True)
    db_config.save()
    print("DB Service Enabled:", db_config.get("service", "enabled"))

    # Retrieve log level and set it
    # You can set the environment variable SWARMZERO_LOG_LEVEL to control the logging level
    # e.g., export SWARMZERO_LOG_LEVEL=DEBUG
    selected_config = toml_config  # Choose which config to use for log level
    logger.setLevel(selected_config.get_log_level())
    logger.info("This is an info log message.")
    logger.debug("This is a debug log message.")
