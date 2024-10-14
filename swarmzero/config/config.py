import logging
import os

import toml


class Config:
    def __init__(self, config_path):
        self.config_path = self.resolve_path(config_path)
        self.config = self.load_config()

    def resolve_path(self, path):
        """Resolves the path relative to the current working directory."""
        if not os.path.isabs(path):
            return os.path.abspath(os.path.join(os.getcwd(), path))
        return path

    def load_config(self):
        """Loads the configuration file."""
        try:
            with open(self.config_path, "r") as f:
                config_data = toml.load(f)
            return config_data
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

    def get(self, section, key, default=None):
        """Gets a value from the configuration."""
        return self.config.get(section, {}).get(key, default)

    def set(self, section, key, value):
        """Sets a value in the configuration."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self.save_config()

    def save_config(self):
        """Saves the current configuration state to disk."""
        with open(self.config_path, "w") as f:
            toml.dump(self.config, f)

    def get_log_level(self):
        SWARMZERO_LOG_LEVEL = os.getenv("SWARMZERO_LOG_LEVEL", "INFO").upper()
        return getattr(logging, SWARMZERO_LOG_LEVEL, logging.INFO)
