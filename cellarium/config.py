"""
Configuration management for Cellarium.

This module defines the Config dataclass that holds all application settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """
    Application configuration.

    Attributes:
        data_path: Path to the .h5ad file containing AnnData object
        db_path: Path to SQLite database for state persistence
        host: Host address to bind the server
        port: Port number for the server
        debug: Enable debug mode with hot reloading
        expression_cache_size: Maximum number of genes to cache in memory
        theme: Default theme ('light' or 'dark')
    """

    data_path: Path
    db_path: Path = field(default_factory=lambda: Path.cwd() / "cellarium.db")
    host: str = "127.0.0.1"
    port: int = 8050
    debug: bool = False

    # Performance tuning
    expression_cache_size: int = 100  # Max genes to cache in LRU cache

    # UI settings
    theme: str = "light"  # 'light' or 'dark'

    def __post_init__(self):
        """Validate and normalize paths."""
        self.data_path = Path(self.data_path).resolve()
        self.db_path = Path(self.db_path).resolve()

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        if not self.data_path.suffix == ".h5ad":
            raise ValueError(f"Data file must be .h5ad format: {self.data_path}")

    @property
    def db_url(self) -> str:
        """SQLite connection URL."""
        return f"sqlite:///{self.db_path}"

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "data_path": str(self.data_path),
            "db_path": str(self.db_path),
            "host": self.host,
            "port": self.port,
            "debug": self.debug,
            "expression_cache_size": self.expression_cache_size,
            "theme": self.theme,
        }
