"""
Entry point for running Cellarium as a module.

Usage:
    python -m cellarium serve data.h5ad --port 8050
"""

from cellarium.cli import app

if __name__ == "__main__":
    app()
