"""Application logging configuration."""

import logging


def configure_logging() -> logging.Logger:
    """Configures basic logging and returns the application logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("obsidian-rag")
