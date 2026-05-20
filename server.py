#!/usr/bin/env python3
"""FastAPI server entry point for Obsidian RAG."""

import uvicorn

from lib.app import create_app

app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )
