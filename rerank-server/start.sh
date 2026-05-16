#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python rerank_server.py