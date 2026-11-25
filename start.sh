#!/usr/bin/env bash
echo "Starting RobustRAG Enterprise Product Server..."
source .venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
