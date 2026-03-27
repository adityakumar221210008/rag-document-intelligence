#!/bin/bash
set -a
source .env
set +a
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
