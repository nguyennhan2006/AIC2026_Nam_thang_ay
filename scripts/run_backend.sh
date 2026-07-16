#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
exec uvicorn online.api.app:app --host 0.0.0.0 --port "${PORT:-8000}" --workers "${WEB_CONCURRENCY:-1}"
