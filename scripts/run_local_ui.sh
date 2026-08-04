#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
exec python -m http.server 5173 --directory online/ui
