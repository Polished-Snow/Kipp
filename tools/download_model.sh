#!/bin/sh
# Download one pinned checkpoint from Kipp's supported registry.
# Usage: tools/download_model.sh [--checkpoint ID] [--output DIR]

set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TOOLS="$ROOT/tools"

cd "$ROOT"
exec uv run --project "$TOOLS" --python 3.12 \
    python "$TOOLS/download_model.py" "$@"
