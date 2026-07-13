#!/bin/sh
# Download the one model revision supported by Kipp Phase 1.

set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TOOLS="$ROOT/tools"

exec uv run --project "$TOOLS" --python 3.12 \
    python "$TOOLS/download_model.py" \
    --output "$ROOT/models/qwen3-4b-base/source" "$@"
