#!/usr/bin/env python3
"""Download and hash Kipp's one pinned upstream model snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any

from huggingface_hub import snapshot_download

REPOSITORY = "Qwen/Qwen3-4B-Base"
REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
FILES = (
    "LICENSE",
    "README.md",
    "config.json",
    "generation_config.json",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(directory: pathlib.Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for name in FILES:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"snapshot is missing {name}")
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return {
        "schema": 1,
        "repository": REPOSITORY,
        "revision": REVISION,
        "files": files,
    }


def manifest_matches(directory: pathlib.Path, manifest_path: pathlib.Path) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        expected = json.loads(manifest_path.read_text(encoding="utf-8"))
        return expected == build_manifest(directory)
    except (OSError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("models/qwen3-4b-base/source"),
    )
    args = parser.parse_args()
    output = args.output.resolve()
    manifest_path = output / "source-manifest.json"

    if manifest_matches(output, manifest_path):
        print(f"Verified existing snapshot: {output}")
        return

    output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPOSITORY,
        revision=REVISION,
        local_dir=output,
        allow_patterns=list(FILES),
    )
    manifest = build_manifest(output)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(manifest_path)
    print(f"Downloaded and verified {REPOSITORY}@{REVISION}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"download_model.py: {error}", file=sys.stderr)
        raise SystemExit(1) from error
