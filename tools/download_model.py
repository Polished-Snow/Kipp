#!/usr/bin/env python3
"""Download and hash one pinned checkpoint from the supported registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any

from huggingface_hub import snapshot_download

import checkpoints

ALLOW_PATTERNS = (
    "LICENSE",
    "README.md",
    "config.json",
    "generation_config.json",
    "*.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
REQUIRED_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(spec: checkpoints.CheckpointSpec,
                   directory: pathlib.Path) -> dict[str, Any]:
    names = sorted(
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.name != "source-manifest.json"
        and not path.name.endswith(".tmp")
    )
    for required in REQUIRED_FILES:
        if required not in names:
            raise FileNotFoundError(f"snapshot is missing {required}")
    if not any(name.endswith(".safetensors") for name in names):
        raise FileNotFoundError("snapshot holds no safetensors shards")
    files = {
        name: {
            "bytes": (directory / name).stat().st_size,
            "sha256": sha256(directory / name),
        }
        for name in names
    }
    return {
        "schema": 1,
        "repository": spec.repository,
        "revision": spec.revision,
        "files": files,
    }


def manifest_matches(spec: checkpoints.CheckpointSpec,
                     directory: pathlib.Path,
                     manifest_path: pathlib.Path) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        expected = json.loads(manifest_path.read_text(encoding="utf-8"))
        return expected == build_manifest(spec, directory)
    except (OSError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="qwen3-4b-base")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args()
    spec = checkpoints.get(args.checkpoint)
    output = args.output or pathlib.Path("models") / spec.id / "source"
    output = output.resolve()
    manifest_path = output / "source-manifest.json"

    if output.is_dir() and manifest_matches(spec, output, manifest_path):
        print(f"Verified existing snapshot: {output}")
        return

    output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=spec.repository,
        revision=spec.revision,
        local_dir=output,
        allow_patterns=list(ALLOW_PATTERNS),
    )
    manifest = build_manifest(spec, output)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(manifest_path)
    print(f"Downloaded and verified {spec.repository}@{spec.revision}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"download_model.py: {error}", file=sys.stderr)
        raise SystemExit(1) from error
