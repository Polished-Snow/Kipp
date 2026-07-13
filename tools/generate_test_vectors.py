#!/usr/bin/env python3
"""Generate immutable Qwen3-4B-Base reference vectors with Transformers."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import struct
import sys
import unicodedata
from typing import Any

import numpy as np
import safetensors
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

REPOSITORY = "Qwen/Qwen3-4B-Base"
REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
DEFAULT_PROMPT = "Hi 世界"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_f32(path: pathlib.Path, tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    values.astype("<f4", copy=False).tofile(path)
    return {
        "path": path.name,
        "shape": list(values.shape),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_tokenizer_cases(
    path: pathlib.Path, tokenizer: Any, prompt: str
) -> dict[str, Any]:
    case_texts = [
        prompt,
        "Hello, world!",
        "cafe\u0301",
        "Qwen3 ١٢٣",
        "\t naïve\n世界",
        "  word",
        "1234",
        "<|endoftext|>",
        "<|fim_prefix|>",
        "<|repo_name|>",
    ]
    with path.open("wb") as file:
        file.write(b"KTOK")
        file.write(struct.pack("<I", len(case_texts)))
        for text in case_texts:
            normalized = unicodedata.normalize("NFC", text)
            input_bytes = text.encode("utf-8")
            normalized_bytes = normalized.encode("utf-8")
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            file.write(struct.pack("<I", len(input_bytes)))
            file.write(input_bytes)
            file.write(struct.pack("<I", len(normalized_bytes)))
            file.write(normalized_bytes)
            file.write(struct.pack("<I", len(token_ids)))
            file.write(struct.pack(f"<{len(token_ids)}I", *token_ids))
    return {
        "path": path.name,
        "case_count": len(case_texts),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=pathlib.Path,
        default=pathlib.Path("models/qwen3-4b-base/source"),
    )
    parser.add_argument(
        "--gguf",
        type=pathlib.Path,
        default=pathlib.Path("models/qwen3-4b-base/kipp-qwen3-4b-base-bf16.gguf"),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("tests/test-vectors/qwen3-4b-base"),
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads(
        (source / "source-manifest.json").read_text(encoding="utf-8")
    )
    if (
        source_manifest.get("repository") != REPOSITORY
        or source_manifest.get("revision") != REVISION
    ):
        raise ValueError("source manifest does not identify Kipp's pinned model")

    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    encoded = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False)
    token_ids = encoded["input_ids"][0].tolist()

    captured: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(_module: object, _inputs: object, value: object) -> None:
            tensor = value[0] if isinstance(value, tuple) else value
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"hook {name} did not receive a tensor")
            captured[name] = tensor[:, -1, :].detach().cpu()

        return hook

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        source,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    hooks = [
        model.model.layers[0].register_forward_hook(capture("layer0")),
        model.model.norm.register_forward_hook(capture("final_norm")),
    ]
    try:
        result = model(**encoded, use_cache=False, return_dict=True)
    finally:
        for hook in hooks:
            hook.remove()

    logits = result.logits[0, -1, :].detach().cpu().to(torch.float32)
    top_values, top_ids = torch.topk(logits, k=10)
    prompt_path = output / "prompt.txt"
    prompt_path.write_text(args.prompt, encoding="utf-8")
    token_path = output / "tokens.u32"
    np.asarray(token_ids, dtype="<u4").tofile(token_path)
    artifacts = {
        "prompt": {
            "path": prompt_path.name,
            "bytes": prompt_path.stat().st_size,
            "sha256": sha256(prompt_path),
        },
        "tokens": {
            "path": token_path.name,
            "shape": [len(token_ids)],
            "bytes": token_path.stat().st_size,
            "sha256": sha256(token_path),
        },
        "tokenizer_cases": write_tokenizer_cases(
            output / "tokenizer-cases.bin", tokenizer, args.prompt
        ),
        "logits": write_f32(output / "logits.f32", logits),
        "layer0": write_f32(output / "layer0-last.f32", captured["layer0"]),
        "final_norm": write_f32(
            output / "final-norm-last.f32", captured["final_norm"]
        ),
    }

    decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    if decoded != args.prompt:
        raise ValueError(
            f"tokenizer round trip changed prompt: {decoded!r} != {args.prompt!r}"
        )

    manifest = {
        "schema": 1,
        "repository": REPOSITORY,
        "revision": REVISION,
        "source_manifest_sha256": sha256(source / "source-manifest.json"),
        "gguf_sha256": sha256(args.gguf.resolve()),
        "prompt_utf8": args.prompt,
        "token_ids": token_ids,
        "decoded_utf8": decoded,
        "argmax": int(torch.argmax(logits).item()),
        "top_10": [
            {"token_id": int(token_id), "logit": float(value)}
            for token_id, value in zip(top_ids.tolist(), top_values.tolist())
        ],
        "artifacts": artifacts,
        "tolerances": {
            "argmax": "exact",
            "full_logits_nmse_max": 1e-5,
        },
        "tooling": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "safetensors": safetensors.__version__,
            "numpy": np.__version__,
            "attention": "eager",
            "weight_compute_dtype": "float32 cast from pinned BF16",
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {manifest_path}")
    print(f"tokens={token_ids}")
    print(f"argmax={manifest['argmax']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"generate_test_vectors.py: {error}", file=sys.stderr)
        raise SystemExit(1) from error
