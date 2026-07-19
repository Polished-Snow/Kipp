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

import checkpoints

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
        # Instruct checkpoints tokenize these as single added tokens; base
        # checkpoints BPE-split them. Both behaviors are captured here.
        "<|im_start|>user\nhi<|im_end|>",
        "<think>reason</think>",
        "<tool_response>ok</tool_response>",
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
    parser.add_argument("--checkpoint", default="qwen3-4b-base")
    parser.add_argument("--source", type=pathlib.Path, default=None)
    parser.add_argument("--gguf", type=pathlib.Path, default=None)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    # The FP32-on-CPU reference is the default and produced every pinned
    # vector. Checkpoints too large for host RAM in FP32 (e.g. 32B needs
    # ~128 GiB) use a BF16-weight reference on the GPU; Kipp's runtime also
    # uses BF16 weights, so this stays a faithful oracle and the 5e-5 gate
    # absorbs the small numeric difference.
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"],
                        default="float32")
    args = parser.parse_args()
    spec = checkpoints.get(args.checkpoint)
    source = args.source or pathlib.Path("models") / spec.id / "source"
    gguf = (
        args.gguf
        or pathlib.Path("models") / spec.id / f"kipp-{spec.id}-bf16.gguf"
    )
    output = args.output or pathlib.Path("tests/test-vectors") / spec.id

    source = source.resolve()
    args.gguf = gguf.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads(
        (source / "source-manifest.json").read_text(encoding="utf-8")
    )
    if (
        source_manifest.get("repository") != spec.repository
        or source_manifest.get("revision") != spec.revision
    ):
        raise ValueError(
            f"source manifest does not match registry entry '{spec.id}'"
        )

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
    compute_dtype = (
        torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        source,
        local_files_only=True,
        dtype=compute_dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    if args.device == "cuda":
        # Move the (BF16) model to the GPU after a low-RAM CPU load, so no
        # `accelerate`/device_map dependency is needed. BF16 32B is ~64 GiB,
        # which fits both host RAM and an 80 GiB GPU.
        model = model.to("cuda")
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
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
    # An FP32 reference is tight (5e-5, dominated by Kipp's BF16 KV
    # contract); a BF16 GPU reference differs from Kipp's BF16-weight path by
    # ~1e-3, so its full-logit bound is looser. The native gate reads this
    # value from nmse-max.txt so the artifact, not the code, sets it.
    nmse_max = 2.0e-3 if args.dtype == "bfloat16" else 5.0e-5
    (output / "nmse-max.txt").write_text(f"{nmse_max:.9g}\n", encoding="utf-8")
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
        "checkpoint": spec.id,
        "repository": spec.repository,
        "revision": spec.revision,
        "source_manifest_sha256": sha256(source / "source-manifest.json"),
        "gguf_sha256": sha256(args.gguf),
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
            # Bounds the BF16 KV storage contract, which the FP32 reference
            # does not apply; measured NMSE is ~4e-7 (4B) to ~2e-5 (0.6B),
            # and drops below 1e-10 when KV rounding is disabled. A BF16 GPU
            # reference (used for checkpoints too large for an FP32 CPU
            # reference) differs from Kipp's BF16-weight/FP32-accumulate path
            # by ~1e-3, so it gets a correspondingly looser full-logit bound;
            # argmax stays exact regardless.
            "full_logits_nmse_max": nmse_max,
        },
        "tooling": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "safetensors": safetensors.__version__,
            "numpy": np.__version__,
            "attention": "eager",
            "reference_device": args.device,
            "weight_compute_dtype": (
                "bfloat16 weights, fp32 logits (GPU)"
                if args.dtype == "bfloat16"
                else "float32 cast from pinned BF16"
            ),
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
