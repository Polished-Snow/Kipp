#!/usr/bin/env python3
"""Quantization quality: wikitext-2 perplexity per weight scheme.

Runs the CLI's --ppl mode once per weight scheme (perplexity is
deterministic, so no repeats), then verifies the Metal backend against the
CPU oracle on a short token slice. Protocol: non-overlapping windows, a
fresh session per window, every position scored except each window's first
token, no burn-in — deliberately simple to state, and therefore NOT
numerically comparable to llama.cpp's perplexity tool (which skips a
per-chunk burn-in region).

Usage:
    python3 bench/ppl_bench.py [--window 2048] [--limit 0] \
        [--output bench/results/ppl-4b.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import subprocess
import sys

import _provenance

PPL = re.compile(r"KIPP_PPL\s+(.*)")
VERIFY_TOLERANCE = 1e-3

DEFAULT_MODELS = [
    "models/qwen3-4b-base/kipp-qwen3-4b-base-bf16.gguf",
    "models/qwen3-4b-base/kipp-qwen3-4b-base-q8_0.gguf",
    "models/qwen3-4b-base/kipp-qwen3-4b-base-affine4_gs32.gguf",
]


def resolve(root, path):
    path = pathlib.Path(path)
    return path if path.is_absolute() else (root / path)


def run_ppl(binary, backend, model, tokens, window, limit):
    cmd = [str(binary), "--backend", backend, "--model", str(model),
           "--ppl", str(tokens), "--ppl-window", str(window)]
    if limit:
        cmd += ["--ppl-limit", str(limit)]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} exited {completed.returncode}\n"
                           f"{completed.stderr[-800:]}")
    match = PPL.search(completed.stderr)
    if not match:
        raise RuntimeError(f"no KIPP_PPL line in output:\n"
                           f"{completed.stderr[-800:]}")
    fields = {}
    for pair in match.group(1).split():
        key, sep, value = pair.partition("=")
        if sep:
            fields[key] = value
    missing = [key for key in
               ("backend", "tokens", "scored", "window", "nll", "ppl",
                "seconds") if key not in fields]
    if missing:
        raise RuntimeError(f"KIPP_PPL line missing {missing}: {match.group(0)}")
    if fields["backend"] != backend:
        raise RuntimeError(f"CLI reported backend {fields['backend']}, "
                           f"expected {backend}")
    return {"tokens": int(fields["tokens"]),
            "tokens_scored": int(fields["scored"]),
            "window": int(fields["window"]),
            "nll": float(fields["nll"]),
            "ppl": float(fields["ppl"]),
            "seconds": float(fields["seconds"])}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="build/kipp-metal")
    ap.add_argument("--cpu-binary", default="build/kipp")
    ap.add_argument("--backend", default="metal")
    ap.add_argument("--tokens", default="bench/data/wikitext2-test.tokens")
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap on evaluated tokens; 0 = the whole file")
    ap.add_argument("--verify-limit", type=int, default=1024,
                    help="token slice for the CPU-vs-Metal agreement check")
    ap.add_argument("--verify-model",
                    default="models/qwen3-0.6b-base/"
                            "kipp-qwen3-0.6b-base-bf16.gguf",
                    help="model for the agreement check; defaults to the "
                    "0.6B because the scalar CPU oracle is far too slow to "
                    "verify a 4B slice in reasonable time, and the check "
                    "exercises the same PPL loop and kernels either way")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--output", default="bench/results/ppl-4b.json")
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    binary = resolve(root, args.binary)
    cpu_binary = resolve(root, args.cpu_binary)
    tokens = resolve(root, args.tokens)
    models = [resolve(root, model) for model in args.models]
    for required in (binary, cpu_binary, tokens, *models):
        if not required.is_file():
            ap.error(f"missing file: {required}")
    token_bytes = tokens.stat().st_size
    if token_bytes % 4 != 0:
        ap.error(f"{tokens} is not a whole number of uint32 tokens")

    verify_model = resolve(root, args.verify_model)
    if not verify_model.is_file():
        ap.error(f"missing file: {verify_model}")

    results = {}
    model_blocks = {}
    for model in models:
        meta = _provenance.model_metadata(model)
        scheme = meta.get("scheme") or model.stem
        print(f"measuring {scheme} ...", flush=True)
        outcome = run_ppl(binary, args.backend, model, tokens, args.window,
                          args.limit)
        outcome["windows"] = math.ceil(outcome["tokens"] / outcome["window"])
        results[scheme] = outcome
        model_blocks[scheme] = meta
        print(f"{scheme:14s} ppl={outcome['ppl']:.4f} "
              f"scored={outcome['tokens_scored']} "
              f"({outcome['seconds']:.1f}s)", flush=True)

    print(f"verifying cpu vs {args.backend} on {args.verify_limit} tokens "
          f"of {verify_model.name} ...", flush=True)
    cpu = run_ppl(cpu_binary, "cpu", verify_model, tokens, args.window,
                  args.verify_limit)
    accel = run_ppl(binary, args.backend, verify_model, tokens, args.window,
                    args.verify_limit)
    rel_diff = abs(cpu["ppl"] - accel["ppl"]) / cpu["ppl"]
    verify = {"model": verify_model.name,
              "tokens": args.verify_limit,
              "cpu_ppl": cpu["ppl"],
              "metal_ppl": accel["ppl"],
              "rel_diff": rel_diff,
              "tolerance": VERIFY_TOLERANCE,
              "pass": rel_diff < VERIFY_TOLERANCE}

    report = {
        "description": "Wikitext-2 test perplexity per weight scheme. "
                       "Non-overlapping windows, fresh session per window, "
                       "every position scored except each window's first "
                       "token, no burn-in — not comparable to llama.cpp's "
                       "perplexity tool.",
        "engine": _provenance.engine_metadata(binary, args.backend, root),
        "hardware": _provenance.hardware_metadata(args.backend),
        "data": {"path": str(tokens.relative_to(root)
                             if tokens.is_relative_to(root) else tokens),
                 "sha256": sha256_file(tokens),
                 "token_count": token_bytes // 4},
        "configuration": {"window": args.window, "limit": args.limit,
                          "chunk": 32,
                          "scoring": "all positions except each window's "
                                     "first token; non-overlapping windows; "
                                     "no burn-in"},
        "models": model_blocks,
        "results": results,
        "verify": verify,
    }
    output = resolve(root, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}", file=sys.stderr)
    if not verify["pass"]:
        print(f"VERIFY FAILED: cpu ppl {cpu['ppl']:.6f} vs "
              f"{args.backend} ppl {accel['ppl']:.6f} "
              f"(rel diff {rel_diff:.2e} >= {VERIFY_TOLERANCE:.0e})",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
