#!/usr/bin/env python3
"""Cross-request prefix reuse: time-to-first-token with a shared prefix.

Starts the pooled server, sends the same long prompt twice (max_tokens=1,
greedy), and compares the server-reported prompt (prefill) time. The second
request adopts every published full block of the prefix and re-evaluates only
the partial tail, so its TTFT should collapse; /metrics confirms the adopted
token count.

Usage:
    python3 bench/prefix_bench.py --model <gguf> [--words 1600]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time
import urllib.request

import _provenance


def post(port, body, timeout=600):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def metric(port, name):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics",
                                timeout=5) as response:
        text = response.read().decode()
    match = re.search(rf"^{re.escape(name)} (\d+)", text, re.M)
    return int(match.group(1)) if match else None


def wait_healthy(port, deadline=120.0):
    start = time.monotonic()
    while time.monotonic() - start < deadline:
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/healthz", timeout=2):
                return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("server did not become healthy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="build/kipp-server-metal")
    ap.add_argument("--backend", default="metal")
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8124)
    ap.add_argument("--words", type=int, default=1600)
    ap.add_argument("--output", default="bench/results/prefix-reuse.json")
    args = ap.parse_args()

    prompt = " ".join(f"item{i}" for i in range(args.words))
    server = subprocess.Popen(
        [args.binary, "--backend", args.backend, "--model", args.model,
         "--port", str(args.port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_healthy(args.port)
        body = {"prompt": prompt, "max_tokens": 1, "temperature": 0}
        cold = post(args.port, body)
        reused_before = metric(args.port, "kipp_prefix_tokens_reused_total")
        warm = post(args.port, body)
        reused_after = metric(args.port, "kipp_prefix_tokens_reused_total")
    finally:
        server.terminate()
        server.wait()

    cold_ms = cold["timings"]["prompt_ms"]
    warm_ms = warm["timings"]["prompt_ms"]
    prompt_tokens = cold["usage"]["prompt_tokens"]
    adopted = (reused_after or 0) - (reused_before or 0)

    root = pathlib.Path(__file__).resolve().parents[1]
    report = {
        "description": "Same long prompt sent twice to the pooled server "
                       "(max_tokens=1, greedy). The second request adopts "
                       "every published full 32-token block and prefills "
                       "only the partial tail.",
        "engine": _provenance.engine_metadata(args.binary, args.backend, root),
        "hardware": _provenance.hardware_metadata(args.backend),
        "model": _provenance.model_metadata(args.model),
        "prompt_tokens": prompt_tokens,
        "adopted_tokens": adopted,
        "cold_prompt_ms": cold_ms,
        "warm_prompt_ms": warm_ms,
        "ttft_speedup": round(cold_ms / warm_ms, 1) if warm_ms else None,
    }
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
