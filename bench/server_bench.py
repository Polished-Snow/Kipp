#!/usr/bin/env python3
"""Aggregate server decode throughput vs. batch size.

Starts kipp-server-metal, then measures aggregate generated-token throughput
for (a) one request with n parallel choices and (b) n concurrent connections
with distinct prompts and seeds. Because decode is bandwidth-bound and the
server batches every active choice into one eval per step, aggregate
throughput should grow with n while per-choice cost stays roughly flat.

Usage:
    python3 bench/server_bench.py --model <gguf> [--binary build/kipp-server-metal]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import statistics
import subprocess
import sys
import time
import urllib.request

import _provenance

PROMPTS = [
    "The capital of France is",
    "In the beginning the universe was",
    "A shopping list for a picnic:",
    "The theory of relativity says",
    "My favorite recipe starts with",
    "The history of computing began",
    "On a quiet morning the harbor",
    "Deep in the forest there lived",
]


def post(port, body, timeout=600):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


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


def completion_tokens(response):
    return response["usage"]["completion_tokens"]


def measure_choices(port, n, max_tokens):
    body = {"prompt": PROMPTS[0], "max_tokens": max_tokens,
            "temperature": 0.8, "seed": 7, "n": n}
    start = time.monotonic()
    response = post(port, body)
    elapsed = time.monotonic() - start
    return completion_tokens(response) / elapsed


def measure_concurrent(port, connections, max_tokens):
    def one(index):
        body = {"prompt": PROMPTS[index % len(PROMPTS)],
                "max_tokens": max_tokens, "temperature": 0.8,
                "seed": 11 + index}
        return completion_tokens(post(port, body))

    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(connections) as pool:
        totals = list(pool.map(one, range(connections)))
    elapsed = time.monotonic() - start
    return sum(totals) / elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="build/kipp-server-metal")
    ap.add_argument("--backend", default="metal")
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--output", default="bench/results/server-batch.json")
    args = ap.parse_args()

    server = subprocess.Popen(
        [args.binary, "--backend", args.backend, "--model", args.model,
         "--port", str(args.port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_healthy(args.port)
        # Discarded warm-up so GPU clocks are steady before measurement.
        measure_choices(args.port, 1, args.max_tokens)
        results = {"choices": {}, "concurrent": {}}
        for n in (1, 2, 4, 8):
            samples = [measure_choices(args.port, n, args.max_tokens)
                       for _ in range(args.runs)]
            results["choices"][str(n)] = round(statistics.median(samples), 2)
            print(f"choices n={n}: {results['choices'][str(n)]} tok/s")
        for c in (1, 2, 4, 8):
            samples = [measure_concurrent(args.port, c, args.max_tokens)
                       for _ in range(args.runs)]
            results["concurrent"][str(c)] = round(statistics.median(samples), 2)
            print(f"concurrent c={c}: {results['concurrent'][str(c)]} tok/s")
    finally:
        server.terminate()
        server.wait()

    root = pathlib.Path(__file__).resolve().parents[1]
    report = {
        "description": "Aggregate generated-token throughput vs batch size: "
                       "one request with n choices, and n concurrent "
                       "connections. Median of runs; sampled decoding "
                       "(temperature 0.8, per-config seeds).",
        "engine": _provenance.engine_metadata(args.binary, args.backend, root),
        "hardware": _provenance.hardware_metadata(args.backend),
        "model": _provenance.model_metadata(args.model),
        "max_tokens": args.max_tokens,
        "runs": args.runs,
        "aggregate_tokens_per_second": results,
    }
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
