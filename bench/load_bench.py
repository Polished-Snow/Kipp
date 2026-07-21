#!/usr/bin/env python3
"""Serving-load benchmark: Poisson arrivals against an OpenAI endpoint.

Drives streaming /v1/completions requests at a given arrival rate and
reports the standard serving metrics: time-to-first-token (TTFT),
time-per-output-token (TPOT), end-to-end latency (mean/p50/p90/p99), output
throughput, and goodput under an SLO. The methodology follows the
vLLM/Sarathi-Serve convention: open-loop Poisson arrivals, streamed
first-token timing, percentile reporting.

Usage:
    python3 bench/load_bench.py --model <gguf> \
        [--rates 0.5,1,2,4] [--requests 32] [--slo-ttft-ms 2000] \
        [--slo-tpot-ms 200]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import random
import statistics
import subprocess
import sys
import time
import urllib.request

import _provenance

PROMPT_WORDS = ["the", "of", "science", "history", "market", "river",
                "engine", "island", "theory", "story", "garden", "signal"]


def make_prompt(rng, words):
    return " ".join(rng.choice(PROMPT_WORDS) for _ in range(words))


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


def stream_request(port, prompt, max_tokens, seed):
    """Send one streaming completion; return per-request metrics."""
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0.8, "seed": seed,
                       "stream": True}).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    start = time.monotonic()
    first = None
    chunks = 0
    with urllib.request.urlopen(request, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            event = json.loads(payload)
            if not event.get("choices"):
                continue
            if event["choices"][0].get("text"):
                if first is None:
                    first = time.monotonic()
                chunks += 1
    end = time.monotonic()
    ttft = (first or end) - start
    decode = end - (first or end)
    tpot = decode / max(1, chunks - 1)
    return {"ttft_s": ttft, "tpot_s": tpot, "e2e_s": end - start,
            "output_chunks": chunks}


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(fraction * len(ordered)))
    return ordered[index]


def summarize(samples, key):
    values = [s[key] for s in samples]
    return {"mean": round(statistics.mean(values), 4),
            "p50": round(percentile(values, 0.50), 4),
            "p90": round(percentile(values, 0.90), 4),
            "p99": round(percentile(values, 0.99), 4)}


def run_rate(port, rate, count, max_tokens, prompt_words, slo_ttft,
             slo_tpot, rng):
    """Open-loop: schedule Poisson arrivals, fire each on its own thread."""
    arrivals = []
    clock = 0.0
    for _ in range(count):
        clock += rng.expovariate(rate)
        arrivals.append(clock)
    results = []
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        futures = []
        for index, offset in enumerate(arrivals):
            delay = start + offset - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            prompt = make_prompt(rng, prompt_words)
            futures.append(pool.submit(stream_request, port, prompt,
                                       max_tokens, 100 + index))
        for future in futures:
            results.append(future.result())
    wall = time.monotonic() - start
    total_tokens = sum(s["output_chunks"] for s in results)
    good = sum(1 for s in results
               if s["ttft_s"] * 1000 <= slo_ttft
               and s["tpot_s"] * 1000 <= slo_tpot)
    return {
        "request_rate_per_s": rate,
        "requests": count,
        "output_tokens_per_s": round(total_tokens / wall, 2),
        "ttft_s": summarize(results, "ttft_s"),
        "tpot_s": summarize(results, "tpot_s"),
        "e2e_s": summarize(results, "e2e_s"),
        "goodput_fraction": round(good / count, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="build/kipp-server-metal")
    ap.add_argument("--backend", default="metal")
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8125)
    ap.add_argument("--rates", default="0.5,1,2,4")
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--prompt-words", type=int, default=180)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--slo-ttft-ms", type=float, default=2000.0)
    ap.add_argument("--slo-tpot-ms", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--output", default="bench/results/load.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    server = subprocess.Popen(
        [args.binary, "--backend", args.backend, "--model", args.model,
         "--port", str(args.port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sweeps = []
    try:
        wait_healthy(args.port)
        # Warm-up request so GPU clocks and the page cache are steady.
        stream_request(args.port, make_prompt(rng, args.prompt_words),
                       args.max_tokens, 1)
        for rate in (float(r) for r in args.rates.split(",")):
            sweep = run_rate(args.port, rate, args.requests,
                             args.max_tokens, args.prompt_words,
                             args.slo_ttft_ms, args.slo_tpot_ms, rng)
            sweeps.append(sweep)
            print(f"rate={rate}/s tok/s={sweep['output_tokens_per_s']} "
                  f"ttft p99={sweep['ttft_s']['p99']}s "
                  f"tpot p99={sweep['tpot_s']['p99']}s "
                  f"goodput={sweep['goodput_fraction']}")
    finally:
        server.terminate()
        server.wait()

    root = pathlib.Path(__file__).resolve().parents[1]
    report = {
        "description": "Open-loop Poisson serving load against the OpenAI "
                       "endpoint; streamed TTFT/TPOT with percentiles and "
                       "goodput under the stated SLO.",
        "engine": _provenance.engine_metadata(args.binary, args.backend, root),
        "hardware": _provenance.hardware_metadata(args.backend),
        "model": _provenance.model_metadata(args.model),
        "configuration": {"requests_per_rate": args.requests,
                          "prompt_words": args.prompt_words,
                          "max_tokens": args.max_tokens,
                          "slo_ttft_ms": args.slo_ttft_ms,
                          "slo_tpot_ms": args.slo_tpot_ms,
                          "seed": args.seed},
        "sweeps": sweeps,
    }
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
