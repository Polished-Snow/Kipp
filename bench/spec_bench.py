#!/usr/bin/env python3
"""Measure prompt-lookup speculative decoding: acceptance, block efficiency,
and wall-clock speedup vs. non-speculative greedy decode, per workload.

For each workload we run the CLI twice (non-spec baseline and --spec), parse
the KIPP_METRIC decode timer and the KIPP_SPEC counters, and report:
  accept_rate      = accepted / drafted        (fraction of drafts accepted)
  block_efficiency = tokens / target_forwards  (accepted tokens per forward;
                     target_forwards = tokens - accepted, since each step
                     yields one guaranteed token plus the accepted drafts)
  speedup          = baseline_decode_s / spec_decode_s   (same greedy output)

Greedy spec output is token-identical to the baseline by construction, so the
only thing that changes is wall-clock. Usage:
    python3 bench/spec_bench.py --model <gguf> [--decode 64] [--runs 3]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import subprocess
import sys

METRIC = re.compile(r"decode_tokens=(\d+) decode_seconds=([0-9.]+)")
SPEC = re.compile(r"KIPP_SPEC drafted=(\d+) accepted=(\d+)")

WORKLOADS = {
    "repetitive": "List: apple banana cherry apple banana cherry apple banana cherry apple banana",
    "code": "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n\ndef mul(a, b):\n    return a",
    "grounded": "Q: The Eiffel Tower is in Paris. Where is the Eiffel Tower?\nA: The Eiffel Tower is in",
    "chat": "Write a short original sentence about the ocean at dawn:",
}


def run(binary, backend, model, prompt, decode, spec):
    cmd = [binary, "--backend", backend, "--model", model, "--prompt", prompt,
           "--decode", str(decode), "--temperature", "0"]
    if spec:
        cmd.append("--spec")
    out = subprocess.run(cmd, capture_output=True, text=True).stderr
    m = METRIC.search(out)
    s = SPEC.search(out)
    if not m:
        raise RuntimeError(f"no KIPP_METRIC in output:\n{out[-400:]}")
    tokens, secs = int(m.group(1)), float(m.group(2))
    drafted = accepted = 0
    if s:
        drafted, accepted = int(s.group(1)), int(s.group(2))
    return {"decode_tokens": tokens, "decode_seconds": secs,
            "drafted": drafted, "accepted": accepted}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="build/kipp-metal")
    ap.add_argument("--backend", default="metal")
    ap.add_argument("--model", required=True)
    ap.add_argument("--decode", type=int, default=64)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--output", default="bench/results/spec.json")
    args = ap.parse_args()

    results = {}
    for name, prompt in WORKLOADS.items():
        base_s, spec_s, accepts = [], [], []
        for _ in range(args.runs):
            b = run(args.binary, args.backend, args.model, prompt, args.decode, False)
            sp = run(args.binary, args.backend, args.model, prompt, args.decode, True)
            base_s.append(b["decode_seconds"] / b["decode_tokens"])
            spec_s.append(sp["decode_seconds"] / sp["decode_tokens"])
            accepts.append(sp)
        last = accepts[-1]
        forwards = max(1, last["decode_tokens"] - last["accepted"])
        accept_rate = (last["accepted"] / last["drafted"]) if last["drafted"] else 0.0
        base_tps = 1.0 / statistics.median(base_s)
        spec_tps = 1.0 / statistics.median(spec_s)
        results[name] = {
            "baseline_decode_tps": round(base_tps, 2),
            "spec_decode_tps": round(spec_tps, 2),
            "speedup": round(spec_tps / base_tps, 3),
            "accept_rate": round(accept_rate, 3),
            "block_efficiency": round(last["decode_tokens"] / forwards, 3),
            "drafted": last["drafted"], "accepted": last["accepted"],
        }
        print(f"{name:12s} base={base_tps:5.1f}  spec={spec_tps:5.1f}  "
              f"speedup={results[name]['speedup']:.2f}x  "
              f"alpha={accept_rate:.2f}  blk_eff={results[name]['block_efficiency']:.2f}")

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"decode": args.decode, "runs": args.runs, "workloads": results}, f, indent=2)
    print(f"\nwrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
