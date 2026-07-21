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

import _provenance

METRIC = re.compile(r"decode_tokens=(\d+) decode_seconds=([0-9.]+)")
SPEC = re.compile(r"KIPP_SPEC drafted=(\d+) accepted=(\d+)")
GATE = re.compile(r"draft_steps=(\d+) plain_steps=(\d+)")

WORKLOADS = {
    "repetitive": "List: apple banana cherry apple banana cherry apple banana cherry apple banana",
    "code": "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n\ndef mul(a, b):\n    return a",
    "grounded": "Q: The Eiffel Tower is in Paris. Where is the Eiffel Tower?\nA: The Eiffel Tower is in",
    "chat": "Write a short original sentence about the ocean at dawn:",
}


def run(binary, backend, model, prompt, decode, spec, gate="on"):
    cmd = [binary, "--backend", backend, "--model", model, "--prompt", prompt,
           "--decode", str(decode), "--temperature", "0"]
    if spec:
        cmd += ["--spec", "--spec-gate", gate]
    out = subprocess.run(cmd, capture_output=True, text=True).stderr
    m = METRIC.search(out)
    s = SPEC.search(out)
    g = GATE.search(out)
    if not m:
        raise RuntimeError(f"no KIPP_METRIC in output:\n{out[-400:]}")
    tokens, secs = int(m.group(1)), float(m.group(2))
    drafted = accepted = 0
    if s:
        drafted, accepted = int(s.group(1)), int(s.group(2))
    draft_steps = plain_steps = 0
    if g:
        draft_steps, plain_steps = int(g.group(1)), int(g.group(2))
    return {"decode_tokens": tokens, "decode_seconds": secs,
            "drafted": drafted, "accepted": accepted,
            "draft_steps": draft_steps, "plain_steps": plain_steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="build/kipp-metal")
    ap.add_argument("--backend", default="metal")
    ap.add_argument("--model", required=True)
    ap.add_argument("--decode", type=int, default=64)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--gate", choices=("on", "off", "both"), default="both",
                    help="adaptive speculation gate (--spec-gate) for the "
                         "speculative runs; 'off' always drafts; 'both' "
                         "measures ungated and gated adjacently per run "
                         "against one shared baseline (drift-immune A/B) "
                         "and writes both output files")
    ap.add_argument("--output", default="bench/results/spec.json")
    ap.add_argument("--output-gated", default="bench/results/spec-gated.json",
                    help="gated-half output when --gate both")
    args = ap.parse_args()

    modes = ["off", "on"] if args.gate == "both" else [args.gate]
    results = {mode: {} for mode in modes}
    for name, prompt in WORKLOADS.items():
        base_s = []
        spec_s = {mode: [] for mode in modes}
        accepts = {mode: [] for mode in modes}
        # Discarded warm-up so GPU clock ramp does not depress the first
        # measured baseline (matches tools/bench.py's policy).
        run(args.binary, args.backend, args.model, prompt, args.decode, False)
        for _ in range(args.runs):
            b = run(args.binary, args.backend, args.model, prompt, args.decode, False)
            base_s.append(b["decode_seconds"] / b["decode_tokens"])
            # Every mode of a run is measured adjacent to its baseline so a
            # slow drift in machine state cannot skew the A/B.
            for mode in modes:
                sp = run(args.binary, args.backend, args.model, prompt,
                         args.decode, True, mode)
                spec_s[mode].append(sp["decode_seconds"] / sp["decode_tokens"])
                accepts[mode].append(sp)
        base_tps = 1.0 / statistics.median(base_s)
        for mode in modes:
            last = accepts[mode][-1]
            forwards = max(1, last["decode_tokens"] - last["accepted"])
            accept_rate = (last["accepted"] / last["drafted"]) if last["drafted"] else 0.0
            steps = last["draft_steps"] + last["plain_steps"]
            draft_duty = (last["draft_steps"] / steps) if steps else 0.0
            spec_tps = 1.0 / statistics.median(spec_s[mode])
            results[mode][name] = {
                "baseline_decode_tps": round(base_tps, 2),
                "spec_decode_tps": round(spec_tps, 2),
                "speedup": round(spec_tps / base_tps, 3),
                "accept_rate": round(accept_rate, 3),
                "block_efficiency": round(last["decode_tokens"] / forwards, 3),
                "drafted": last["drafted"], "accepted": last["accepted"],
                "draft_steps": last["draft_steps"],
                "plain_steps": last["plain_steps"],
                "draft_duty": round(draft_duty, 3),
            }
            print(f"{name:12s} gate={mode:3s} base={base_tps:5.1f}  "
                  f"spec={spec_tps:5.1f}  "
                  f"speedup={results[mode][name]['speedup']:.2f}x  "
                  f"alpha={accept_rate:.2f}  "
                  f"blk_eff={results[mode][name]['block_efficiency']:.2f}  "
                  f"duty={draft_duty:.2f}")

    root = pathlib.Path(__file__).resolve().parents[1]
    outputs = {"off": args.output, "on": args.output}
    if args.gate == "both":
        outputs = {"off": args.output, "on": args.output_gated}
    for mode in modes:
        report = {
            "engine": _provenance.engine_metadata(args.binary, args.backend,
                                                  root),
            "hardware": _provenance.hardware_metadata(args.backend),
            "model": _provenance.model_metadata(args.model),
            "configuration": {"decode": args.decode, "runs": args.runs,
                              "warmup_runs": 1, "gate": mode,
                              "paired_baseline": args.gate == "both"},
            "workloads": results[mode],
        }
        path = pathlib.Path(outputs[mode])
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
