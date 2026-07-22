#!/usr/bin/env python3
"""Parse a verda_cuda_gate.sh log into bench/results/cuda-a100-gates.json.

The gate binaries already print machine-parseable lines (MODEL nmse=...,
PHASE4 ... nmse=..., PASS/FAIL ...); verda_cuda_gate.sh adds
KIPP_CUDA_GATE_* provenance markers. This collector turns one tee'd log
into a committed, paper-bindable result file.

Usage: python3 tools/ops/collect_cuda_gates.py cuda-gate-YYYYMMDD.log
Stdlib only, matching bench/*.py conventions.
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

GPU_RE = re.compile(r"^KIPP_CUDA_GATE_GPU (.+?),\s*(.+)$", re.M)
COMMIT_RE = re.compile(r"^KIPP_CUDA_GATE_COMMIT ([0-9a-f]{40})$", re.M)
INSTANCE_RE = re.compile(r"^KIPP_CUDA_GATE_INSTANCE (\S+) (\S+)$", re.M)
CHECKPOINT_RE = re.compile(r"^KIPP_CUDA_GATE_CHECKPOINT (\S+)$", re.M)
MODEL_RE = re.compile(
    r"MODEL nmse=([0-9.eE+-]+) nmse_max=([0-9.eE+-]+) "
    r"expected_argmax=(\d+) actual_argmax=(\d+)")
PHASE4_RE = re.compile(r"PHASE4 (\S+) nmse=([0-9.eE+-]+) "
                       r"cpu_argmax=(\d+) cuda_argmax=(\d+)")
OPERATORS_RE = re.compile(r"PASS cuda_operators device=(.+)$", re.M)


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: collect_cuda_gates.py <gate-log>")
    log = pathlib.Path(sys.argv[1]).read_text(errors="replace")

    gpu = GPU_RE.search(log)
    commit = COMMIT_RE.search(log)
    instance = INSTANCE_RE.search(log)

    checkpoints = {}
    segments = CHECKPOINT_RE.split(log)
    # split() yields [pre, id1, seg1, id2, seg2, ...]
    for index in range(1, len(segments) - 1, 2):
        checkpoint = segments[index]
        segment = segments[index + 1]
        entry = {}
        model = MODEL_RE.search(segment)
        if model:
            entry["model"] = {
                "verdict": ("PASS" if "PASS model_integration" in segment
                            else "FAIL"),
                "nmse": float(model.group(1)),
                "nmse_max": float(model.group(2)),
                "argmax_match": model.group(3) == model.group(4),
            }
        phase4_nmse = [float(m.group(2)) for m in PHASE4_RE.finditer(segment)]
        if phase4_nmse or "phase4_cuda" in segment:
            entry["phase4-cuda"] = {
                "verdict": ("PASS" if "PASS phase4_cuda" in segment
                            else "FAIL"),
                "nmse_max_observed": max(phase4_nmse) if phase4_nmse else None,
                "positions_checked": len(phase4_nmse),
            }
        if entry:
            checkpoints[checkpoint] = entry

    operators = OPERATORS_RE.search(log)
    failures = [line for line in log.splitlines()
                if line.startswith("FAIL ")]

    doc = {
        "description": ("CUDA correctness gates on a disposable Verda A100 "
                        "via tools/ops/verda_cuda_gate.sh: --model (CPU "
                        "logits vs pinned vectors, built and run on the "
                        "instance) and --phase4-cuda (CUDA vs CPU oracle, "
                        "arg max exact + NMSE <= 1e-4) per checkpoint, plus "
                        "the CUDA operator suite."),
        "engine_commit": commit.group(1) if commit else "unknown",
        "hardware": {
            "provider": "Verda",
            "instance_type": instance.group(1) if instance else "1A100.22V",
            "image": instance.group(2) if instance else None,
            "gpu": gpu.group(1).strip() if gpu else None,
            "driver": gpu.group(2).strip() if gpu else None,
        },
        "checkpoints": checkpoints,
        "cuda_operators": {
            "verdict": "PASS" if operators else "FAIL",
            "device": operators.group(1).strip() if operators else None,
        },
        "failures": failures,
    }
    out = ROOT / "bench/results/cuda-a100-gates.json"
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {out}")
    print(f"checkpoints: {list(checkpoints)}")
    if failures:
        print("FAILURES PRESENT:", failures)
        sys.exit(1)


if __name__ == "__main__":
    main()
