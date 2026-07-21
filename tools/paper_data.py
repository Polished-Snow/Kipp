#!/usr/bin/env python3
"""Generate the paper's measured-number artifacts from bench/results/*.json.

Single source of truth for every number the paper quotes: emits
paper/generated/results-macros.tex (one LaTeX macro per measured value,
each with a provenance comment naming its JSON and field) and
paper/data/*.dat (pgfplots-readable tables). Re-running a benchmark and
re-running this script is the only way a number in the paper changes.

Modes:
    python3 tools/paper_data.py            # generate (skips absent inputs)
    python3 tools/paper_data.py --check    # CI: inputs committed+unmodified,
                                           # outputs byte-identical, spec
                                           # baselines coherent; writes nothing
    --allow-missing                        # downgrade absent inputs to
                                           # warnings in --check (interim use
                                           # between code and results commits)

Stdlib only, by design (mirrors bench/*.py).
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"
GENERATED = ROOT / "paper" / "generated"
DATA = ROOT / "paper" / "data"

MACROS_TEX = GENERATED / "results-macros.tex"

# Every results file the paper may draw from. --check requires all of them
# committed and unmodified; generation skips absent ones with a warning.
EXPECTED_INPUTS = [
    "4b-bf16-decode.json", "4b-bf16-prefill.json",
    "4b-q8_0-decode.json", "4b-q8_0-prefill.json",
    "4b-affine4_gs32-decode.json", "4b-affine4_gs32-prefill.json",
    "4b-bf16-prefill2k.json", "4b-q8_0-prefill2k.json",
    "4b-affine4_gs32-prefill2k.json",
    "ctx-3.json", "ctx-200.json", "ctx-800.json", "ctx-1600.json",
    "ctx-3200.json", "ctx-6400.json", "ctx-12800.json",
    "06b-bf16-decode.json", "06b-bf16-prefill.json",
    "8b-bf16-decode.json", "8b-bf16-prefill.json",
    "8b-q8_0-decode.json", "8b-q8_0-prefill.json",
    "spec.json", "spec-gated.json",
    "server-batch.json", "prefix-reuse.json", "load.json",
    "ppl-4b.json", "llamacpp-qwen3-4b.json",
    "gate-costs.json", "faults.json",
]

CTX_POINTS = [3, 200, 800, 1600, 3200, 6400, 12800]
SCHEMES = [("bf16", "bf16"), ("q8_0", "q8_0"), ("affine4_gs32", "affine4")]
SPEC_WORKLOADS = ["repetitive", "code", "grounded", "chat"]
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                6: "six", 7: "seven", 8: "eight"}
DIGIT_NAMES = {"1": "One", "2": "Two", "4": "Four", "8": "Eight"}
GATE_MACRO_NAMES = {
    "model": "Model", "multilogit": "Multilogit",
    "phase2-model": "PhaseTwoModel", "paged-cpu": "PagedCpu",
    "pooled-cpu": "PooledCpu", "metal-operators": "MetalOperators",
    "phase3-metal": "PhaseThreeMetal", "paged-metal": "PagedMetal",
    "multilogit-metal": "MultilogitMetal", "pooled-metal": "PooledMetal",
    "fault-reference": "FaultReference", "sanitize": "Sanitize",
    "unit": "Unit", "test": "Unit",
}


def fmt_tps(value):
    """Tokens/second: one decimal below 100, whole numbers above."""
    return f"{value:.1f}" if value < 100 else f"{value:.0f}"


def fmt_ratio(value):
    return f"{value:.2f}"


def fmt_ppl(value):
    return f"{value:.3f}"


def fmt_pct(value):
    return f"{value:.2g}" if abs(value) < 1 else f"{value:.1f}"


def fmt_nmse(value):
    """LaTeX scientific notation (math mode supplied by the caller)."""
    if value == 0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / 10 ** exponent
    return f"{mantissa:.2g}\\times 10^{{{exponent}}}"


def fmt_int_comma(value):
    return f"{value:,.0f}"


def load_results():
    """Read every expected input that exists; report the rest."""
    loaded = {}
    missing = []
    for name in EXPECTED_INPUTS:
        path = RESULTS / name
        if path.is_file():
            with open(path) as handle:
                loaded[name] = json.load(handle)
        else:
            missing.append(name)
    return loaded, missing


def result_commit(data):
    commit = ""
    if isinstance(data.get("engine"), dict):
        commit = data["engine"].get("commit", "")
    commit = commit or data.get("engine_commit", "")
    return commit[:7] if commit else "uncommitted"


def config_field(data, key, default=None):
    """Read a run parameter from either schema generation: the current
    bench scripts nest decode/runs/gate under `configuration`; older
    committed files carry them at the top level."""
    configuration = data.get("configuration")
    if isinstance(configuration, dict) and key in configuration:
        return configuration[key]
    return data.get(key, default)


def model_scheme(data):
    """Weight scheme of a result's model, across schema generations:
    new files carry a manifest dict with an explicit `scheme`; older
    files record only the GGUF path, so fall back to parsing it."""
    model = data.get("model")
    if isinstance(model, dict):
        if model.get("scheme"):
            return model["scheme"]
        model = model.get("path", "")
    if isinstance(model, str):
        for scheme in ("affine4_gs32", "q8_0", "bf16"):
            if scheme in model:
                return scheme
    return None


class MacroWriter:
    def __init__(self):
        self.lines = [
            "% Generated by tools/paper_data.py -- DO NOT EDIT.",
            "% Every macro cites the committed bench/results/*.json it was",
            "% read from. Regenerate with `make paper-data`; CI verifies",
            "% byte-identical regeneration with `make paper-check`.",
        ]
        self.count = 0

    def section(self, title):
        self.lines.append(f"% -- {title} --")

    def macro(self, name, value, source, field, commit, extra=""):
        suffix = f" {extra}" if extra else ""
        self.lines.append(
            f"\\newcommand{{\\{name}}}{{{value}}}"
            f"% {source} {field} commit {commit}{suffix}")
        self.count += 1

    def text(self):
        return "\n".join(self.lines) + "\n"


def dat_text(source_names, loaded, header, rows):
    lines = ["% Generated by tools/paper_data.py -- DO NOT EDIT."]
    for name in source_names:
        if name in loaded:
            lines.append(f"% source {name} commit {result_commit(loaded[name])}")
    lines.append(header)
    lines.extend(rows)
    return "\n".join(lines) + "\n"


def generate(loaded):
    """Build {relative output path: text} from the loaded results."""
    outputs = {}
    macros = MacroWriter()

    def metric(name, key):
        block = loaded[name][key]
        return block["median"], block["median_absolute_deviation"]

    # Decode / prefill by weight scheme (4B).
    macros.section("decode / prefill by weight scheme, Qwen3-4B")
    scheme_macro = {"bf16": "BF", "q8_0": "QEight", "affine4_gs32": "AffineFour"}
    prefill_macro = {"bf16": "PrefillBF", "q8_0": "PrefillQEight",
                     "affine4_gs32": "PrefillAffine"}
    quant_rows = []
    for scheme, label in SCHEMES:
        decode_name = f"4b-{scheme}-decode.json"
        prefill_name = f"4b-{scheme}-prefill.json"
        if decode_name not in loaded or prefill_name not in loaded:
            continue
        decode, decode_mad = metric(decode_name, "decode_tokens_per_second")
        prefill, prefill_mad = metric(prefill_name, "prefill_tokens_per_second")
        commit = result_commit(loaded[decode_name])
        macros.macro(f"Decode{scheme_macro[scheme]}", fmt_tps(decode),
                     decode_name, "decode_tokens_per_second.median", commit,
                     f"MAD {decode_mad:.2g}")
        macros.macro(prefill_macro[scheme], fmt_tps(prefill),
                     prefill_name, "prefill_tokens_per_second.median",
                     result_commit(loaded[prefill_name]),
                     f"MAD {prefill_mad:.2g}")
        quant_rows.append(f"{label} {decode:.2f} {prefill:.2f} "
                          f"{decode_mad:.3f} {prefill_mad:.3f}")
    # Matched 2048-token prefill points for the llama.cpp head-to-head.
    prefill2k_macro = {"bf16": "PrefillTwoKBF", "q8_0": "PrefillTwoKQEight",
                       "affine4_gs32": "PrefillTwoKAffine"}
    for scheme, _ in SCHEMES:
        name = f"4b-{scheme}-prefill2k.json"
        if name not in loaded:
            continue
        prefill, prefill_mad = metric(name, "prefill_tokens_per_second")
        macros.macro(prefill2k_macro[scheme], fmt_tps(prefill), name,
                     "prefill_tokens_per_second.median",
                     result_commit(loaded[name]), f"MAD {prefill_mad:.2g}")
    if "4b-bf16-decode.json" in loaded:
        rss = loaded["4b-bf16-decode.json"]["peak_resident_bytes"]["median"]
        macros.macro("PeakRSS", f"{rss / 1e6:.0f}", "4b-bf16-decode.json",
                     "peak_resident_bytes.median (MB, scheme-invariant)",
                     result_commit(loaded["4b-bf16-decode.json"]))
    if quant_rows:
        outputs["data/quant-throughput.dat"] = dat_text(
            [f"4b-{s}-{k}.json" for s, _ in SCHEMES
             for k in ("decode", "prefill")],
            loaded, "scheme decode prefill decode_mad prefill_mad", quant_rows)

    # Context scaling.
    ctx_rows = []
    ctx_names = []
    for tokens in CTX_POINTS:
        name = f"ctx-{tokens}.json"
        if name not in loaded:
            continue
        ctx_names.append(name)
        decode, decode_mad = metric(name, "decode_tokens_per_second")
        prefill, prefill_mad = metric(name, "prefill_tokens_per_second")
        ctx_rows.append(f"{tokens} {decode:.2f} {decode_mad:.3f} "
                        f"{prefill:.2f} {prefill_mad:.3f}")
    if ctx_rows:
        outputs["data/ctx-scaling.dat"] = dat_text(
            ctx_names, loaded,
            "tokens decode decode_mad prefill prefill_mad", ctx_rows)
        macros.section("context scaling (q8_0, decode 32)")
        first = ctx_names[0]
        last = ctx_names[-1]
        macros.macro("CtxDecodeShort",
                     fmt_tps(loaded[first]["decode_tokens_per_second"]["median"]),
                     first, "decode_tokens_per_second.median",
                     result_commit(loaded[first]))
        macros.macro("CtxDecodeLongest",
                     fmt_tps(loaded[last]["decode_tokens_per_second"]["median"]),
                     last, "decode_tokens_per_second.median",
                     result_commit(loaded[last]))
        peak_name = max(
            ctx_names,
            key=lambda n: loaded[n]["prefill_tokens_per_second"]["median"])
        macros.macro("CtxPrefillPeak",
                     fmt_tps(loaded[peak_name]["prefill_tokens_per_second"]["median"]),
                     peak_name, "prefill_tokens_per_second.median",
                     result_commit(loaded[peak_name]))

    # Model-size sweep.
    size_rows = []
    size_names = []
    size_macros = [
        ("06b-bf16", "ZeroSixBBF", "0.6b", 0.6, "bf16"),
        ("8b-bf16", "EightBBF", "8b", 8.0, "bf16"),
        ("8b-q8_0", "EightBQEight", "8b", 8.0, "q8_0"),
    ]
    macros.section("model-size sweep")
    for prefix, macro_suffix, model, params, scheme in size_macros:
        decode_name = f"{prefix}-decode.json"
        prefill_name = f"{prefix}-prefill.json"
        if decode_name not in loaded or prefill_name not in loaded:
            continue
        size_names.extend([decode_name, prefill_name])
        decode, _ = metric(decode_name, "decode_tokens_per_second")
        prefill, _ = metric(prefill_name, "prefill_tokens_per_second")
        macros.macro(f"Decode{macro_suffix}", fmt_tps(decode), decode_name,
                     "decode_tokens_per_second.median",
                     result_commit(loaded[decode_name]))
        macros.macro(f"Prefill{macro_suffix}", fmt_tps(prefill), prefill_name,
                     "prefill_tokens_per_second.median",
                     result_commit(loaded[prefill_name]))
        size_rows.append(f"{model} {params} {scheme} {decode:.2f} {prefill:.2f}")
    for scheme, label in SCHEMES:
        decode_name = f"4b-{scheme}-decode.json"
        prefill_name = f"4b-{scheme}-prefill.json"
        if decode_name in loaded and prefill_name in loaded:
            decode, _ = metric(decode_name, "decode_tokens_per_second")
            prefill, _ = metric(prefill_name, "prefill_tokens_per_second")
            size_rows.append(f"4b 4.0 {label} {decode:.2f} {prefill:.2f}")
    if size_names:
        outputs["data/model-size.dat"] = dat_text(
            size_names, loaded, "model params_b scheme decode prefill",
            sorted(size_rows, key=lambda row: float(row.split()[1])))

    # Speculative decoding (ungated spec.json, gated spec-gated.json).
    if "spec.json" in loaded and "spec-gated.json" in loaded:
        spec = loaded["spec.json"]["workloads"]
        gated = loaded["spec-gated.json"]["workloads"]
        spec_commit = result_commit(loaded["spec.json"])
        gated_commit = result_commit(loaded["spec-gated.json"])
        suffixes = {"repetitive": "Rep", "code": "Code",
                    "grounded": "Grounded", "chat": "Chat"}
        macros.section("speculative decoding, ungated (spec.json)")
        macros.macro("SpecBaseRep", fmt_tps(spec["repetitive"]["baseline_decode_tps"]),
                     "spec.json", "repetitive.baseline_decode_tps", spec_commit)
        macros.macro("SpecDecodeRep", fmt_tps(spec["repetitive"]["spec_decode_tps"]),
                     "spec.json", "repetitive.spec_decode_tps", spec_commit)
        for workload, suffix in suffixes.items():
            macros.macro(f"SpecSpeedup{suffix}",
                         fmt_ratio(spec[workload]["speedup"]),
                         "spec.json", f"{workload}.speedup", spec_commit)
        macros.section("speculative decoding, adaptive gate (spec-gated.json)")
        for workload, suffix in suffixes.items():
            macros.macro(f"SpecAlpha{suffix}",
                         fmt_ratio(gated[workload]["accept_rate"]),
                         "spec-gated.json", f"{workload}.accept_rate",
                         gated_commit)
        for workload, suffix in suffixes.items():
            macros.macro(f"SpecBlk{suffix}",
                         fmt_ratio(gated[workload]["block_efficiency"]),
                         "spec-gated.json", f"{workload}.block_efficiency",
                         gated_commit)
        for workload, suffix in suffixes.items():
            macros.macro(f"GatedSpeedup{suffix}",
                         fmt_ratio(gated[workload]["speedup"]),
                         "spec-gated.json", f"{workload}.speedup", gated_commit)
        for workload, suffix in suffixes.items():
            macros.macro(f"GatedDuty{suffix}",
                         fmt_ratio(gated[workload]["draft_duty"]),
                         "spec-gated.json", f"{workload}.draft_duty",
                         gated_commit)
        spec_rows = [
            f"{workload} {gated[workload]['accept_rate']:.2f} "
            f"{spec[workload]['speedup']:.2f} {gated[workload]['speedup']:.2f} "
            f"{gated[workload]['draft_duty']:.2f}"
            for workload in SPEC_WORKLOADS]
        outputs["data/spec-speedup.dat"] = dat_text(
            ["spec.json", "spec-gated.json"], loaded,
            "workload alpha ungated gated duty", spec_rows)

    # Server batch scaling.
    if "server-batch.json" in loaded:
        batch = loaded["server-batch.json"]["aggregate_tokens_per_second"]
        commit = result_commit(loaded["server-batch.json"])
        macros.section("server batch scaling (server-batch.json)")
        rows = []
        for count in ("1", "2", "4", "8"):
            name = DIGIT_NAMES[count]
            macros.macro(f"BatchChoicesTps{name}",
                         fmt_tps(batch["choices"][count]),
                         "server-batch.json",
                         f"aggregate_tokens_per_second.choices.{count}", commit)
            macros.macro(f"BatchConnsTps{name}",
                         fmt_tps(batch["concurrent"][count]),
                         "server-batch.json",
                         f"aggregate_tokens_per_second.concurrent.{count}",
                         commit)
            rows.append(f"{count} {batch['choices'][count]:.2f} "
                        f"{batch['concurrent'][count]:.2f}")
        outputs["data/server-batch.dat"] = dat_text(
            ["server-batch.json"], loaded,
            "n choices_tps connections_tps", rows)

    # Cross-request prefix reuse.
    if "prefix-reuse.json" in loaded:
        prefix = loaded["prefix-reuse.json"]
        commit = result_commit(prefix)
        macros.section("cross-request prefix reuse (prefix-reuse.json)")
        macros.macro("PrefixPromptTokens", fmt_int_comma(prefix["prompt_tokens"]),
                     "prefix-reuse.json", "prompt_tokens", commit)
        macros.macro("PrefixAdoptedTokens", fmt_int_comma(prefix["adopted_tokens"]),
                     "prefix-reuse.json", "adopted_tokens", commit)
        macros.macro("PrefixColdSeconds", f"{prefix['cold_prompt_ms'] / 1000:.1f}",
                     "prefix-reuse.json", "cold_prompt_ms / 1000", commit)
        macros.macro("PrefixWarmMs", f"{prefix['warm_prompt_ms']:.0f}",
                     "prefix-reuse.json", "warm_prompt_ms", commit)
        macros.macro("PrefixSpeedup", fmt_tps(prefix["ttft_speedup"]),
                     "prefix-reuse.json", "ttft_speedup", commit)

    # Serving under load.
    if "load.json" in loaded:
        load = loaded["load.json"]
        sweeps = load["sweeps"]
        commit = result_commit(load)
        macros.section("serving under open-loop load (load.json)")
        by_rate = sorted(sweeps, key=lambda s: s["request_rate_per_s"])
        low, high = by_rate[0], by_rate[-1]
        peak = max(sweeps, key=lambda s: s["output_tokens_per_s"])
        macros.macro("LoadPeakTps", fmt_tps(peak["output_tokens_per_s"]),
                     "load.json", "max output_tokens_per_s", commit)
        macros.macro("LoadRateHigh", f"{high['request_rate_per_s']:g}",
                     "load.json", "max request_rate_per_s", commit)
        macros.macro("LoadTtftPFiftyLow",
                     f"{low['ttft_s']['p50'] * 1000:.0f}",
                     "load.json", "lowest-rate ttft_s.p50 (ms)", commit)
        macros.macro("LoadTtftPNinetyNineHigh",
                     f"{high['ttft_s']['p99']:.1f}",
                     "load.json", "highest-rate ttft_s.p99 (s)", commit)
        macros.macro("LoadGoodputHigh",
                     fmt_ratio(high["goodput_fraction"]),
                     "load.json", "highest-rate goodput_fraction", commit)
        rows = [
            f"{s['request_rate_per_s']:g} {s['output_tokens_per_s']:.2f} "
            f"{s['ttft_s']['p50']:.4f} {s['ttft_s']['p90']:.4f} "
            f"{s['ttft_s']['p99']:.4f} {s['tpot_s']['p50']:.4f} "
            f"{s['tpot_s']['p99']:.4f} {s['goodput_fraction']:.3f}"
            for s in by_rate]
        outputs["data/load-latency.dat"] = dat_text(
            ["load.json"], loaded,
            "rate output_tps ttft_p50 ttft_p90 ttft_p99 tpot_p50 tpot_p99 "
            "goodput", rows)

    # Perplexity by weight scheme.
    if "ppl-4b.json" in loaded:
        ppl = loaded["ppl-4b.json"]["results"]
        commit = result_commit(loaded["ppl-4b.json"])
        macros.section("perplexity by weight scheme (ppl-4b.json)")
        ppl_macro = {"bf16": "PplBF", "q8_0": "PplQEight",
                     "affine4_gs32": "PplAffine"}
        for scheme, name in ppl_macro.items():
            macros.macro(name, fmt_ppl(ppl[scheme]["ppl"]), "ppl-4b.json",
                         f"results.{scheme}.ppl", commit)
        base = ppl["bf16"]["ppl"]
        macros.macro("PplDeltaQEightPct",
                     fmt_pct((ppl["q8_0"]["ppl"] - base) / base * 100),
                     "ppl-4b.json", "(q8_0 - bf16)/bf16 * 100", commit)
        macros.macro("PplDeltaAffinePct",
                     fmt_pct((ppl["affine4_gs32"]["ppl"] - base) / base * 100),
                     "ppl-4b.json", "(affine4_gs32 - bf16)/bf16 * 100", commit)
        rows = []
        for scheme, label in SCHEMES:
            decode_name = f"4b-{scheme}-decode.json"
            if scheme in ppl and decode_name in loaded:
                decode = loaded[decode_name]["decode_tokens_per_second"]["median"]
                rows.append(f"{label} {ppl[scheme]['ppl']:.4f} {decode:.2f}")
        if rows:
            outputs["data/quant-ppl.dat"] = dat_text(
                ["ppl-4b.json"] + [f"4b-{s}-decode.json" for s, _ in SCHEMES],
                loaded, "scheme ppl decode_tps", rows)

    # llama.cpp head-to-head.
    if "llamacpp-qwen3-4b.json" in loaded:
        llama = loaded["llamacpp-qwen3-4b.json"]["results"]
        commit = loaded["llamacpp-qwen3-4b.json"]["llamacpp"]["commit"][:7]
        macros.section("llama.cpp head-to-head (llamacpp-qwen3-4b.json)")
        llama_macro = {"bf16": "BF", "q8_0": "QEight", "q4_0": "QFour"}
        for scheme, suffix in llama_macro.items():
            if scheme not in llama:
                continue
            decode = llama[scheme]["decode"]
            prefill = llama[scheme]["prefill"]
            macros.macro(f"LlamaDecode{suffix}",
                         fmt_tps(decode["avg_tokens_per_second"]),
                         "llamacpp-qwen3-4b.json",
                         f"results.{scheme}.decode.avg_tokens_per_second",
                         commit, f"stddev {decode['stddev']:.2g}")
            macros.macro(f"LlamaPrefill{suffix}",
                         fmt_tps(prefill["avg_tokens_per_second"]),
                         "llamacpp-qwen3-4b.json",
                         f"results.{scheme}.prefill.avg_tokens_per_second",
                         commit, f"stddev {prefill['stddev']:.2g}")
        # Ratios use the matched-2048 Kipp prefill point, since llama-bench
        # measured prefill at n_prompt=2048.
        if "bf16" in llama and "4b-bf16-prefill2k.json" in loaded:
            kipp_prefill = loaded["4b-bf16-prefill2k.json"][
                "prefill_tokens_per_second"]["median"]
            ratio = llama["bf16"]["prefill"]["avg_tokens_per_second"] / kipp_prefill
            macros.macro("LlamaPrefillRatioBF", f"{ratio:.1f}",
                         "llamacpp-qwen3-4b.json",
                         "bf16 prefill / kipp bf16 prefill2k median", commit)
        if "q8_0" in llama and "4b-q8_0-decode.json" in loaded:
            kipp_decode = loaded["4b-q8_0-decode.json"][
                "decode_tokens_per_second"]["median"]
            ratio = kipp_decode / llama["q8_0"]["decode"]["avg_tokens_per_second"]
            macros.macro("LlamaDecodeRatioQEight", f"{ratio:.2f}",
                         "llamacpp-qwen3-4b.json",
                         "kipp q8_0 decode median / llama q8_0 decode avg",
                         commit)

    # Fault injection.
    if "faults.json" in loaded:
        faults = loaded["faults.json"]["faults"]
        commit = result_commit(loaded["faults.json"])
        macros.section("fault injection (faults.json)")
        seeded = [name for name in faults if name != "control"]
        macros.macro("FaultCount", NUMBER_WORDS[len(seeded)], "faults.json",
                     "seeded fault modes", commit)
        fault_macro = {"control": "FaultControlNmse",
                       "read-block": "FaultNmseReadBlock",
                       "read-slot": "FaultNmseReadSlot",
                       "rollover": "FaultNmseRollover",
                       "swap-kv": "FaultNmseSwapKv"}
        for fault, name in fault_macro.items():
            if fault in faults:
                macros.macro(name, fmt_nmse(faults[fault]["reference_nmse"]),
                             "faults.json", f"faults.{fault}.reference_nmse",
                             commit)

    # Gate costs.
    if "gate-costs.json" in loaded:
        costs = loaded["gate-costs.json"]
        commit = result_commit(costs)
        macros.section("gate costs (gate-costs.json)")
        for gate, entry in costs["gates"].items():
            name = GATE_MACRO_NAMES.get(gate)
            if name is None:
                name = "".join(part.capitalize()
                               for part in gate.replace("_", "-").split("-"))
            macros.macro(f"GateCost{name}", f"{entry['seconds']:.1f}",
                         "gate-costs.json", f"gates.{gate}.seconds", commit)
        minutes = f"{costs['total_seconds'] / 60:.0f}"
        macros.macro("GateCostTotalMinutes", minutes, "gate-costs.json",
                     f"total_seconds {costs['total_seconds']} -> minutes",
                     commit)
        macros.macro("GateCostTotal", minutes, "gate-costs.json",
                     f"total_seconds {costs['total_seconds']} -> minutes",
                     commit)
        # Metal full-logit NMSE as reported by the --phase3-metal gate run.
        if "metal_full_logit_nmse" in costs:
            macros.macro("MetalNmse",
                         fmt_nmse(costs["metal_full_logit_nmse"]),
                         "gate-costs.json", "metal_full_logit_nmse", commit)

    outputs["generated/results-macros.tex"] = macros.text()
    return outputs, macros.count


def git_input_problems(loaded, missing, allow_missing):
    """--check: every expected input committed and unmodified."""
    problems = []
    for name in missing:
        message = f"missing input: bench/results/{name}"
        if allow_missing:
            print(f"warning: {message}", file=sys.stderr)
        else:
            problems.append(message)
    for name in loaded:
        rel = f"bench/results/{name}"
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", rel],
            cwd=ROOT, capture_output=True, text=True)
        if tracked.returncode != 0:
            problems.append(f"untracked input: {rel}")
            continue
        status = subprocess.run(["git", "status", "--porcelain", rel],
                                cwd=ROOT, capture_output=True, text=True)
        if status.stdout.strip():
            problems.append(f"modified input (commit it first): {rel}")
    return problems


def spec_coherence_problems(loaded):
    """The spec A/B is only meaningful if both halves share one baseline."""
    problems = []
    if "spec.json" not in loaded or "spec-gated.json" not in loaded:
        return problems
    spec = loaded["spec.json"]["workloads"]
    gated = loaded["spec-gated.json"]["workloads"]
    # New-schema files record which gate setting produced them; when the
    # key is present, the ungated/gated roles must not be swapped.
    expected_gates = (("spec.json", "off"), ("spec-gated.json", "on"))
    for name, expected in expected_gates:
        gate = config_field(loaded[name], "gate")
        if gate is not None and gate != expected:
            problems.append(
                f"{name} records configuration.gate '{gate}' but this file "
                f"must hold the gate-{expected} run")
    # The baselines are compared against the plain q8_0 decode number, so
    # the spec runs must have used the q8_0 model.
    for name in ("spec.json", "spec-gated.json"):
        scheme = model_scheme(loaded[name])
        if scheme is not None and scheme != "q8_0":
            problems.append(
                f"{name} was measured on scheme '{scheme}'; the spec A/B "
                "protocol uses the q8_0 model")
    # The two halves of the A/B must agree with each other. They are NOT
    # compared against the plain 4b-q8_0 decode median: spec workloads
    # legitimately decode slower than the short-prompt microbenchmark
    # because their prompts carry much longer attention contexts
    # (repetitive/code baselines sit near 50 tok/s at 98 tok/s decode).
    for workload in SPEC_WORKLOADS:
        base_a = spec[workload]["baseline_decode_tps"]
        base_b = gated[workload]["baseline_decode_tps"]
        if abs(base_a - base_b) / max(base_a, base_b) > 0.10:
            problems.append(
                f"spec baseline incoherent for '{workload}': spec.json "
                f"{base_a} vs spec-gated.json {base_b} (>10% apart; "
                "regenerate both in one session)")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify inputs are committed and outputs current")
    parser.add_argument("--allow-missing", action="store_true",
                        help="with --check: warn instead of fail on absent "
                             "inputs (interim between code and results commits)")
    args = parser.parse_args()

    loaded, missing = load_results()
    outputs, macro_count = generate(loaded)

    if args.check:
        problems = git_input_problems(loaded, missing, args.allow_missing)
        problems += spec_coherence_problems(loaded)
        for rel, text in outputs.items():
            path = ROOT / "paper" / rel
            if not path.is_file():
                problems.append(f"missing output: paper/{rel} "
                                "(run `make paper-data`)")
            elif path.read_text() != text:
                problems.append(f"stale output: paper/{rel} "
                                "(run `make paper-data` and commit)")
        if problems:
            print("paper-check failed:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        print(f"paper-check ok: {len(loaded)} inputs, "
              f"{len(outputs)} outputs, {macro_count} macros")
        return 0

    if missing:
        print("skipped absent inputs: " + ", ".join(missing), file=sys.stderr)
    GENERATED.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    for rel, text in outputs.items():
        path = ROOT / "paper" / rel
        path.write_text(text)
        print(f"wrote paper/{rel}")
    print(f"{macro_count} macros from {len(loaded)} result files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
