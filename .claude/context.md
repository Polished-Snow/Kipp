# Kipp — project context for Claude Code

Read [`AGENT.md`](../AGENT.md) first: it is the tool-agnostic source of truth
for everyone working in this repo. This file is the Claude-specific companion
and a snapshot of where the project stands.

## What Kipp is

A small, hand-written C11 inference engine for the **pinned Qwen3 dense
family** (0.6B–32B, base + instruct), registered in
`src/kipp_checkpoints.h`. It is deliberately narrow — not a generic GGUF or
Qwen runner. Model support grows only by adding a registry entry and passing
the gates, never by loosening validation. The CPU scalar path is the
correctness oracle; Metal (Apple) and CUDA (NVIDIA) are the fast backends and
must never silently diverge from it.

## Capabilities in place (all gated)

- **Model family + registry**: strict per-checkpoint revision/shape
  validation; BF16 reference path byte-identical to v0.0.1.
- **Quantization**: Q8_0 and 4-bit affine (gs32) on the seven per-layer
  projections; CPU + Metal. Quality is measured (wikitext-2 PPL 7.731 /
  7.733 / 8.171 for bf16/q8_0/affine4 via the CLI `--ppl`/`--ppl-window`
  mode, and `kipp_session_eval_scored` lets scoring use the matrix
  kernels). **Prefill runs on the simdgroup-matrix kernels at quantized ≈
  BF16 parity** (528/488/509 tok/s @348, 481/441/466 @2k on the M5 Max).
  A reserved-MSL-keyword bug silently disabled ALL MMA kernels
  2026-07-20→22 (vector fallback passed every gate); fixed, and
  `tools/bench.py` now hard-fails on the fallback warning.
- **Serving**: OpenAI Completions + Chat Completions, full sampling pack,
  logprobs, timings, Prometheus `/metrics` (KV-pool gauges + TTFT/
  queue-wait/decode-time counters). Hardened 2026-07-22: 32-way concurrent
  decode (SERVER_MAX_GENERATIONS == KIPP_EVAL_BATCH_LIMIT),
  idle-connection reaping (`--idle-timeout`), JSON/HTTP parsers extracted
  to `src/kipp_json.c`/`src/kipp_http.c` with fuzz tests. Sampling has
  zero-copy fast paths (bitwise-identical tokens). CLI has a multi-turn
  `--chat` REPL (suffix-only KV evaluation across turns, prefix property
  unit-tested per variant).
- **Paged KV (delivered)**: CPU and Metal address KV through per-session
  32-position block tables; the scramble gates (`--paged-cpu`,
  `--paged-metal`) prove placement-invariance bitwise over a **3-block/96-
  token** sequence (2 blocks was degenerate for rollover faults — found by
  the mutation study). CUDA remains contiguous.
- **Cross-request KV prefix sharing (delivered)**: pooled sessions
  (`kipp_model_open_pooled`, CPU + Metal) borrow one model-owned slab;
  immutable shared full blocks, publish-at-finish, content-addressed pool
  with memcmp verification. Server default for CPU/Metal (`--kv-pool-mib`);
  reservation-based admission; gates `--pooled-cpu` / `--pooled-metal`
  (identity + shared-prefix bitwise; the batched-mixed case holds the
  batching contract — bitwise on CPU, 1e-4+argmax on Metal). Measured
  175× TTFT on a repeated 6.9k-token prompt (prefix-reuse.json).
- **Speculative decoding**: prompt-lookup + **adaptive gating**
  (`kipp_spec_gate`): token-identical to greedy (gated; verify rounds force
  vector kernels — MMA reduction order flips near-tie argmaxes). Paired
  A/B (`spec_bench.py --gate both`, drift-immune shared baseline): gated
  2.27× on repetitive text, 1.24× on code (above parity), floor 0.84×.
- **Draft-model speculation**: `--draft-model` uses a smaller registered
  checkpoint to propose up to eight tokens; target verification preserves
  the target model's plain greedy sequence (`make test-draft-spec`).
- **Long-context Metal decode**: split-K GQA partitions the KV scan across
  up to eight threadgroups past 1,024 cached positions. The
  `--longctx-metal` gate checks partition invariance.
- **Mutation study**: `build/kipp_test_fault` (KIPP_FAULT=1..4) +
  `--fault-reference`; results in `bench/results/faults.json` — tolerance
  and scramble gates are complementary; the rollover fault has NMSE exactly
  0 vs reference and only the scramble gate catches it.

## Working rules

- Gate on real hardware before claiming a backend works; state which ones you
  ran. Build/gate commands live in `AGENT.md` and `docs/REPRODUCING.md`.
- Measure only via the GPU steady-state protocol in `bench/README.md`.
- **Hardware**: this machine is an Apple **M5 Max (40-core GPU, 128 GB)**
  since 2026-07-20; earlier committed numbers were from a base M5 and are
  ~4× lower. Never mix them; result files self-describe hardware.
- Commit, branch, push, PR, or merge only when the user asks explicitly.
- CUDA is validated on ephemeral cloud GPUs at milestones, not locally,
  and is frozen: contiguous KV, BF16-only, final-row logits. Last
  revalidated 2026-07-22 on a Verda H100 (4 checkpoints, worst NMSE
  5.9e-7; `bench/results/cuda-h100-gates.json`); runbook in
  `docs/RELEASING.md`.
- Longer-lived cross-session notes live in the user's Claude memory, not
  here; keep this file to durable, repo-level context.
