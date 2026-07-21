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
  projections; CPU + Metal. Quantized **simdgroup-matrix (MMA) prefill
  kernels** close the old vector-kernel prefill regression (~2.3× lift,
  parity with BF16 prefill).
- **Serving**: OpenAI Completions + Chat Completions, full sampling pack,
  logprobs, timings, Prometheus `/metrics` (now including KV-pool gauges).
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
  (six bitwise cases each). Measured 211× TTFT on a repeated 6.9k-token
  prompt.
- **Speculative decoding**: prompt-lookup + **adaptive gating**
  (`kipp_spec_gate`): token-identical to greedy (gated; verify rounds force
  vector kernels — MMA reduction order flips near-tie argmaxes), 2.2× on
  repetitive text, ≥0.85× elsewhere.
- **Mutation study**: `build/kipp_test_fault` (KIPP_FAULT=1..4) +
  `--fault-reference`; results in `bench/results/faults.json` — tolerance
  and scramble gates are complementary; the rollover fault has NMSE exactly
  0 vs reference and only the scramble gate catches it.

## Paper (`paper/`)

Full systems-paper structure (11 sections): Proposition 1
(placement invariance, H1–H3), scramble-gate walkthrough + gate-lattice
figures, mutation-study detection matrix (§7.3, the centerpiece), llama.cpp
head-to-head (Kipp wins decode, loses prefill 6.6×), context scaling,
batching, prefix sharing, gate costs. Every number is a macro bound to a
committed `bench/results/*.json` (rule in `paper/README.md`). Compiles with
`tectonic main.tex`.

## Working rules

- Gate on real hardware before claiming a backend works; state which ones you
  ran. Build/gate commands in `AGENT.md` + `REPRODUCE.md`.
- **Hardware**: this machine is an Apple **M5 Max (40-core GPU, 128 GB)**
  since 2026-07-20; earlier committed numbers were from a base M5 and are
  ~4× lower. Never mix them; result files self-describe hardware.
- Commit, branch, push, PR, or merge only when the user asks explicitly.
- CUDA is validated on ephemeral cloud GPUs at milestones, not locally, and
  is frozen: contiguous KV, BF16-only, final-row logits.
- Longer-lived cross-session notes live in the user's Claude memory, not
  here; keep this file to durable, repo-level context.
