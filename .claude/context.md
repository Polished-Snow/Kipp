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
  validation; the BF16 reference path is byte-identical to the original
  v0.0.1 single-checkpoint engine.
- **Quantization**: Q8_0 (near-lossless) and 4-bit affine (gs32) on the seven
  per-layer projections; CPU + Metal, bit-accurate.
- **Serving**: OpenAI Completions + Chat Completions (native Qwen3 ChatML),
  full sampling pack, generated-token logprobs, `stream_options.include_usage`,
  a llama.cpp-style `timings` object, and Prometheus `/metrics`.
- **Multi-logit eval** (`kipp_session_eval_n`) and greedy **speculative
  decoding** (prompt-lookup, CLI `--spec`), token-identical to plain greedy.
- **Paged KV** (Phase 5, in progress): both the CPU oracle and the Metal
  backend address KV through a per-session 32-position block table. Identity
  mapping is byte-for-byte the contiguous layout; the `--paged-cpu` /
  `--paged-metal` gates prove correctness under a scrambled table. The shared
  cross-request block pool lives in `src/kipp_kv_pool.c` (unit-tested,
  not yet wired into the backends). CUDA is still contiguous.

## Working rules

- Gate on real hardware before claiming a backend works; state which ones you
  ran. Build/gate commands are in `AGENT.md`.
- Commit, branch, push, PR, or merge only when the user asks explicitly.
- CUDA is validated on ephemeral cloud GPUs at milestones, not locally.
- Longer-lived cross-session notes live in the user's Claude memory, not here;
  keep this file to durable, repo-level context.
