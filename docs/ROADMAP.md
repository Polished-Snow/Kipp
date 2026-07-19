# Kipp Roadmap

This sequence is binding unless `ARCHITECTURE.md` is explicitly revised.
Correctness gates every phase; dates and speculative performance claims are
intentionally absent.

**Status (v0.0.2):** This release expands Kipp from the single pinned
checkpoint of v0.0.1 to the **Qwen3 dense family** (0.6B–32B, base +
instruct) behind a strict compiled-in checkpoint registry, with the BF16
reference path still byte-identical to v0.0.1. It adds Q8_0 and 4-bit affine
quantization (CPU + Metal, bit-accurate); an OpenAI Completions **and** Chat
Completions server (native Qwen3 ChatML, full sampling pack, generated-token
logprobs, `stream_options.include_usage`, `timings`, Prometheus `/metrics`)
on a FIFO event-loop scheduler with chunked prefill, admission control, and
continuous batching (`kipp_eval_batch`, gated batched-equals-isolated);
multi-logit evaluation (`kipp_session_eval_n`) and greedy prompt-lookup
speculative decoding (token-identical to plain greedy); and **paged KV** on
both the CPU oracle and the Metal backend — a per-session 32-position block
table gated bitwise-equal to the contiguous layout (`--paged-cpu`,
`--paged-metal`). Everything is gated on Apple M5 (CPU + Metal); the whole
family plus quantization is additionally validated on NVIDIA A100 via
ephemeral cloud runs. Still deferred: the cross-request KV block pool
(`src/kipp_kv_pool.c` exists and is unit-tested but not yet wired into the
backends), Metal flash-attention prefill, and quantized KV.

**v2 expansion (approved 2026-07-16):** scope grew from one pinned
checkpoint to the pinned Qwen3 dense family via a compiled-in registry
(`src/kipp_checkpoints.h`); dimensions, context, RoPE theta, embedding
tying, and stop tokens became per-checkpoint runtime configuration with the
4B refactor gated byte-identical.

Delivered since: validated checkpoints spanning Qwen3 dense 0.6B through
32B on CPU plus a GPU backend (Metal on M5, CUDA on a Verda A100);
`/v1/chat/completions` with a native ChatML renderer; the full sampling
surface (top-k, min-p, penalties, logit_bias); a `/metrics` endpoint; and
**weight quantization** — Q8_0 (near-lossless, ~1.6x decode, brings 8B
under the M5 Metal buffer cap) and affine 4-bit gs32 (~2x decode, 2.6x
smaller, coherent output), both gated on CPU + Metal. Remaining: production
KV block pooling with cross-request prefix caching, a token-budget scheduler,
and quantized KV — each behind its own CPU-vs-GPU gate. Prompt-lookup
speculative decoding and generated-token logprobs are already delivered.

## Phase 0 — Specify the model

- Pin `Qwen/Qwen3-4B-Base` and its exact revision.
- Confirm model, tokenizer, and weight licenses.
- Fix tensor shapes, tokenizer behavior, precision, memory, and API boundaries.

## Phase 1 — Build the CPU reference path

- Convert the pinned BF16 weights into Kipp's strict GGUF subset.
- Implement mmap-backed loading, native tokenization, scalar operators, and
  the readable full-prompt forward pass.
- Pass deterministic full-logit vectors with exact argmax and NMSE at most
  `1e-5`.

## Phase 2 — Add the KV cache

- Add bounded contiguous BF16 K/V storage and session lifecycle.
- Match no-cache CPU recomputation for incremental decoding.

## Phase 3 — Add the Metal backend

- Implement Apple Silicon kernels behind the fixed backend boundary.
- Validate prefill and decode against CPU on actual Metal hardware.

## Phase 4 — Add the CUDA backend

- Implement isolated resident-weight CUDA execution.
- Validate on the provided NVIDIA machine; stop if hardware is unavailable.

## Phase 5 — Continuous batching

- Deliver batched multi-sequence evaluation and a small FIFO scheduler.
- Require batched requests to reproduce isolated execution.
- Keep cross-session KV sharing separate until backend pools, cache-pressure
  admission, cancellation, and eviction are integrated.

## Phase 6 — Serve

- Add the approved OpenAI Completions and Chat Completions subset, including
  optional SSE streaming and Prometheus metrics.
- Keep HTTP, scheduling, and backend execution separate.

## Phase 7 — Explicitly reviewed extensions

Delivered through explicit reviews:

- Q8_0 and affine 4-bit weight quantization
- prompt-lookup speculative decoding and multi-row logits
- pinned Qwen3 dense checkpoint registry and native chat rendering

Still deferred:

- ROCm/HIP support, on a separate community-maintained branch
- additional model families
- generic tensor-runtime or arbitrary-GGUF compatibility
- cross-session prefix/radix caching, SSD streaming, and quantized KV
- broad API parity with llama.cpp, vLLM, or SGLang
