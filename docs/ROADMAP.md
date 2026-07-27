# Kipp Roadmap

This sequence is binding unless `ARCHITECTURE.md` is explicitly revised.
Correctness gates every phase. Status notes summarize delivered work and
link measured claims back to the benchmark records.

**Status (v0.0.4):** This release is about Metal prefill, which was the one
axis where Kipp measurably lost to llama.cpp. Prefill is now **~2.5× faster on
every weight scheme** at a 2,048-token prompt (BF16 504→1308, Q8_0 453→1142,
affine4 482→1140 tok/s, same-session A/B on an M5 Max) with **decode unchanged**,
narrowing the llama.cpp prefill gap from 4.5× to ~1.7× while decode stays ~1.6×
ahead. The cause was occupancy rather than bandwidth: weight traffic follows the
in-kernel token tile, not the round size, so a 32-token round left the 1,024-row
K and V projections dispatching 16 threadgroups onto a 40-core device. Rounds are
now device-derived and wide, the matmul token tile is per kernel (wider for BF16,
unchanged for the quantized kernels, where widening it costs 88%), and one
overloaded constant became three so that widening the round no longer drags the
split-K partial buffer with it — which alone was worth 11% of decode. Activation
scratch moved from every session to one per model.

Correctness work landed alongside it, and was the prerequisite: the pinned test
vectors are three tokens long while a round only takes the simdgroup-matrix path
at eight or more, so the primary numeric gates had **never executed the matrix
kernels at all**. The new `--prefill-metal` gate covers a ragged multi-round
prefill against both the vector path and the CPU oracle and prints a full-logit
fingerprint; `kipp_flash_gqa_prefill` gained its first operator test;
`KIPP_METAL_REQUIRE_MMA` and a shader-geometry probe turn two classes of silent
degradation into load failures; and an unsigned-wrap out-of-bounds write in all
three matmul kernels was fixed. An opt-in **Q8_0 KV cache** (`--kv-quant q8_0`)
also ships, ~1.9× smaller than BF16 and gated for tolerance and placement
invariance — a memory feature, not a speed one.

**Status (v0.0.3):** Kipp delivered the items that v0.0.2 deferred and
hardened the serving path. **Cross-request
KV prefix sharing is now the CPU/Metal serving default**: pooled sessions
share one model-owned slab through a content-addressed 32-token block pool
(publish-at-finish, memcmp-verified), gated by `--pooled-cpu` /
`--pooled-metal` and measured at 175× TTFT on a repeated 6,890-token
prompt. The Metal **flash-attention prefill kernel is live** (a
reserved-MSL-keyword bug had silently disabled every simdgroup-matrix
kernel for two days; the benchmark harness now refuses to record numbers
from a fallback build), putting quantized prefill at BF16 parity
(528/488/509 tok/s at 348 tokens on the M5 Max). The server gained 32-way
concurrent decode, idle-connection reaping, and TTFT/queue-wait metrics;
the CLI gained a multi-turn `--chat` REPL (suffix-only KV evaluation) and
a wikitext `--ppl` perplexity mode; sampling gained zero-copy fast paths;
`kipp_session_eval_scored` exposes tolerance-bound multi-row scoring; the
JSON and HTTP parsers moved to their own fuzz-tested translation units;
and the whole registry surface was revalidated on an ephemeral NVIDIA
H100 (four checkpoints, worst NMSE 5.9e-7). Speculation now measures a
paired-baseline A/B with an adaptive-gate floor of 0.84× and above-parity
code decoding.

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
ephemeral cloud runs. At that point, cross-request KV pooling and Metal
flash-attention prefill were still deferred; both are now delivered, as is
an opt-in Q8_0 quantized KV cache.

**v2 expansion (approved 2026-07-16):** scope grew from one pinned
checkpoint to the pinned Qwen3 dense family via a compiled-in registry
(`src/kipp_checkpoints.h`); dimensions, context, RoPE theta, embedding
tying, and stop tokens became per-checkpoint runtime configuration with the
4B refactor gated byte-identical.

Delivered since: validated checkpoints spanning Qwen3 dense 0.6B through
32B on CPU plus a GPU backend (Metal on M5-class hardware, CUDA on Verda
A100/H100 instances);
`/v1/chat/completions` with a native ChatML renderer; the full sampling
surface (top-k, min-p, penalties, logit_bias); a `/metrics` endpoint; and
**weight quantization** — Q8_0 (near-lossless, ~1.6× decode, brings 8B
under the M5 Metal buffer cap) and affine 4-bit gs32 (~2× decode, 2.6×
smaller, coherent output), both gated on CPU + Metal. Production KV block
pooling with cross-request prefix caching is delivered and is the serving
default on CPU/Metal. An opt-in **Q8_0 KV cache** (`--kv-quant q8_0`) is
delivered on CPU + Metal, ~1.9× smaller than BF16 and gated for tolerance
and placement invariance. Remaining: a token-budget scheduler and 4-bit KV,
each behind its own CPU-vs-GPU gate, plus a threadgroup-staged K-blocked
matmul, which is what the remaining prefill gap to llama.cpp now needs. Prompt-lookup and draft-model speculative
decoding plus generated-token logprobs are already delivered.

## Phase 0 — Specify the model

- Pin `Qwen/Qwen3-4B-Base` and its exact revision.
- Confirm model, tokenizer, and weight licenses.
- Fix tensor shapes, tokenizer behavior, precision, memory, and API boundaries.

## Phase 1 — Build the CPU reference path

- Convert the pinned BF16 weights into Kipp's strict GGUF subset.
- Implement mmap-backed loading, native tokenization, scalar operators, and
  the readable full-prompt forward pass.
- Pass deterministic full-logit vectors with exact argmax and NMSE at most
  `5e-5`.

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
- Deliver cross-session KV sharing only after backend pools, cache-pressure
  admission, cancellation, and eviction are integrated and gated.

## Phase 6 — Serve

- Add the approved OpenAI Completions and Chat Completions subset, including
  optional SSE streaming and Prometheus metrics.
- Keep HTTP, scheduling, and backend execution separate.

## Phase 7 — Explicitly reviewed extensions

Delivered through explicit reviews:

- Q8_0 and affine 4-bit weight quantization
- prompt-lookup and draft-model speculative decoding, plus multi-row logits
- pinned Qwen3 dense checkpoint registry and native chat rendering
- cross-request KV prefix sharing on CPU and Metal

Still deferred:

- ROCm/HIP support, on a separate community-maintained branch
- additional model families
- generic tensor-runtime or arbitrary-GGUF compatibility
- radix-tree prefix indexing, SSD streaming, and 4-bit KV
  (cross-session prefix caching and an opt-in Q8_0 KV cache are delivered)
- broad API parity with llama.cpp, vLLM, or SGLang
