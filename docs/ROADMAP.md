# Kipp Roadmap

This sequence is binding unless `ARCHITECTURE.md` is explicitly revised.
Correctness gates every phase. Status notes summarize delivered work and
link measured claims back to the benchmark records.

**Status (v0.0.8):** The quantized projections join BF16 on the Metal 4
tensor units. `kipp_matmul_q8_0_tensor` and `kipp_matmul_affine4_tensor`
dequantize each weight block into a threadgroup bf16 tile and feed
`matmul2d`, taking **Q8_0 prefill @2,048 from 1,247 to 2,875 tok/s (2.31×)**
and affine4 1,250 → 2,857 (2.29×) — same-session A/B against the simdgroup
kernels each replaces, decode untouched. Selection is a pure function of
token count so the paged/pooled bitwise gates stay single-class; the
simdgroup quant kernels stay frozen for non-M5 devices,
`KIPP_METAL_TENSOR_DISABLE`, and Q8_0 KV, with tensor-state quant
fingerprints re-baselined. No llama.cpp head-to-head this release —
llama-bench's quantized prefill is thermally unstable on this laptop chassis
(2.1k–4.4k across thermal states in one session), so a fair same-session
comparison could not be taken. Remaining after this release: server prefill
chunk tuning, a token-budget scheduler, CUDA revalidation, and 4-bit KV.

**Status (v0.0.7):** Long-context prefill attention — the lever v0.0.6
named as next — now runs on the Metal 4 tensor units. Batched-prefill GQA
is processed one 1,024-KV-position panel at a time (block-table gather →
Q·Kᵀ `matmul2d` → panel-granular online softmax → P·V `matmul2d` →
cross-panel rescale), and the win tracks attention's growing share of the
wall: **+37% at a 12,800-token prefill, +26% at 6,400, +11% at 2,048**
(same-session A/B vs the simdgroup flash-attention kernel it replaces,
cool-start matched on an M5 Max), with decode untouched. At 12,800 tokens
Kipp reaches 2,410 tok/s against llama.cpp's 1,561 (**1.54×**) — llama.cpp's
flash-attention is simdgroup-only even on M5, so it stays on the slow
attention class at long context. It ships behind dual-state gating
(tensor-state + BF16 KV) with tensor fingerprints re-baselined tighter on
every scheme; the simdgroup attention class stays frozen for non-M5
devices, `KIPP_METAL_TENSOR_DISABLE`, and Q8_0 KV. Remaining after this
release: quantized tensor-matmul variants, server prefill chunk tuning, a
small-part panel-threshold, CUDA revalidation, and 4-bit KV.

**Status (v0.0.6):** The instruction-class gap v0.0.5 documented is closed.
A gated Metal 4 `mpp::tensor_ops` matmul path now runs BF16 layer
projections on M5-class devices: **prefill 1,312 → 3,682 tok/s (2.80×) at a
2,048-token prompt**, decode untouched, quantized schemes bit-frozen, at
parity with llama.cpp's own tensor path. The kernel deliberately departs
from llama.cpp's staged design — no operand staging (explicit threadgroup
staging lost its sixth straight measurement on this hardware) and FP32
activations consumed directly, which deletes the whole BF16 staging pass
and an activation rounding step; the path is numerically tighter than the
simdgroup class it replaces (pooled NMSE 5.1e-07 vs 2.1e-06). It ships
behind a runtime compile-and-probe ladder with an allow-list, env kill
switches, `KIPP_METAL_REQUIRE_TENSOR` as the anti-silent-fallback tripwire,
an affirmative disabled-line assertion in CI (which cannot run this path),
and per-kernel-class fingerprint tripwires. Remaining after this release:
the attention/elementwise residue now dominates prefill (1.40× at 12,800
tokens vs 2.80× at 2,048 — long-context attention is the next lever),
quant tensor variants, server prefill chunk tuning, and CUDA revalidation.

**Status (v0.0.5):** This release corrects the record and closes the campaign
the correction reframed. The published llama.cpp head-to-head proved **stale
in Kipp's favor** — the same command, binary, and weights measure ~2× the
recorded llama.cpp numbers today, so the comparison was re-measured
same-session and the README table replaced (decode: llama.cpp ~3% ahead on
matched schemes, not the claimed 1.7× Kipp win; prefill: llama.cpp
2.9×/2.25× ahead on BF16/Q8_0). The cause of that prefill lead is an instruction class: on M5,
llama.cpp routes GEMM through the Metal 4 tensor API (neural accelerators),
worth ~2.3–3× on prefill and nothing on decode; restricted to the
simdgroup-matrix class Kipp uses, llama.cpp trails Kipp on every scheme. A
new isolated matmul instrument (`--mm-bench-metal`) showed the simdgroup
kernels are compute/issue-bound (not traffic-bound), which retires the
planned "threadgroup-staged K-blocked matmul"; what shipped instead is a
bit-exact shared-dequant rotation of the quantized matmuls (**Q8_0 +10.2%,
affine4 +11.7%** at a 2,048-token prompt, decode unchanged) plus prebuilt
release binaries on every GitHub release. **The next engine campaign is a
gated `mpp::tensor_ops` matmul path** behind the same runtime-probe ladder
and oracle gates as the simdgroup kernels. CUDA was not revalidated.

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
and placement invariance. The **Metal 4 `mpp::tensor_ops` matmul path** is
delivered for BF16 projections on M5-class devices (prefill 2.80× at 2,048
tokens, at parity with llama.cpp's tensor path; the previously planned
K-blocked matmul was measured unnecessary — the simdgroup kernels were
compute-bound, not traffic-bound), **panel-flash attention** puts
long-context prefill GQA on the same tensor units (v0.0.7: +37% at 12,800
tokens, leading llama.cpp 1.54× there), and the **quantized projections**
followed (v0.0.8: Q8_0/affine4 prefill 2.3× via a dequant-to-threadgroup
tile feeding `matmul2d`). Remaining: server prefill chunk tuning, a
small-part panel-threshold, a token-budget scheduler, and 4-bit KV, each
behind its own CPU-vs-GPU gate.
Prompt-lookup and draft-model speculative decoding plus generated-token
logprobs are already delivered.

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
