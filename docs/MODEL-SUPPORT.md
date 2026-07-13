# Kipp Model Support

This document describes Kipp v0.0.1.

## Supported CPU and Metal target

- Upstream: `Qwen/Qwen3-4B-Base`
- Revision: `906bfd4b4dc7f14ee4320094d8b41684abff8539`
- Model license: Apache-2.0
- Architecture: dense Qwen3 decoder, 4.0B parameters, 36 layers
- Attention: 32 query heads, 8 KV heads, 128 dimensions per head
- Context declared by the model: 32,768 tokens
- Tokenizer: native Qwen2 byte-level BPE, 151,936-token vocabulary
- Source weights: three pinned BF16 safetensors shards
- Runtime artifact: Kipp's strict, BF16, Qwen3-only GGUF-v3 subset

The stateless CPU oracle accepts at most 256 prompt tokens and rounds K/V
through BF16 before attention. CPU sessions select a capacity from 1 through
32,768 tokens and own separate contiguous BF16 K and V slabs in
`[layer][position][kv_head][head_dim]` order. Cache storage costs 147,456 bytes
per token and is allocated once per session.

The pinned full-logit Phase 1 gate passes with exact argmax. Cached prefill,
incremental decode, reset, capacity, overflow, and independent-session tests
also pass against no-cache recomputation with exact argmax and NMSE at most
`1e-6`.

The Phase 3 Metal backend is validated on an Apple M5. It wraps the complete
read-only GGUF mmap in one shared no-copy Metal buffer, allocates persistent
activation and contiguous BF16 KV buffers per session, and executes prefill
and incremental decode entirely through Metal kernels. Prefill batches up to
32 tokens per command buffer; on devices with bfloat simdgroup matrices the
projections run as simdgroup-matrix multiplies over BF16-staged activations
with FP32 accumulation (elsewhere a token-tiled vector kernel is used, and
single-token decode always uses the vector kernel on FP32 activations).
Attention streams the KV cache with a split-K online-softmax kernel (eight
partial softmaxes per head merged in threadgroup memory) that materializes
no score buffer and keeps single-token decode parallel at long contexts.
Only final logits cross back to the host. The `Hi 世界` gate passes at every
checked position with exact argmax; the maximum CPU-versus-Metal full-logit
NMSE is `2.960e-8` for decode and `1.851e-6` for matrix-kernel prefill, both
well below the required `1e-4`.

Metal must be selected explicitly. A CPU-only binary returns
`KIPP_ERROR_UNSUPPORTED` for a Metal request, and a Metal initialization or
runtime shader compilation failure is fatal rather than falling back.

Core C additionally owns greedy and seeded temperature/top-p sampling
(`kipp_sample`), session truncation (`kipp_session_truncate`), and batched
multi-session evaluation (`kipp_eval_batch`, gated so batched output matches
isolated output — exactly on CPU, within `1e-4` on Metal). The server in
`src/kipp_server.c` exposes the OpenAI Completions subset from
`ARCHITECTURE.md` on loopback — optionally streamed over SSE — and serves
concurrent connections through a poll() event loop with FIFO admission,
chunked prefill, and batched decoding: requests and multi-choice
completions share each step's weight pass (measured ~3.1x aggregate decode
throughput with eight concurrent clients on the M5, and the same again for
`n = 8` within one request). Consecutive single-choice prompts sharing a
prefix reuse a cached session.

No quantized format is supported. No instruction-tuned, MoE, vision-language,
Qwen3-Next, Qwen3.5, other Qwen3 size, or arbitrary GGUF checkpoint is
supported. CUDA kernels exist in the tree but remain unvalidated until the
Phase 4 gate runs on NVIDIA hardware.

A backend becomes supported only after its architecture exit gate runs on the
actual hardware and matches the CPU reference within the documented tolerance.
See `ARCHITECTURE.md` for the binding contract and phase criteria.
