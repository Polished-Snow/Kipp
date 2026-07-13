# Kipp Architecture

This document is the implementation contract for the Kipp v1 architecture
generation (first shipped as release v0.0.1). Changes to the
target model, model format, backend boundary, KV layout, or API surface require
an explicit architecture review before code is changed.

## Scope and invariants

Kipp v1 is a hand-written inference engine for one exact model:
`Qwen/Qwen3-4B-Base`. It is not a general Qwen runner, a generic GGUF runner,
or a tensor framework.

- C11 owns model validation, tokenization, lifecycle, sampling, and the CPU
  reference path.
- Objective-C is confined to the Metal bridge. Metal kernels and CUDA kernels
  remain isolated from one another.
- Python and shell are tooling only. They may download, convert, and generate
  reference vectors, but they are never part of the inference process.
- Correctness is established on CPU before a GPU backend is optimized.
- A backend is not called working until it has run on that backend's hardware.
- No backend silently falls back to another backend.

## Target model

### Exact checkpoint

- Repository: `Qwen/Qwen3-4B-Base`
- Revision: `906bfd4b4dc7f14ee4320094d8b41684abff8539`
- Model license: Apache-2.0
- Architecture: dense, decoder-only causal transformer
- Parameters: 4.0B total, approximately 3.6B excluding embeddings
- Weight dtype: BF16
- Weight bytes in the pinned safetensors index: 8,045,591,552
- Context length: 32,768 tokens; no YaRN or other RoPE scaling in v1

The runtime accepts this architecture and size only. A different Qwen3 size,
an instruction-tuned checkpoint, Qwen3 MoE, Qwen3-Next, Qwen3-VL, or Qwen3.5
is a separate model-support decision.

### Model dimensions

- Layers: 36
- Hidden width: 2,560
- Feed-forward width: 9,728
- Query heads: 32
- Key/value heads: 8
- Head width: 128
- Vocabulary: 151,936
- Activation: SwiGLU using SiLU
- Normalization: RMSNorm with epsilon `1e-6`
- Position encoding: standard RoPE with theta `1,000,000`
- Attention: causal grouped-query attention, without sliding-window attention
- Embeddings: input and output weights are tied
- Linear layers: no biases

Each layer performs these operations in order:

1. RMSNorm the residual stream.
2. Project Q, K, and V.
3. Reshape into heads and apply per-head RMSNorm to Q and K.
4. Apply RoPE to Q and K.
5. Run causal grouped-query attention, with four query heads sharing each KV
   head.
6. Apply the attention output projection and add the residual.
7. RMSNorm the new residual stream.
8. Compute `down_proj(SiLU(gate_proj(x)) * up_proj(x))`.
9. Add the feed-forward residual.

After layer 35, Kipp applies the final RMSNorm and multiplies by the transposed
tied token-embedding matrix to produce 151,936 FP32 logits.

### Tokenizer

Kipp owns a native, Qwen-specific implementation of the
`Qwen2Tokenizer` byte-level BPE used by the pinned checkpoint. It loads the
vocabulary, merge ranks, special tokens, and pre-tokenization data embedded in
Kipp's GGUF file.

- No BOS token is inserted automatically.
- Token 151643 (`<|endoftext|>`) is EOS and padding.
- Token 151644 is `<|im_start|>` and token 151645 is `<|im_end|>`.
- UTF-8, byte fallback, special-token recognition, and multilingual
  pre-tokenization require deterministic native tests.
- The base model does not make chat-template behavior part of the v1 API.

The tokenizer is model-specific. Kipp will not implement a registry of
tokenizer types.

## Model artifact and loader

### Conversion boundary

`tools/download_model.sh` will download the pinned Hugging Face revision and
verify recorded checksums. `tools/convert_to_gguf.py` will convert the three
BF16 safetensors shards into one mmap-friendly GGUF artifact. Tooling may use
Python ML packages to generate independent reference data, but the converter
must not make the native runtime depend on them.

The converter writes only the metadata and tensor types required by
Qwen3-4B-Base. It records:

- source repository and immutable revision;
- source-file and output-file SHA-256 hashes;
- exact model dimensions and tokenizer metadata;
- BF16 tensor offsets, dimensions, and aligned byte lengths; and
- the tied-embedding relationship, with no duplicate output tensor.

### Strict GGUF subset

The native loader parses the GGUF header, metadata, tensor directory, and
tokenizer data. It then binds an explicit `kipp_qwen3_weights` structure:

- token embedding and final RMSNorm;
- for each of 36 layers, attention RMSNorm, Q/K/V/output projections, Q/K
  per-head RMSNorm, feed-forward RMSNorm, and gate/up/down projections.

The loader rejects the file before inference if the architecture, revision,
dimensions, dtype, tensor names, tensor shapes, alignment, offsets, or required
metadata differ from the v1 contract. Unknown tensors and architectures are
errors, not extension points.

No code from llama.cpp or ggml is vendored. GGUF is implemented from its
published format and Kipp's deliberately small required subset.

## Memory model

### Weights

Each opened model has one read-only file mapping. Tensor views are validated
offsets into that mapping; the C loader does not eagerly copy weights.

- **CPU:** uses a private read-only mapping and converts BF16 values while
  computing, with FP32 activations and accumulation.
- **Metal:** uses a shared read-only mapping and wraps the complete mapping in
  no-copy Metal storage. Tensor offsets remain offsets into that one buffer.
- **CUDA:** validates the host mapping first, then copies all required tensors
  to VRAM during model creation. If weights, session state, and scratch space
  do not fit, model creation fails explicitly.

Partial GPU offload, automatic placement, and backend fallback are out of
scope.

### Activations and logits

Backends own reusable activation and scratch storage. Allocation occurs when a
backend model or session is created, not inside the layer loop. Intermediate
activations use FP32 for the CPU reference path. GPU storage and accumulation
precision may differ only when tests establish the documented tolerance.

Only final FP32 logits cross from a GPU backend to host memory during normal
evaluation.

### KV cache

Phase 2 starts with a contiguous, non-paged cache. A session owns separate K
and V BF16 slabs with logical layout:

```text
[layer][position][kv_head][head_dim]
```

The fixed dimensions are 36 layers, 8 KV heads, and 128 values per head. Each
cached token therefore costs 147,456 bytes across all layers:

```text
36 * 2(K,V) * 8 * 128 * 2(BF16) = 147,456 bytes
```

The caller selects session capacity at creation, up to 32,768 tokens. An
8,192-token cache is 1.125 GiB; a full 32,768-token cache is 4.5 GiB. Capacity
is allocated once, checked on every append, and released with the session.
Reset zeroes logical length; it need not clear unreachable bytes.

Sessions support truncation back to a prefix of their timeline
(`kipp_session_truncate`), which is the primitive behind the server's
serial prefix reuse. Phase 2 includes no blocks, paging, cross-session
prefix sharing, persistence, or eviction. Phase 5 may replace the internal
slabs with blocks without changing the backend contract.

### SSD streaming

SSD streaming is not part of Kipp v1.

ds4 keeps model loading mmap-backed and uses backend-specific placement:
`model_open()` maps weights, Metal uses `ds4_gpu_set_model_map()`, and CUDA
uses `ds4_gpu_set_model_map_range()`. Its `ds4_ssd_*` path streams sparsely
selected routed-expert weights while other expert work can hide I/O.

Qwen3-4B is dense: every layer and every feed-forward weight is needed for
every token. Streaming dense layers would reread nearly the entire model on
each decode step, unlike ds4's sparse-expert case. Kipp therefore relies on
the operating system's page cache for mapped CPU/Metal weights and requires
resident CUDA weights. Dense layer streaming may be reconsidered only after
measurement on a larger future checkpoint. Live KV remains memory-resident;
KV checkpoint files are also deferred.

## Backend boundary

Kipp uses one internal, model-specific backend contract. It is not a generic
tensor API and does not expose device buffers to callers.

The contract will live in `src/kipp_backend.h` and have this shape:

```c
typedef struct {
    void *session;
    const uint32_t *tokens;
    uint32_t token_count;
    uint32_t start_position;
    float *logits;
} kipp_eval_item;

typedef struct {
    int  (*model_create)(const kipp_model_view *model, void **backend_model);
    void (*model_destroy)(void *backend_model);
    int  (*session_create)(void *backend_model, uint32_t capacity,
                           void **backend_session);
    void (*session_destroy)(void *backend_session);
    int  (*eval)(void *backend_model, kipp_eval_item *items,
                 size_t item_count);
} kipp_backend_ops;
```

The final declarations may add error-output parameters and const qualifiers,
but they must preserve these ownership and execution semantics.

`kipp_model_view` contains the validated mmap, fixed model dimensions, and
explicit semantic weight views. Core C owns model parsing, tokenization,
sampling, public lifecycle, and error reporting. A backend owns its prepared
weight representation, activation scratch, KV storage, and Qwen3 execution.

`eval` is synchronous. Each item appends `token_count` tokens at
`start_position`, which must equal that session's logical KV length, and
writes the logits for the final supplied token. Phase 1 calls it with one
item and no reused cache. Phase 2 activates session KV state. Multi-item
calls are live through the public `kipp_eval_batch`: backends may interleave
the items so weights are read once per step across sequences (the Metal
backend packs chunks of every item into shared rounds), and batched results
must match isolated evaluation within the backend's documented tolerance.

The CPU implementation lives with the readable reference forward pass in
`src/kipp.c`. `src/metal/kipp_metal.m` and `src/cuda/kipp_cuda.cu` each expose
their own static `kipp_backend_ops`. Device-specific headers, handles, and
errors do not cross this interface.

## Public engine and session model

The public C API will use opaque model and session handles:

- a model owns the validated file mapping and one selected backend model;
- a session owns one independent token timeline and KV cache;
- model objects are immutable after creation;
- sessions are not concurrently mutable;
- destroying a model requires all of its sessions to be destroyed first.

The engine accepts token IDs or UTF-8 text, but both paths converge before
backend evaluation. Sampling consumes host FP32 logits. Backend selection is
explicit at model creation and never changes for that model.

## Server and API surface

`src/kipp_server.c` exposes a deliberately small, non-streaming subset of
the OpenAI Completions API on loopback:

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/completions`

The completion request supports one string prompt, up to eight completions
(`n`), a bounded `max_tokens`, `temperature`, `top_p`, `seed`, stop strings,
and an optional `stream` flag that switches the response to server-sent
events in the OpenAI chunk shape (terminated by `data: [DONE]`; each chunk
carries its choice index). Multiple choices decode together through
`kipp_eval_batch`, so each step reads the weights once for all choices. Streamed text is held
back while it could still grow into a stop string, because emitted bytes
cannot be recalled. The non-streaming response reports the model identifier,
generated text, finish reason, and prompt/completion token counts.
Unsupported fields receive a clear client error rather than being ignored.

The server runs a single-threaded poll() event loop over concurrent
loopback connections. Parsed completion requests join a FIFO and are
admitted while the total choice count fits the batch limit; each scheduler
step evaluates one prompt chunk (chunked prefill, 32 tokens) or one decode
token per active choice through `kipp_eval_batch`, so concurrent requests
read the weights once per step. Slow or disconnected clients are cancelled
without disturbing other requests. A single cached session plus its
evaluated token timeline provides serial prefix reuse: when an idle
single-choice request shares a prefix with the timeline, the session is
rolled back with `kipp_session_truncate` and only the suffix is prefilled.
Multi-entry prefix caching and KV blocks remain future Phase 5 work and
must not change this API surface.

The v1 server does not claim support for Chat Completions, Anthropic
Messages, tools, structured output, embeddings, multimodal input,
authentication, TLS, distributed execution, speculative decoding, or
multi-model serving. Chat APIs require an explicitly approved instruction
checkpoint and are not emulated with the base model.

HTTP parsing, compatibility translation, scheduling, and model execution
remain separate layers. The server cannot access backend buffers or KV memory
directly.

## Build phases and exit conditions

Every phase ends with test results and a review stop. A later phase does not
begin until the previous phase is approved.

### Phase 1 — CPU reference path

Build in this order:

1. Pinned download, strict conversion, manifest, and test-vector tooling.
2. GGUF metadata/tensor validation and mmap-backed semantic weight binding.
3. Native Qwen tokenizer with byte-level, Unicode, special-token, encode, and
   decode tests.
4. Scalar BF16 conversion, RMSNorm, matmul, RoPE, Q/K norm, softmax, GQA,
   SwiGLU, and residual unit tests.
5. The readable 36-layer full-prompt forward pass and tied output projection.

Tooling captures immutable token IDs, selected intermediate tensors, and full
FP32 logits from the pinned checkpoint. CPU completion requires exact metadata
and token IDs, identical final argmax, and full-logit NMSE no greater than
`1e-5` against BF16-weight/FP32-accumulation reference vectors. The test
manifest records commands, tool versions, hashes, and tolerances.

Exit condition: a fixed UTF-8 prompt is converted to the expected tokens and a
single native CPU forward pass produces accepted full-vocabulary logits.

### Phase 2 — KV cache

Add session allocation, reset, bounded append, full-prompt prefill, and
one-token incremental decode. The no-cache CPU path rounds K/V to the same BF16
storage contract so it remains a valid oracle.

Tests cover zero/one/many tokens, exact capacity, overflow rejection, reset,
and multiple independent sessions. Cached decode must preserve the final
argmax and have NMSE no greater than `1e-6` against CPU recomputation at every
checked position.

Exit condition: incremental decoding matches no-cache recomputation for the
fixed vector and cache lifecycle tests pass.

### Phase 3 — Metal backend

Implement no-copy mapped weights, persistent scratch/KV buffers, and Qwen3
kernels on the local Apple M5. Optimize only after operator-level comparisons
pass.

Bitwise equality is not required because parallel reductions change floating
point operation order. The accepted contract is identical argmax and
full-logit NMSE no greater than `1e-4` versus CPU for prefill and every checked
decode position. Any wider tolerance requires a documented plan change.

Exit condition: the Metal backend passes those vectors on the M5 and reports
measured memory, prefill throughput, and decode throughput.

### Phase 4 — CUDA backend

Implement the same backend contract with resident VRAM weights and isolated
CUDA kernels. Do not change Metal code to accommodate CUDA.

Exit condition: on the provided NVIDIA machine, CUDA preserves CPU argmax and
full-logit NMSE no greater than `1e-4` for prefill and decode, with memory and
throughput reported. If CUDA hardware is unavailable, this phase stops without
a success claim.

### Phase 5 — Continuous batching and scheduler

Only after one GPU backend is correct, introduce KV blocks, allocation, and a
small FIFO scheduler. Keep request state separate from execution and the HTTP
layer. Prefix hashing and radix trees remain deferred.

Exit condition: mixed-length concurrent requests reproduce their isolated
backend outputs within that backend's existing tolerance, including cache
exhaustion, cancellation, and block-reuse tests.

### Phase 6 — Server

Implement only the approved OpenAI Completions subset around the tested
scheduler and engine.

Exit condition: endpoint schema, unsupported-field, context-limit, overload,
disconnect, deterministic-seed, and end-to-end completion tests pass without
the HTTP layer touching backend state.

### Phase 7 — Explicitly deferred work

- model quantization;
- prefix caching and radix attention;
- KV persistence or offload;
- SSD weight streaming;
- speculative or multi-token decoding;
- instruction/chat checkpoints and APIs;
- additional model sizes or families;
- distributed inference; and
- ROCm/HIP, on a separate community-maintained branch.

Each item requires a new plan, tests, and a measured reason to add complexity.

## Research provenance

The references below were read for architecture concepts only. Kipp does not
vendor their code or translate functions near-verbatim.

- `antirez/ds4` at `80ebbc396aee40eedc1d829222f3362d10fa4c6c`
  (MIT): `ds4.c` (`model_open`, weight binding, session state),
  `ds4_gpu.h` (model-map and device-tensor boundary), `ds4_ssd.c`,
  `ds4_ssd.h`, and `ds4_server.c`.
- `ggml-org/llama.cpp` at
  `049326a00025d00b08cc188ed716b681e984a3f8` (MIT):
  `conversion/qwen.py`, `gguf-py/gguf/tensor_mapping.py`,
  `src/llama-model-loader.*`, `src/llama-mmap.*`, `src/llama-vocab.*`, and
  `tests/test-llama-archs.cpp`.
- `ggml-org/ggml` at `2faff76703c6b3a8ba9ebf92e2bb1bc93b7b1b1a`
  (MIT): `docs/gguf.md`, `src/gguf.cpp`, and
  `src/ggml-backend-impl.h`.
- `ggml-org/whisper.cpp` at
  `6fc7c33b4c3a2cec83e4b65abd5e96a890480375` (MIT):
  `src/whisper.cpp` and `src/whisper-arch.h`.
- `karpathy/llm.c` at `f1e2ace651495b74ae22d45d1723443fd00ecd3a`
  (MIT): `train_gpt2.c`, `llmc/tokenizer.h`, and `dev/test/`.
- `GeeeekExplorer/nano-vllm` at
  `bb823b3e06983d71485a8e1f23715ebd87d98ef8` (MIT):
  `nanovllm/engine/block_manager.py`, `sequence.py`, `scheduler.py`, and
  `llm_engine.py`.
- `vllm-project/vllm` at
  `e12b91b032daed2afc34d77cca20902cef957b3c` (Apache-2.0):
  `vllm/v1/request.py`, `vllm/v1/core/kv_cache_manager.py`, and
  `vllm/v1/core/sched/scheduler.py`.
- `sgl-project/sglang` at
  `b86466d54b2ff3f7d1635fd9a856a95ab3dba9b6` (Apache-2.0):
  `python/sglang/srt/managers/scheduler.py`,
  `python/sglang/srt/managers/schedule_batch.py`, and
  `python/sglang/srt/mem_cache/`.

The pinned Qwen model's `config.json`, `tokenizer_config.json`, safetensors
index, README, and Apache-2.0 license are the authority for model-specific
facts. Reference repositories inform boundaries and testing strategy, not
Kipp's implementation text.
