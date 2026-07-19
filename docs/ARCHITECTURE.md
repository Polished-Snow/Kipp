# Kipp Architecture

This document is the implementation contract for the Kipp v2 architecture
generation. Changes to the supported-model registry, model format, backend
boundary, KV layout, or API surface require an explicit architecture review
before code is changed.

Revision note: the v1 contract (release v0.0.1) supported exactly one
checkpoint, `Qwen/Qwen3-4B-Base`. On 2026-07-16 an approved architecture
review expanded scope to the pinned Qwen3 dense family below. Everything
else in the v1 contract — strict validation, correctness gating, backend
isolation — carries forward unchanged.

## Scope and invariants

Kipp v2 is a hand-written inference engine for a fixed registry of pinned
Qwen3 dense checkpoints. It is not a general Qwen runner, a generic GGUF
runner, or a tensor framework: a checkpoint outside the registry is a
rejection, not an extension point.

- C11 owns model validation, tokenization, lifecycle, sampling, and the CPU
  reference path.
- Objective-C is confined to the Metal bridge. Metal kernels and CUDA kernels
  remain isolated from one another.
- Python and shell are tooling only. They may download, convert, and generate
  reference vectors, but they are never part of the inference process.
- Correctness is established on CPU before a GPU backend is optimized.
- A backend is not called working until it has run on that backend's hardware.
- No backend silently falls back to another backend.

## Supported checkpoints

### The registry

`src/kipp_checkpoints.h` compiles in the full list of supported checkpoints,
each pinned to one immutable Hugging Face revision with its exact
dimensions, context length, RoPE theta, embedding tying, variant, and stop
tokens. `tools/checkpoints.py` mirrors the table for the tooling, and
`tests/test_tooling.py` cross-checks the two copies. Adding a checkpoint
means adding one registry entry (both copies), converting, generating
vectors, and running the gates — never changing engine code.

The registry currently holds the Qwen3 dense family (all Apache-2.0):
base and instruct 0.6B, 1.7B, 4B, 8B, and 14B; instruct-only 32B
(Qwen3-32B-Base was never released); and the 4B Instruct/Thinking 2507
refreshes. The exact pinned revisions live only in the registry sources.

Family-invariant dimensions, validated for every entry:

- Key/value heads: 8
- Head width: 128
- Vocabulary: 151,936
- Activation: SwiGLU using SiLU
- Normalization: RMSNorm with epsilon `1e-6`, plus per-head Q/K RMSNorm
- Position encoding: standard NEOX-style RoPE; no YaRN or other RoPE scaling
- Attention: causal grouped-query attention, without sliding-window attention
- Weight dtype: BF16; linear layers have no biases

Per-checkpoint dimensions, validated against the matching registry entry:

- Layers (28–64), hidden width (1,024–5,120), feed-forward width
  (3,072–25,600), and query-head count (16–64)
- Context length: 32,768 (base), 40,960 (instruct), or 262,144 (4B-2507)
- RoPE theta: `1,000,000`, or `5,000,000` for the 2507 checkpoints
- Embedding tying: 0.6B/1.7B/4B tie the output head to the token embedding;
  8B/14B/32B carry a separate `lm_head.weight` tensor
- Stop tokens: base checkpoints stop on `<|endoftext|>` (151643); instruct
  checkpoints stop on `<|im_end|>` (151645) and `<|endoftext|>`

Two derived facts the code must never conflate: the attention width is
`query_heads * 128` and is NOT the hidden width (they differ at 0.6B and
32B), and the forward pass is otherwise identical at every size.

Qwen3 MoE, Qwen3-Next, Qwen3-VL, and Qwen3.5 remain separate model-support
decisions requiring a new architecture review.

Each layer performs these operations in order:

1. RMSNorm the residual stream.
2. Project Q, K, and V.
3. Reshape into heads and apply per-head RMSNorm to Q and K.
4. Apply RoPE to Q and K.
5. Run causal grouped-query attention, with `query_heads / 8` query heads
   sharing each KV head.
6. Apply the attention output projection and add the residual.
7. RMSNorm the new residual stream.
8. Compute `down_proj(SiLU(gate_proj(x)) * up_proj(x))`.
9. Add the feed-forward residual.

After the final layer, Kipp applies the final RMSNorm and multiplies by the
output head — the transposed token embedding when tied, the checkpoint's
`lm_head.weight` otherwise — to produce 151,936 FP32 logits.

### Tokenizer

Kipp owns a native, Qwen-specific implementation of the
`Qwen2Tokenizer` byte-level BPE used by every supported checkpoint. It loads
the vocabulary, merge ranks, special tokens, and pre-tokenization data
embedded in Kipp's GGUF file. The vocabulary, merges, pre-tokenizer, and
NFC normalization are byte-identical across the whole family, and so is the
special-token set: every checkpoint — base included — declares the
think/tool tokens (`<tool_response>` 151665, `</tool_response>` 151666,
`<think>` 151667, `</think>` 151668) in `tokenizer_config.json`'s
`added_tokens_decoder`, which upstream merges over `tokenizer.json`. The
converter performs the same merge; skipping it would silently mis-tokenize
those strings, which the golden tokenizer cases now cover explicitly.

- No BOS token is inserted automatically.
- Token 151643 (`<|endoftext|>`) is EOS and padding for base checkpoints;
  instruct checkpoints emit `<|im_end|>` (151645) at end of turn, and both
  belong to their stop set.
- Token 151644 is `<|im_start|>` and token 151645 is `<|im_end|>`.
- UTF-8, byte fallback, special-token recognition, and multilingual
  pre-tokenization require deterministic native tests.
- Chat-template rendering is server-side work; the engine API stays
  token/text-based.

The tokenizer is model-specific. Kipp will not implement a registry of
tokenizer types.

## Model artifact and loader

### Conversion boundary

`tools/download_model.sh --checkpoint ID` downloads the registry entry's
pinned Hugging Face revision and verifies recorded checksums.
`tools/convert_to_gguf.py --checkpoint ID` converts the BF16 safetensors
(sharded or single-file) into one mmap-friendly GGUF artifact per
checkpoint. Tooling may use Python ML packages to generate independent
reference data, but the converter must not make the native runtime depend on
them.

The converter writes only the metadata and tensor types required by the
registry entry. It records:

- source repository and immutable revision;
- source-file and output-file SHA-256 hashes;
- exact model dimensions and tokenizer metadata;
- BF16 tensor offsets, dimensions, and aligned byte lengths; and
- the embedding-tying relationship — no duplicate output tensor when tied,
  the checkpoint's own `lm_head.weight` when untied.

### Strict GGUF subset

The native loader parses the GGUF header, metadata, tensor directory, and
tokenizer data. It matches `(kipp.source.repository, kipp.source.revision)`
against the compiled-in registry, requires every metadata field to equal the
matched entry, and then binds an explicit `kipp_qwen3_weights` structure:

- token embedding, final RMSNorm, and the output head (an alias of the
  token embedding when tied, a bound `lm_head.weight` otherwise);
- for each of the entry's layers, attention RMSNorm, Q/K/V/output
  projections, Q/K per-head RMSNorm, feed-forward RMSNorm, and gate/up/down
  projections.

The loader rejects the file before inference if the checkpoint is not in the
registry or if the dimensions, dtype, tensor names, tensor shapes, tensor
count, alignment, offsets, or required metadata differ from the matched
entry. Unknown tensors and architectures are errors, not extension points.

No code from llama.cpp or ggml is vendored. GGUF is implemented from its
published format and Kipp's deliberately small required subset.

### Weight quantization

An artifact may quantize the seven per-layer projection tensors while
keeping the token embedding, lm_head, and all norms in BF16. A
`kipp.quant.scheme` metadata string (`bf16`, `q8_0`, or `affine4_gs32`)
records the scheme, and the tensor directory carries the per-tensor type;
the loader validates both. Two formats, both operating on 32-weight blocks
with `columns % 32 == 0` (true for every family dimension):

- **Q8_0** — `{fp16 scale; int8 qs[32]}`, 34 bytes/block (8.5 bpw),
  `w = scale·q`. Near-lossless (full-logit NMSE under the BF16 gate bound).
- **AFFINE4_GS32** — `{16 packed nibbles; fp16 scale; fp16 bias}`, 20
  bytes/group (5.0 bpw), `w = scale·q + bias`, `q ∈ [0,15]`. Q4-class.

The CPU reference dequantizes per block inside a type-dispatched matvec;
Metal adds token-tiled `kipp_matvec_q8_0`/`kipp_matvec_affine4` decode and
prefill kernels selected by the weight tensor's type. Quantized artifacts
gate as ordinary artifacts: argmax must match the BF16 reference exactly,
with a format-appropriate full-logit NMSE bound (Q8_0 near-lossless; 4-bit
Q4-class), and Metal must match CPU within the standard `1e-4`. The
converter quantizes against the fp16-rounded scale so encoder and decoder
agree exactly.

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

A session owns separate K and V BF16 stores with logical layout:

```text
[layer][position][kv_head][head_dim]
```

KV heads (8) and head width (128) are family constants; the layer count is
the checkpoint's. Each cached token therefore costs
`layers * 2(K,V) * 8 * 128 * 2(BF16)` bytes across all layers — 147,456
bytes for the 36-layer 4B, 114,688 for 28-layer sizes, 262,144 for the
64-layer 32B. That per-token figure is the reported logical KV size and is
unchanged by paging.

The caller selects session capacity at creation, up to the checkpoint's
context length. For the 4B an 8,192-token cache is 1.125 GiB and a full
32,768-token cache is 4.5 GiB. Capacity is allocated once, checked on every
append, and released with the session. Reset zeroes logical length; it need
not clear unreachable bytes.

Both the CPU oracle and the Metal backend address their KV through a page
table (Phase 5). Physical storage is a run of fixed 32-position blocks, and
a per-session `block_table` maps each logical block (`position >> 5`) to the
physical block that holds it; the physical store is rounded up to a whole
number of blocks. On CPU this is a table indexed inside `kv_cache_offset`;
on Metal the `kipp_kv_write` and `kipp_flash_gqa` kernels take the block
table as a buffer and resolve `block_table[pos >> 5] * 32 + (pos & 31)` per
access. With the identity table this is byte-for-byte the original
contiguous layout — the `--model` and `--phase3-metal` gates still reproduce
the reference — and a non-identity table relocates positions without
changing any computed value. The `--paged-cpu` and `--paged-metal` gates
prove this per backend by evaluating a multi-block prompt under a
deliberately scrambled block table and asserting the final logits are
bitwise-identical to the identity mapping; `--phase3-metal` further confirms
paged Metal matches the paged CPU oracle within the 1e-4 tolerance. CUDA
remains contiguous for now. The shared cross-request block pool
(`src/kipp_kv_pool.c`) and the eval-item `block_table` contract that lets
the core drive both backends are the next Phase 5 steps.

Sessions support truncation back to a prefix of their timeline
(`kipp_session_truncate`), which is the primitive behind the server's
serial prefix reuse. Cross-session prefix sharing, persistence, and
eviction through the block pool remain future Phase 5 work and do not change
the backend contract.

### SSD streaming

SSD streaming is not part of Kipp v1.

ds4 keeps model loading mmap-backed and uses backend-specific placement:
`model_open()` maps weights, Metal uses `ds4_gpu_set_model_map()`, and CUDA
uses `ds4_gpu_set_model_map_range()`. Its `ds4_ssd_*` path streams sparsely
selected routed-expert weights while other expert work can hide I/O.

Qwen3 dense checkpoints need every layer and every feed-forward weight for
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
writes logits for the last `logits_count` tokens (1 = only the final token,
the common case; the public `kipp_session_eval_n` exposes the multi-row
form). Multi-row logits are the primitive behind speculative-decode
verification and prompt logprobs; the CPU oracle writes them bitwise-equal
to per-position evaluation, and Metal within the standard `1e-4`. CUDA and
stateless evaluation currently support only the final row. Phase 1 calls it with one
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

`src/kipp_server.c` exposes a deliberately small OpenAI-compatible surface
on loopback, with ordinary JSON responses and optional SSE streaming:

- `GET /healthz`
- `GET /metrics`
- `GET /v1/models`
- `POST /v1/completions`
- `POST /v1/chat/completions`

The completion request supports one string prompt, up to eight completions
(`n`), a bounded `max_tokens`, and the full sampling surface — `temperature`,
`top_p`, `top_k`, `min_p`, `frequency_penalty`, `presence_penalty`,
`repetition_penalty`, `logit_bias`, and `seed` — plus stop strings and an
optional `stream` flag that switches the response to server-sent events in
the OpenAI chunk shape (terminated by `data: [DONE]`; each chunk carries its
choice index). Generated-token log-probabilities are available (legacy
`logprobs` = top-N in [0,5]; chat `logprobs`/`top_logprobs` in [0,20]),
computed as a numerically-stable log-softmax of the raw model logits at
sample time and emitted in the endpoint's native shape — streamed per chunk
or attached to the final response. `stream_options.include_usage` adds a
trailing usage-only chunk, and every non-streamed response (plus that final
chunk) carries a llama.cpp-style `timings` object: prompt/predicted token
counts, milliseconds, and tokens-per-second from a monotonic clock.

Chat completions are available for instruct checkpoints. Messages are rendered
through the native Qwen3 ChatML renderer (`src/kipp_chat.c`) with
`chat_template_kwargs.enable_thinking` support; base checkpoints reject chat
requests since they have no template. Penalties apply over a 64-token
per-choice window, and sampling uses the single `kipp_sample_ex` core entry
point shared by the CLI and server.

Multiple choices decode together through `kipp_eval_batch`, so each step reads
the weights once for all choices. Streamed text is held back while it could
still grow into a stop string, because emitted bytes cannot be recalled. The
non-streaming response reports the model identifier, generated text, finish
reason, and prompt/completion token counts. Unsupported fields receive a clear
client error rather than being ignored.

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
Multi-entry prefix caching and production KV block pooling remain future
reviewed work and must not change this API surface.

Greedy self-speculative decoding is available (CLI `--spec`): each step
drafts the continuation with a prompt-lookup n-gram matcher
(`src/kipp_spec.c`), verifies the whole draft in one multi-logit forward,
and accepts the longest greedy-matching prefix — rolling the KV back with
`kipp_session_truncate` for rejected drafts. The emitted sequence is exactly
the plain greedy sequence (a gated invariant); it just takes fewer forward
passes (measured ~1.9x on repetitive/structured content, multiplicative with
quantization).

The v1 server does not claim support for Anthropic Messages, tools,
structured output, embeddings, multimodal input, authentication, TLS,
distributed execution, or multi-model serving. Chat APIs require an explicitly approved instruction
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
5. The readable full-prompt forward pass and output projection.

Tooling captures immutable token IDs, selected intermediate tensors, and full
FP32 logits from each pinned checkpoint. CPU completion requires exact metadata
and token IDs, identical final argmax, and full-logit NMSE no greater than
`5e-5` against BF16-weight/FP32-accumulation reference vectors. The test
manifest records commands, tool versions, hashes, and tolerances.

The `5e-5` bound is a documented recalibration of v1's `1e-5`: the FP32
reference does not round K/V through BF16, while Kipp's KV storage contract
does, and smaller checkpoints feel that rounding more (measured NMSE
`3.8e-7` at 4B and `1.8e-5` at 0.6B; both fall below `1e-10` when the
rounding is disabled, demonstrating the implementation itself is exact).

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

### Phase 7 — Explicitly reviewed extensions

The checkpoint registry, instruction/chat support, Q8_0 and affine 4-bit
weight quantization, multi-row logits, and prompt-lookup speculative decoding
were admitted through separate reviews and correctness gates.

Still deferred:

- cross-session prefix caching and radix attention;
- KV persistence, offload, or quantization;
- SSD weight streaming;
- additional model families;
- distributed inference; and
- ROCm/HIP, on a separate community-maintained branch.

Each deferred item requires a new plan, tests, and a measured reason to add
complexity.

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
