# Kipp Model Support

This document describes Kipp v0.0.2.

"Supported" has a narrow meaning here: a checkpoint is supported only when
it has a pinned entry in the compiled-in registry (`src/kipp_checkpoints.h`,
mirrored by `tools/checkpoints.py`), its conversion and test vectors were
generated from that pinned revision, and every gate below has run green on
real hardware. A registry entry alone is necessary but not sufficient.

## The registry

All entries are dense Qwen3 decoders (Apache-2.0), BF16, 8 KV heads,
head width 128, vocabulary 151,936, RMS epsilon `1e-6`, converted to Kipp's
strict GGUF-v3 subset. Per-entry dimensions, context lengths, RoPE theta,
embedding tying, and stop tokens are recorded in the registry sources.

| Checkpoint id | Layers | Hidden | Variant | Tied head | Gate status |
|---|---|---|---|---|---|
| qwen3-0.6b-base | 28 | 1024 | base | yes | **validated (CPU + Metal M5 + CUDA A100)** |
| qwen3-0.6b | 28 | 1024 | instruct | yes | registry only |
| qwen3-1.7b-base | 28 | 2048 | base | yes | registry only |
| qwen3-1.7b | 28 | 2048 | instruct | yes | registry only |
| qwen3-4b-base | 36 | 2560 | base | yes | **validated (CPU + Metal M5 + CUDA A100)** |
| qwen3-4b | 36 | 2560 | instruct | yes | registry only |
| qwen3-4b-instruct-2507 | 36 | 2560 | instruct-2507 | yes | **validated (CPU + Metal M5 + CUDA A100)** |
| qwen3-4b-thinking-2507 | 36 | 2560 | thinking-2507 | yes | registry only |
| qwen3-8b-base | 36 | 4096 | base | no | **validated (CPU + CUDA A100)** — Metal blocked by buffer size (see below) |
| qwen3-8b | 36 | 4096 | instruct | no | registry only |
| qwen3-14b-base | 40 | 5120 | base | no | **validated (CPU + CUDA A100)** — Metal blocked by size |
| qwen3-14b | 40 | 5120 | instruct | no | registry only |
| qwen3-32b | 64 | 5120 | instruct | no | **validated (CPU + CUDA A100)** — Metal blocked by size |

"Registry only" means the entry exists and the loader will accept a
correctly converted artifact, but the gates have not run for it yet, so no
support claim is made.

**Metal single-buffer limit.** The Metal backend wraps the entire GGUF mmap
in one no-copy `MTLBuffer`, so a model is Metal-runnable on a given device
only if its artifact fits `MTLDevice.maxBufferLength`. On this project's
Apple M5 that cap is **13.32 GiB** (recommended working set 17.76 GiB), so
the 8B BF16 artifact (15.26 GiB) is rejected at model creation with a clear
error rather than a fallback. The 8B Q8_0 artifact fits and is validated on
Metal, while its BF16 artifact remains unsupported there. 14B and 32B BF16
exceed the cap by more and are validated on CUDA in the cloud; any Metal
claim for those sizes requires a separately gated quantized artifact.

## What the gates check

For each checkpoint: exact registry/tokenizer/metadata match; golden
tokenizer cases (including the four instruct-era special tokens, which
base checkpoints must BPE-split and instruct checkpoints must isolate);
exact reference token IDs; full-vocabulary logits versus an FP32
HF Transformers reference with identical argmax and NMSE at most `1e-5`
(Phase 1); cached incremental decode equal to no-cache recomputation with
identical argmax and NMSE at most `1e-6` at every position (Phase 2); and
Metal-versus-CPU with identical argmax and NMSE at most `1e-4` for prefill,
every checked decode position, truncate/resume, and batched-equals-isolated
evaluation (Phase 3, Apple M5).

## Validated results

### qwen3-4b-base (CPU + Metal, Apple M5)

The v2 registry refactor was gated byte-identical against v0.0.1 output:
identical top-10 logits at `%.9g` on CPU and Metal, and an identical
8-token greedy Metal decode. Phase 1 NMSE `3.79e-7`; Phase 2 NMSE `0`;
Phase 3 maximum NMSE `2.96e-8` (decode) / `1.41e-8` (batch suffix), all
with exact argmax. Sampling, truncation, batched evaluation, and the
19-test server suite pass unchanged.

### qwen3-0.6b-base (CPU + Metal, Apple M5)

The first non-4B checkpoint, exercising 16 query heads and an attention
width (2,048) wider than the hidden width (1,024). All 13 golden tokenizer
cases pass, including the think/tool special tokens. Phase 1 NMSE
`1.83e-5` with exact argmax (`1.14e-11` with the BF16 KV rounding disabled,
confirming the deviation is the storage contract, not the implementation);
Phase 2 NMSE `0` at every position; Phase 3 Metal maximum NMSE `5.9e-7`
with exact argmax across decode, prefill, truncate/resume, and batched
evaluation.

### qwen3-8b-base (CPU, Apple M5)

The first untied checkpoint, binding a separate `lm_head.weight` instead of
aliasing the token embedding. Phase 1 NMSE `3.63e-7` with exact argmax and
Phase 2 NMSE `0` on CPU. The Metal gate is blocked, not failed: the 15.26
GiB artifact exceeds the M5's 13.32 GiB single-buffer cap and is rejected at
model creation. Metal validation is deferred to the quantized artifact.

### qwen3-4b-instruct-2507 (CPU + Metal, Apple M5)

The first instruct checkpoint, exercising RoPE theta `5e6`, the 262,144
context limit, and the instruct stop set. All 13 golden tokenizer cases
pass with the think/tool strings isolated as single tokens. Phase 1 NMSE
`7.31e-6` with exact argmax; Phase 2 NMSE `0` at every position; Phase 3
Metal maximum NMSE `5.1e-8` with exact argmax. A ChatML-format smoke decode
on Metal produces a coherent reply and generation stops on `<|im_end|>`
(151645) via the stop set rather than running to the token limit.

### Other checkpoints

Gate transcripts land here as each checkpoint's vectors are generated and
run. Until then the table above is the authority.

### CUDA (NVIDIA A100 80GB, first validation 2026-07-17)

The Phase 4 CUDA gate ran for the first time on a Verda A100-SXM4-80GB
(driver 580.126.09, CUDA 12.8, sm_80), built warning-clean with the
existing `CUDA_GENERIC_ARCH_FLAGS`. All four available checkpoints pass
`--phase4-cuda` with identical argmax and CUDA-vs-CPU full-logit NMSE well
under the `1e-4` bound at prefill, every decode position, stateless prefill,
and reset/reuse:

| Checkpoint | max NMSE vs CPU | argmax |
|---|---|---|
| qwen3-4b-base | 1.78e-8 | exact |
| qwen3-0.6b-base | 5.86e-7 | exact |
| qwen3-4b-instruct-2507 | 2.52e-7 | exact |
| qwen3-8b-base | 4.98e-8 | exact |
| qwen3-14b-base | 2.82e-6 (CUDA) | exact |
| qwen3-32b | 1.10e-7 (CUDA) | exact |

14B and 32B also pass CPU Phase 1 (argmax exact) and Phase 2 (NMSE `0`). 14B
uses the FP32 CPU reference (Phase 1 NMSE `7.8e-6`, under the `5e-5` bound).
32B is too large for an FP32 host reference (~256 GiB), so its reference is
BF16-on-GPU; that reference differs from Kipp's BF16-weight/FP32-accumulate
path by more, so the full-logit bound for a BF16 reference is recorded as
`2e-3` in the vector's `nmse-max.txt` (32B measured `7.3e-4`, argmax exact).
The native Phase 1 gate reads this per-vector bound rather than hardcoding
one. Both BF16 artifacts exceed the M5 Metal buffer cap, so their Metal
claims wait on quantization.

Validated checkpoints now span the registry's 0.6B through 32B range on at
least CPU plus one GPU backend. Entries marked “registry only” remain
unvalidated.

CUDA copies all weights resident into VRAM at model creation, so it is not
subject to Metal's single-buffer cap — this is how 8B (which Metal cannot
map on the M5) is validated. The gate ran against the exact same pinned
golden vectors used locally. `tools/ops/verda_cuda_gate.sh` captures the
disposable Verda workflow; it requires an explicit cost acknowledgement and
deletes both the VM and its OS volume on exit.

## Quantization (Q8_0)

Kipp converts the seven per-layer projection tensors to **Q8_0** (32-weight
blocks: fp16 scale + int8 quants, 8.5 bpw); token embedding, lm_head, and all
norms stay BF16. `tools/convert_to_gguf.py --quant q8_0` writes a
`kipp-<id>-q8_0.gguf` artifact with a `kipp.quant.scheme` metadata key; the
loader validates per-tensor types and byte counts. Both CPU and Metal decode
Q8_0 (the Metal path adds token-tiled `kipp_matvec_q8_0` decode/prefill
kernels).

Validated on Apple M5:

| Checkpoint | artifact | CPU NMSE vs bf16 ref | Metal vs CPU NMSE | 4B decode |
|---|---|---|---|---|
| qwen3-4b-base q8_0 | 4.32 GiB (was 7.5) | 1.9e-5 | 5.5e-10 | 22.2 tok/s (bf16 13.8) |
| qwen3-8b-base q8_0 | 9.20 GiB (was 15.26) | 1.6e-5 | 1.0e-10 | — |

Both pass with exact argmax. Q8_0 is near-lossless (NMSE under the 5e-5 bf16
bound) and ~1.6× faster decode / ~2× faster prefill on the 4B. Critically,
**8B now runs on Metal** — its 9.20 GiB Q8_0 artifact fits the 13.32 GiB
single-buffer cap that its 15.26 GiB BF16 artifact exceeded. 14B (Q8 ≈ 15
GiB) still exceeds the cap and awaits 4-bit; 8B/14B/32B Q8_0 remain CUDA- and
CPU-runnable.

## Quantization (AFFINE4_GS32)

Kipp also implements a private 4-bit affine format with 32-weight groups:
16 packed nibbles plus fp16 scale and bias (5.0 bits per weight). CPU and
Metal have dedicated decode paths, and the converter emits the format with
`--quant affine4_gs32`. Unlike Q8_0, this is a Q4-class lossy format; model
gates therefore use an explicit per-vector tolerance rather than presenting
it as near-lossless. A checkpoint gains a quantized support claim only after
its full-logit and backend gates pass.

## Backends

The CPU oracle, the Metal backend (Apple M5), and the CUDA backend (NVIDIA
A100) are validated backends. Metal must be selected explicitly — a Metal
initialization or shader compilation failure is fatal rather than falling
back. Metal compiles its pipelines per model, specializing the hidden width
and query-head count through function constants; the mmap of each GGUF is
wrapped in one shared no-copy buffer (hence the single-buffer cap). CUDA
copies the required tensors to VRAM at model creation and fails explicitly
if they do not fit.

CUDA is validated on the checkpoints in the table above; other registry
entries stay "registry only" for CUDA until their gate runs.

BF16, Q8_0, and AFFINE4_GS32 are the only accepted weight schemes. No MoE,
vision-language, Qwen3-Next, Qwen3.5, or arbitrary GGUF checkpoint is
supported; those require a new architecture review, not a registry entry.

See `ARCHITECTURE.md` for the binding contract and phase criteria.
