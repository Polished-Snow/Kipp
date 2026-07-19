#!/usr/bin/env python3
"""Stream a pinned Qwen3 snapshot into Kipp's strict GGUF subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import struct
import sys
from dataclasses import dataclass
from typing import Any, BinaryIO, Iterable

import checkpoints

ALIGNMENT = 32
GGUF_VERSION = 3
GGML_TYPE_BF16 = 30
GGML_TYPE_Q8_0 = 8
GGML_TYPE_AFFINE4_GS32 = 1000  # Kipp-private id
QUANT_BLOCK = 32

# The seven per-layer projection tensors carry the model's quant scheme;
# token embedding, lm_head, and every norm stay BF16.
PROJECTION_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)

QUANT_TYPE_ID = {
    "bf16": GGML_TYPE_BF16,
    "q8_0": GGML_TYPE_Q8_0,
    "affine4_gs32": GGML_TYPE_AFFINE4_GS32,
}


def is_projection(name: str) -> bool:
    return name.endswith(PROJECTION_SUFFIXES)


def type_byte_count(type_id: int, element_count: int) -> int:
    if type_id == GGML_TYPE_BF16:
        return element_count * 2
    if element_count % QUANT_BLOCK != 0:
        raise ValueError("quantized tensor size must be a multiple of 32")
    blocks = element_count // QUANT_BLOCK
    if type_id == GGML_TYPE_Q8_0:
        return blocks * 34
    if type_id == GGML_TYPE_AFFINE4_GS32:
        return blocks * 20
    raise ValueError(f"unknown tensor type id {type_id}")


def bf16_bytes_to_f32(raw: bytes, shape: tuple[int, ...]):
    import numpy as np

    values = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
    return values.view(np.float32).reshape(shape)


def quantize_q8_0(weights_f32) -> bytes:
    """(rows, cols) f32 -> Q8_0 blocks (34 bytes each): fp16 d + int8 qs[32].
    Quantizes against the fp16-rounded scale so encoder and decoder agree."""
    import numpy as np

    w = np.ascontiguousarray(weights_f32, dtype=np.float32)
    rows, cols = w.shape
    blocks = w.reshape(rows * cols // QUANT_BLOCK, QUANT_BLOCK)
    amax = np.max(np.abs(blocks), axis=1)
    d = (amax / 127.0).astype(np.float16)
    dq = d.astype(np.float32)
    inv = np.where(dq > 0.0, 1.0 / dq, 0.0)
    qs = np.clip(np.rint(blocks * inv[:, None]), -127, 127).astype(np.int8)
    out = np.empty((blocks.shape[0], 34), dtype=np.uint8)
    out[:, 0:2] = d.view(np.uint8).reshape(-1, 2)
    out[:, 2:34] = qs.view(np.uint8)
    return out.tobytes()


def quantize_affine4_gs32(weights_f32) -> bytes:
    """(rows, cols) f32 -> AFFINE4_GS32 groups (20 bytes): 16 packed nibbles
    + fp16 scale + fp16 bias, w = scale*q + bias, q in [0,15]."""
    import numpy as np

    w = np.ascontiguousarray(weights_f32, dtype=np.float32)
    rows, cols = w.shape
    groups = w.reshape(rows * cols // QUANT_BLOCK, QUANT_BLOCK)
    gmin = np.min(groups, axis=1)
    gmax = np.max(groups, axis=1)
    scale = ((gmax - gmin) / 15.0).astype(np.float16)
    bias = gmin.astype(np.float16)
    sf = scale.astype(np.float32)
    bf = bias.astype(np.float32)
    inv = np.where(sf > 0.0, 1.0 / sf, 0.0)
    q = np.clip(np.rint((groups - bf[:, None]) * inv[:, None]), 0, 15).astype(
        np.uint8
    )
    lo = q[:, 0::2]
    hi = q[:, 1::2]
    packed = (lo | (hi << 4)).astype(np.uint8)  # (ngroups, 16)
    out = np.empty((groups.shape[0], 20), dtype=np.uint8)
    out[:, 0:16] = packed
    out[:, 16:18] = scale.view(np.uint8).reshape(-1, 2)
    out[:, 18:20] = bias.view(np.uint8).reshape(-1, 2)
    return out.tobytes()


def quantize_tensor(raw: bytes, shape: tuple[int, ...], quant: str) -> bytes:
    weights = bf16_bytes_to_f32(raw, shape)
    if quant == "q8_0":
        return quantize_q8_0(weights)
    if quant == "affine4_gs32":
        return quantize_affine4_gs32(weights)
    raise ValueError(f"unknown quant scheme {quant}")

UINT8 = 0
UINT32 = 4
FLOAT32 = 6
BOOL = 7
STRING = 8
ARRAY = 9


@dataclass(frozen=True)
class TensorSource:
    name: str
    shard: pathlib.Path
    shape: tuple[int, ...]
    source_offset: int
    source_bytes: int  # BF16 bytes in the source shard
    byte_count: int  # bytes in the output (may be quantized)
    gguf_dimensions: tuple[int, ...]
    gguf_offset: int
    type_id: int  # GGML type id in the output
    quant: str  # "bf16" | "q8_0" | "affine4_gs32"


def align(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) & -alignment


def product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def pack_string(value: str | bytes) -> bytes:
    encoded = value.encode("utf-8") if isinstance(value, str) else value
    return struct.pack("<Q", len(encoded)) + encoded


def metadata_string(key: str, value: str) -> bytes:
    return pack_string(key) + struct.pack("<I", STRING) + pack_string(value)


def metadata_u32(key: str, value: int) -> bytes:
    return pack_string(key) + struct.pack("<II", UINT32, value)


def metadata_f32(key: str, value: float) -> bytes:
    return pack_string(key) + struct.pack("<If", FLOAT32, value)


def metadata_bool(key: str, value: bool) -> bytes:
    return pack_string(key) + struct.pack("<IB", BOOL, int(value))


def metadata_array_strings(key: str, values: list[bytes]) -> bytes:
    body = b"".join(pack_string(value) for value in values)
    return (
        pack_string(key)
        + struct.pack("<IIQ", ARRAY, STRING, len(values))
        + body
    )


def metadata_array_u32(key: str, values: list[int]) -> bytes:
    body = struct.pack(f"<{len(values)}I", *values) if values else b""
    return (
        pack_string(key)
        + struct.pack("<IIQ", ARRAY, UINT32, len(values))
        + body
    )


def metadata_array_u8(key: str, values: list[int]) -> bytes:
    return (
        pack_string(key)
        + struct.pack("<IIQ", ARRAY, UINT8, len(values))
        + bytes(values)
    )


def expected_shapes(
    spec: checkpoints.CheckpointSpec,
) -> dict[str, tuple[int, ...]]:
    hidden = spec.embedding_length
    kv_width = checkpoints.KV_HEAD_COUNT * checkpoints.HEAD_DIM
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (checkpoints.VOCAB_SIZE, hidden),
        "model.norm.weight": (hidden,),
    }
    if not spec.tied_embeddings:
        shapes["lm_head.weight"] = (checkpoints.VOCAB_SIZE, hidden)
    for layer in range(spec.block_count):
        prefix = f"model.layers.{layer}"
        shapes.update(
            {
                f"{prefix}.input_layernorm.weight": (hidden,),
                f"{prefix}.self_attn.q_proj.weight":
                    (spec.attention_width, hidden),
                f"{prefix}.self_attn.k_proj.weight": (kv_width, hidden),
                f"{prefix}.self_attn.v_proj.weight": (kv_width, hidden),
                f"{prefix}.self_attn.o_proj.weight":
                    (hidden, spec.attention_width),
                f"{prefix}.self_attn.q_norm.weight": (checkpoints.HEAD_DIM,),
                f"{prefix}.self_attn.k_norm.weight": (checkpoints.HEAD_DIM,),
                f"{prefix}.post_attention_layernorm.weight": (hidden,),
                f"{prefix}.mlp.gate_proj.weight":
                    (spec.feed_forward_length, hidden),
                f"{prefix}.mlp.up_proj.weight":
                    (spec.feed_forward_length, hidden),
                f"{prefix}.mlp.down_proj.weight":
                    (hidden, spec.feed_forward_length),
            }
        )
    if len(shapes) != spec.tensor_count:
        raise AssertionError(f"internal tensor contract has {len(shapes)} tensors")
    return shapes


def read_safetensors_header(path: pathlib.Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as file:
        header_length_data = file.read(8)
        if len(header_length_data) != 8:
            raise ValueError(f"{path}: truncated safetensors header")
        header_length = struct.unpack("<Q", header_length_data)[0]
        if header_length > 100 * 1024 * 1024:
            raise ValueError(f"{path}: unreasonable safetensors header")
        header = json.loads(file.read(header_length))
    return 8 + header_length, header


def collect_tensors(source: pathlib.Path,
                    spec: checkpoints.CheckpointSpec,
                    quant: str = "bf16") -> list[TensorSource]:
    index_path = source / "model.safetensors.index.json"
    weight_map: dict[str, str]
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
    else:
        # Small checkpoints ship one unsharded model.safetensors.
        single = source / "model.safetensors"
        if not single.is_file():
            raise FileNotFoundError(
                "snapshot has neither a safetensors index nor model.safetensors"
            )
        _, header = read_safetensors_header(single)
        weight_map = {
            name: single.name
            for name in header
            if name != "__metadata__"
        }
    expected = expected_shapes(spec)
    if set(weight_map) != set(expected):
        missing = sorted(set(expected) - set(weight_map))
        extra = sorted(set(weight_map) - set(expected))
        raise ValueError(f"tensor set mismatch; missing={missing}, extra={extra}")

    headers: dict[str, tuple[int, dict[str, Any]]] = {}
    records: list[tuple[str, pathlib.Path, tuple[int, ...], int, int]] = []
    for name, shard_name in weight_map.items():
        if shard_name not in headers:
            headers[shard_name] = read_safetensors_header(source / shard_name)
        data_start, header = headers[shard_name]
        info = header.get(name)
        if not isinstance(info, dict):
            raise ValueError(f"{shard_name}: missing safetensors tensor {name}")
        dtype = info.get("dtype")
        shape = tuple(info.get("shape", ()))
        offsets = info.get("data_offsets", ())
        if dtype != "BF16" or shape != expected[name] or len(offsets) != 2:
            raise ValueError(
                f"{name}: expected BF16 {expected[name]}, got {dtype} {shape}"
            )
        begin, end = map(int, offsets)
        source_bytes = product(shape) * 2
        if begin < 0 or end - begin != source_bytes:
            raise ValueError(f"{name}: invalid safetensors byte range")
        records.append(
            (name, source / shard_name, shape, data_start + begin, source_bytes)
        )

    records.sort(key=lambda record: (record[1].name, record[3], record[0]))
    tensors: list[TensorSource] = []
    output_offset = 0
    for name, shard, shape, source_offset, source_bytes in records:
        tensor_quant = quant if is_projection(name) else "bf16"
        type_id = QUANT_TYPE_ID[tensor_quant]
        byte_count = type_byte_count(type_id, product(shape))
        output_offset = align(output_offset)
        tensors.append(
            TensorSource(
                name=name,
                shard=shard,
                shape=shape,
                source_offset=source_offset,
                source_bytes=source_bytes,
                byte_count=byte_count,
                gguf_dimensions=tuple(reversed(shape)),
                gguf_offset=output_offset,
                type_id=type_id,
                quant=tensor_quant,
            )
        )
        output_offset += byte_count
    return tensors


def gpt2_byte_decoder() -> dict[str, int]:
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codepoints = byte_values.copy()
    extra = 0
    present = set(byte_values)
    for byte in range(256):
        if byte not in present:
            byte_values.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    return {chr(codepoint): byte for byte, codepoint in zip(byte_values, codepoints)}


def decode_bpe_token(token: str, decoder: dict[str, int]) -> bytes:
    try:
        return bytes(decoder[character] for character in token)
    except KeyError as error:
        raise ValueError(f"unsupported tokenizer character U+{ord(error.args[0]):04X}") from error


def tokenizer_metadata(source: pathlib.Path) -> tuple[list[bytes], list[int], list[int], list[int], list[int]]:
    tokenizer = json.loads((source / "tokenizer.json").read_text(encoding="utf-8"))
    model = tokenizer["model"]
    vocab: dict[str, int] = model["vocab"]
    decoder = gpt2_byte_decoder()
    tokens = [b""] * 151936
    special = [0] * 151936
    token_string_by_id: dict[int, str] = {}

    for token_string, token_id_value in vocab.items():
        token_id = int(token_id_value)
        if not 0 <= token_id < len(tokens):
            raise ValueError(f"vocabulary token ID {token_id} is out of range")
        tokens[token_id] = decode_bpe_token(token_string, decoder)
        token_string_by_id[token_id] = token_string

    for added in tokenizer.get("added_tokens", []):
        token_id = int(added["id"])
        if not 0 <= token_id < len(tokens):
            raise ValueError(f"added token ID {token_id} is out of range")
        tokens[token_id] = str(added["content"]).encode("utf-8")
        # Tokenizers isolates every AddedToken before BPE, even when its
        # `special` flag controls only skip-special-token behavior.
        special[token_id] = 1

    # Hugging Face additionally merges tokenizer_config.json's
    # added_tokens_decoder over tokenizer.json. Every Qwen3 checkpoint —
    # base included — declares the think/tool tokens (151665-151668) only
    # there, so skipping this merge would silently mis-tokenize them.
    tokenizer_config = json.loads(
        (source / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    for key, added in tokenizer_config.get("added_tokens_decoder", {}).items():
        token_id = int(key)
        if not 0 <= token_id < len(tokens):
            raise ValueError(f"decoder token ID {token_id} is out of range")
        content = str(added["content"]).encode("utf-8")
        if special[token_id] and tokens[token_id] != content:
            raise ValueError(
                f"added token {token_id} disagrees between tokenizer.json "
                f"and tokenizer_config.json"
            )
        tokens[token_id] = content
        special[token_id] = 1

    id_by_string = {token: int(token_id) for token, token_id in vocab.items()}
    left_ids: list[int] = []
    right_ids: list[int] = []
    result_ids: list[int] = []
    for rank, merge in enumerate(model["merges"]):
        if isinstance(merge, str):
            try:
                left, right = merge.split(" ", 1)
            except ValueError as error:
                raise ValueError(f"invalid merge at rank {rank}: {merge!r}") from error
        else:
            left, right = merge
        if left not in id_by_string or right not in id_by_string:
            raise ValueError(f"merge {rank} references an unknown token")
        result = left + right
        if result not in id_by_string:
            raise ValueError(f"merge {rank} result is not in the vocabulary")
        left_ids.append(id_by_string[left])
        right_ids.append(id_by_string[right])
        result_ids.append(id_by_string[result])

    for byte in range(256):
        if bytes([byte]) not in tokens:
            raise ValueError(f"tokenizer lacks raw byte token {byte:#04x}")
    return tokens, special, left_ids, right_ids, result_ids


def build_metadata(source: pathlib.Path,
                   spec: checkpoints.CheckpointSpec,
                   quant: str = "bf16") -> list[bytes]:
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    required = {
        "model_type": "qwen3",
        "num_hidden_layers": spec.block_count,
        "hidden_size": spec.embedding_length,
        "intermediate_size": spec.feed_forward_length,
        "num_attention_heads": spec.attention_head_count,
        "num_key_value_heads": checkpoints.KV_HEAD_COUNT,
        "head_dim": checkpoints.HEAD_DIM,
        "max_position_embeddings": spec.context_length,
        "vocab_size": checkpoints.VOCAB_SIZE,
        "rope_theta": spec.rope_theta,
        "rms_norm_eps": checkpoints.RMS_NORM_EPS,
        "tie_word_embeddings": spec.tied_embeddings,
        "torch_dtype": "bfloat16",
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"config {key}: expected {expected!r}, got {config.get(key)!r}")

    tokens, special, merge_left, merge_right, merge_result = tokenizer_metadata(source)
    return [
        metadata_string("general.architecture", "qwen3"),
        metadata_string("general.name", spec.id),
        metadata_u32("general.alignment", ALIGNMENT),
        metadata_string("kipp.source.repository", spec.repository),
        metadata_string("kipp.source.revision", spec.revision),
        metadata_string("kipp.quant.scheme", quant),
        metadata_u32("qwen3.block_count", spec.block_count),
        metadata_u32("qwen3.embedding_length", spec.embedding_length),
        metadata_u32("qwen3.feed_forward_length", spec.feed_forward_length),
        metadata_u32("qwen3.attention.head_count", spec.attention_head_count),
        metadata_u32("qwen3.attention.head_count_kv",
                     checkpoints.KV_HEAD_COUNT),
        metadata_u32("qwen3.attention.key_length", checkpoints.HEAD_DIM),
        metadata_u32("qwen3.context_length", spec.context_length),
        metadata_u32("qwen3.vocab_size", checkpoints.VOCAB_SIZE),
        metadata_f32("qwen3.rope.freq_base", spec.rope_theta),
        metadata_f32("qwen3.attention.layer_norm_rms_epsilon",
                     checkpoints.RMS_NORM_EPS),
        metadata_bool("qwen3.tie_word_embeddings", spec.tied_embeddings),
        metadata_string("tokenizer.ggml.model", "gpt2"),
        metadata_string("tokenizer.ggml.pre", "qwen2"),
        metadata_bool("tokenizer.ggml.add_bos_token", False),
        metadata_u32("tokenizer.ggml.eos_token_id", spec.eos_token_id),
        metadata_array_strings("kipp.tokenizer.tokens", tokens),
        metadata_array_u8("kipp.tokenizer.special", special),
        metadata_array_u32("kipp.tokenizer.merge_left", merge_left),
        metadata_array_u32("kipp.tokenizer.merge_right", merge_right),
        metadata_array_u32("kipp.tokenizer.merge_result", merge_result),
    ]


def tensor_directory_entry(tensor: TensorSource) -> bytes:
    dimensions = struct.pack(
        f"<{len(tensor.gguf_dimensions)}Q", *tensor.gguf_dimensions
    )
    return (
        pack_string(tensor.name)
        + struct.pack("<I", len(tensor.gguf_dimensions))
        + dimensions
        + struct.pack("<IQ", tensor.type_id, tensor.gguf_offset)
    )


def copy_range(source: BinaryIO, destination: BinaryIO, offset: int, count: int) -> None:
    source.seek(offset)
    remaining = count
    while remaining:
        chunk = source.read(min(8 * 1024 * 1024, remaining))
        if not chunk:
            raise EOFError("safetensors payload ended early")
        destination.write(chunk)
        remaining -= len(chunk)


def write_gguf(source: pathlib.Path, output: pathlib.Path,
               spec: checkpoints.CheckpointSpec,
               quant: str = "bf16") -> dict[str, Any]:
    tensors = collect_tensors(source, spec, quant)
    metadata = build_metadata(source, spec, quant)
    header = (
        b"GGUF"
        + struct.pack("<IQQ", GGUF_VERSION, len(tensors), len(metadata))
        + b"".join(metadata)
        + b"".join(tensor_directory_entry(tensor) for tensor in tensors)
    )
    data_start = align(len(header))
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    expected_size = data_start + tensors[-1].gguf_offset + tensors[-1].byte_count
    output_sha256: str

    open_shard: pathlib.Path | None = None
    shard_file: BinaryIO | None = None
    try:
        with temporary.open("wb") as destination:
            destination.write(header)
            destination.write(b"\0" * (data_start - len(header)))
            for tensor in tensors:
                expected_position = data_start + tensor.gguf_offset
                current_position = destination.tell()
                if current_position > expected_position:
                    raise AssertionError("tensor offsets overlap")
                destination.write(b"\0" * (expected_position - current_position))
                if tensor.shard != open_shard:
                    if shard_file is not None:
                        shard_file.close()
                    shard_file = tensor.shard.open("rb")
                    open_shard = tensor.shard
                assert shard_file is not None
                if tensor.quant == "bf16":
                    copy_range(
                        shard_file,
                        destination,
                        tensor.source_offset,
                        tensor.byte_count,
                    )
                else:
                    shard_file.seek(tensor.source_offset)
                    raw = shard_file.read(tensor.source_bytes)
                    if len(raw) != tensor.source_bytes:
                        raise EOFError("safetensors payload ended early")
                    packed = quantize_tensor(raw, tensor.shape, tensor.quant)
                    if len(packed) != tensor.byte_count:
                        raise AssertionError(
                            f"{tensor.name}: quantized to {len(packed)} bytes, "
                            f"expected {tensor.byte_count}"
                        )
                    destination.write(packed)
            destination.flush()
            os.fsync(destination.fileno())
        if temporary.stat().st_size != expected_size:
            raise ValueError(
                "temporary GGUF size mismatch: "
                f"expected {expected_size}, got {temporary.stat().st_size}"
            )
        with temporary.open("rb") as validation:
            if validation.read(8) != b"GGUF" + struct.pack("<I", GGUF_VERSION):
                raise ValueError("temporary GGUF failed header self-validation")
        output_sha256 = sha256(temporary)
        temporary.replace(output)
    finally:
        if shard_file is not None:
            shard_file.close()
        if temporary.exists():
            temporary.unlink()

    if output.stat().st_size != expected_size:
        raise ValueError(
            f"GGUF size mismatch: expected {expected_size}, got {output.stat().st_size}"
        )
    return {
        "schema": 1,
        "checkpoint": spec.id,
        "repository": spec.repository,
        "revision": spec.revision,
        "quant": quant,
        "format": "GGUF-v3 Kipp Qwen3 subset",
        "tensor_count": len(tensors),
        "bytes": output.stat().st_size,
        "sha256": output_sha256,
        "source_manifest_sha256": sha256(source / "source-manifest.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="qwen3-4b-base")
    parser.add_argument(
        "--quant", choices=["bf16", "q8_0", "affine4_gs32"], default="bf16"
    )
    parser.add_argument("--source", type=pathlib.Path, default=None)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args()
    spec = checkpoints.get(args.checkpoint)
    source = args.source or pathlib.Path("models") / spec.id / "source"
    output = (
        args.output
        or pathlib.Path("models") / spec.id / f"kipp-{spec.id}-{args.quant}.gguf"
    )
    source = source.resolve()
    output = output.resolve()

    source_manifest = json.loads(
        (source / "source-manifest.json").read_text(encoding="utf-8")
    )
    if (
        source_manifest.get("repository") != spec.repository
        or source_manifest.get("revision") != spec.revision
    ):
        raise ValueError(
            f"source manifest does not match registry entry '{spec.id}'"
        )

    manifest = write_gguf(source, output, spec, args.quant)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output}")
    print(f"SHA-256 {manifest['sha256']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"convert_to_gguf.py: {error}", file=sys.stderr)
        raise SystemExit(1) from error
