#!/usr/bin/env python3
"""Stream the pinned Qwen3-4B-Base snapshot into Kipp's strict GGUF subset."""

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

REPOSITORY = "Qwen/Qwen3-4B-Base"
REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
ALIGNMENT = 32
GGUF_VERSION = 3
GGML_TYPE_BF16 = 30
EXPECTED_TENSOR_COUNT = 398

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
    byte_count: int
    gguf_dimensions: tuple[int, ...]
    gguf_offset: int


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


def expected_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (151936, 2560),
        "model.norm.weight": (2560,),
    }
    for layer in range(36):
        prefix = f"model.layers.{layer}"
        shapes.update(
            {
                f"{prefix}.input_layernorm.weight": (2560,),
                f"{prefix}.self_attn.q_proj.weight": (4096, 2560),
                f"{prefix}.self_attn.k_proj.weight": (1024, 2560),
                f"{prefix}.self_attn.v_proj.weight": (1024, 2560),
                f"{prefix}.self_attn.o_proj.weight": (2560, 4096),
                f"{prefix}.self_attn.q_norm.weight": (128,),
                f"{prefix}.self_attn.k_norm.weight": (128,),
                f"{prefix}.post_attention_layernorm.weight": (2560,),
                f"{prefix}.mlp.gate_proj.weight": (9728, 2560),
                f"{prefix}.mlp.up_proj.weight": (9728, 2560),
                f"{prefix}.mlp.down_proj.weight": (2560, 9728),
            }
        )
    if len(shapes) != EXPECTED_TENSOR_COUNT:
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


def collect_tensors(source: pathlib.Path) -> list[TensorSource]:
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    expected = expected_shapes()
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
        byte_count = product(shape) * 2
        if begin < 0 or end - begin != byte_count:
            raise ValueError(f"{name}: invalid safetensors byte range")
        records.append((name, source / shard_name, shape, data_start + begin, byte_count))

    records.sort(key=lambda record: (record[1].name, record[3], record[0]))
    tensors: list[TensorSource] = []
    output_offset = 0
    for name, shard, shape, source_offset, byte_count in records:
        output_offset = align(output_offset)
        tensors.append(
            TensorSource(
                name=name,
                shard=shard,
                shape=shape,
                source_offset=source_offset,
                byte_count=byte_count,
                gguf_dimensions=tuple(reversed(shape)),
                gguf_offset=output_offset,
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


def build_metadata(source: pathlib.Path) -> list[bytes]:
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    required = {
        "model_type": "qwen3",
        "num_hidden_layers": 36,
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 32768,
        "vocab_size": 151936,
        "rope_theta": 1000000,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"config {key}: expected {expected!r}, got {config.get(key)!r}")

    tokens, special, merge_left, merge_right, merge_result = tokenizer_metadata(source)
    return [
        metadata_string("general.architecture", "qwen3"),
        metadata_string("general.name", "Qwen3-4B-Base"),
        metadata_u32("general.alignment", ALIGNMENT),
        metadata_string("kipp.source.repository", REPOSITORY),
        metadata_string("kipp.source.revision", REVISION),
        metadata_u32("qwen3.block_count", 36),
        metadata_u32("qwen3.embedding_length", 2560),
        metadata_u32("qwen3.feed_forward_length", 9728),
        metadata_u32("qwen3.attention.head_count", 32),
        metadata_u32("qwen3.attention.head_count_kv", 8),
        metadata_u32("qwen3.attention.key_length", 128),
        metadata_u32("qwen3.context_length", 32768),
        metadata_u32("qwen3.vocab_size", 151936),
        metadata_f32("qwen3.rope.freq_base", 1000000.0),
        metadata_f32("qwen3.attention.layer_norm_rms_epsilon", 1.0e-6),
        metadata_bool("qwen3.tie_word_embeddings", True),
        metadata_string("tokenizer.ggml.model", "gpt2"),
        metadata_string("tokenizer.ggml.pre", "qwen2"),
        metadata_bool("tokenizer.ggml.add_bos_token", False),
        metadata_u32("tokenizer.ggml.eos_token_id", 151643),
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
        + struct.pack("<IQ", GGML_TYPE_BF16, tensor.gguf_offset)
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


def write_gguf(source: pathlib.Path, output: pathlib.Path) -> dict[str, Any]:
    tensors = collect_tensors(source)
    metadata = build_metadata(source)
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
                copy_range(
                    shard_file,
                    destination,
                    tensor.source_offset,
                    tensor.byte_count,
                )
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
        "repository": REPOSITORY,
        "revision": REVISION,
        "format": "GGUF-v3 Kipp Qwen3 subset",
        "tensor_count": len(tensors),
        "bytes": output.stat().st_size,
        "sha256": output_sha256,
        "source_manifest_sha256": sha256(source / "source-manifest.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=pathlib.Path,
        default=pathlib.Path("models/qwen3-4b-base/source"),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("models/qwen3-4b-base/kipp-qwen3-4b-base-bf16.gguf"),
    )
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()

    source_manifest = json.loads(
        (source / "source-manifest.json").read_text(encoding="utf-8")
    )
    if (
        source_manifest.get("repository") != REPOSITORY
        or source_manifest.get("revision") != REVISION
    ):
        raise ValueError("source manifest is not Kipp's pinned model")

    manifest = write_gguf(source, output)
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
