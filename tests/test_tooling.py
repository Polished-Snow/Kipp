#!/usr/bin/env python3
"""Fast, model-free tests for Kipp's artifact tooling."""

from __future__ import annotations

import pathlib
import re
import struct
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import checkpoints  # noqa: E402
import convert_to_gguf as converter  # noqa: E402
import generate_chat_vectors as chat_vectors  # noqa: E402
import generate_unicode_tables as unicode_tables  # noqa: E402


def parse_c_registry() -> list[dict[str, object]]:
    """Extract the compiled-in registry from src/kipp_checkpoints.h."""
    text = (ROOT / "src" / "kipp_checkpoints.h").read_text(encoding="utf-8")
    body = text.split("kipp_supported_checkpoints[] = {", 1)[1]
    body = body.split("};", 1)[0]
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL)
    pattern = re.compile(
        r'\{"(?P<id>[^"]+)",\s*"(?P<repo>[^"]+)",\s*'
        r'"(?P<rev>[0-9a-f]{40})",\s*KIPP_VARIANT_(?P<variant>\w+),\s*'
        r"(?P<layers>\d+)u,\s*(?P<hidden>\d+)u,\s*(?P<ffn>\d+)u,\s*"
        r"(?P<heads>\d+)u,\s*(?P<ctx>\d+)u,\s*(?P<theta>[\d.]+)f,\s*"
        r"(?P<tied>true|false),\s*"
        r"KIPP_(?P<eos>ENDOFTEXT|IM_END)_TOKEN_ID,\s*"
        r"KIPP_(?P<stops>BASE|INSTRUCT)_STOPS\}",
        re.DOTALL,
    )
    entries = [match.groupdict() for match in pattern.finditer(body)]
    if not entries:
        raise AssertionError("no registry entries parsed from C header")
    return entries


C_VARIANT_NAMES = {
    checkpoints.VARIANT_BASE: "BASE",
    checkpoints.VARIANT_INSTRUCT: "INSTRUCT",
    checkpoints.VARIANT_INSTRUCT_2507: "INSTRUCT_2507",
    checkpoints.VARIANT_THINKING_2507: "THINKING_2507",
}


class RegistryTests(unittest.TestCase):
    def test_c_and_python_registries_match(self) -> None:
        c_entries = {entry["id"]: entry for entry in parse_c_registry()}
        self.assertEqual(
            set(c_entries), set(checkpoints.BY_ID),
            "registry ids differ between C and Python",
        )
        for spec in checkpoints.SUPPORTED_CHECKPOINTS:
            entry = c_entries[spec.id]
            self.assertEqual(entry["repo"], spec.repository, spec.id)
            self.assertEqual(entry["rev"], spec.revision, spec.id)
            self.assertEqual(
                entry["variant"], C_VARIANT_NAMES[spec.variant], spec.id
            )
            self.assertEqual(int(entry["layers"]), spec.block_count, spec.id)
            self.assertEqual(
                int(entry["hidden"]), spec.embedding_length, spec.id
            )
            self.assertEqual(
                int(entry["ffn"]), spec.feed_forward_length, spec.id
            )
            self.assertEqual(
                int(entry["heads"]), spec.attention_head_count, spec.id
            )
            self.assertEqual(int(entry["ctx"]), spec.context_length, spec.id)
            self.assertEqual(
                float(entry["theta"]), spec.rope_theta, spec.id
            )
            self.assertEqual(
                entry["tied"] == "true", spec.tied_embeddings, spec.id
            )
            expected_eos = (
                "ENDOFTEXT"
                if spec.eos_token_id == checkpoints.ENDOFTEXT_TOKEN_ID
                else "IM_END"
            )
            self.assertEqual(entry["eos"], expected_eos, spec.id)
            expected_stops = (
                "BASE" if spec.stop_tokens == (checkpoints.ENDOFTEXT_TOKEN_ID,)
                else "INSTRUCT"
            )
            self.assertEqual(entry["stops"], expected_stops, spec.id)

    def test_registry_invariants(self) -> None:
        ids = [spec.id for spec in checkpoints.SUPPORTED_CHECKPOINTS]
        self.assertEqual(len(ids), len(set(ids)))
        for spec in checkpoints.SUPPORTED_CHECKPOINTS:
            self.assertLessEqual(spec.block_count, 64)
            self.assertEqual(spec.attention_head_count % 8, 0)
            self.assertRegex(spec.revision, r"^[0-9a-f]{40}$")
            if spec.variant == checkpoints.VARIANT_BASE:
                self.assertEqual(
                    spec.stop_tokens, (checkpoints.ENDOFTEXT_TOKEN_ID,)
                )
            else:
                self.assertIn(checkpoints.IM_END_TOKEN_ID, spec.stop_tokens)


class ConverterTests(unittest.TestCase):
    def test_tensor_contract_4b(self) -> None:
        spec = checkpoints.get("qwen3-4b-base")
        shapes = converter.expected_shapes(spec)
        self.assertEqual(len(shapes), 398)
        self.assertEqual(shapes["model.embed_tokens.weight"], (151936, 2560))
        self.assertEqual(
            shapes["model.layers.35.mlp.down_proj.weight"], (2560, 9728)
        )
        self.assertNotIn("lm_head.weight", shapes)

    def test_tensor_contract_untied_and_narrow(self) -> None:
        spec = checkpoints.get("qwen3-8b-base")
        shapes = converter.expected_shapes(spec)
        self.assertEqual(len(shapes), 2 + 36 * 11 + 1)
        self.assertEqual(shapes["lm_head.weight"], (151936, 4096))
        # 0.6B: attention width (2048) exceeds the hidden width (1024).
        narrow = converter.expected_shapes(checkpoints.get("qwen3-0.6b-base"))
        self.assertEqual(
            narrow["model.layers.0.self_attn.q_proj.weight"], (2048, 1024)
        )
        self.assertEqual(
            narrow["model.layers.0.self_attn.o_proj.weight"], (1024, 2048)
        )

    def test_alignment(self) -> None:
        self.assertEqual(converter.align(0), 0)
        self.assertEqual(converter.align(1), 32)
        self.assertEqual(converter.align(32), 32)
        self.assertEqual(converter.align(33), 64)

    def test_gpt2_byte_decoder_is_bijective(self) -> None:
        decoder = converter.gpt2_byte_decoder()
        self.assertEqual(len(decoder), 256)
        self.assertEqual(set(decoder.values()), set(range(256)))

    def test_metadata_wire_encoding(self) -> None:
        encoded = converter.metadata_u32("test.key", 123)
        key_length = struct.unpack_from("<Q", encoded)[0]
        self.assertEqual(encoded[8 : 8 + key_length], b"test.key")
        value_type, value = struct.unpack_from("<II", encoded, 8 + key_length)
        self.assertEqual(value_type, converter.UINT32)
        self.assertEqual(value, 123)


class UnicodeGeneratorTests(unittest.TestCase):
    def test_range_compaction(self) -> None:
        ranges = unicode_tables.ranges_for(lambda codepoint: 10 <= codepoint <= 12)
        self.assertEqual(ranges, [(10, 12)])

    def test_canonical_table_excludes_compatibility_and_hangul(self) -> None:
        decompositions = dict(unicode_tables.canonical_decompositions())
        self.assertIn(ord("é"), decompositions)
        self.assertNotIn(ord("①"), decompositions)
        self.assertNotIn(ord("가"), decompositions)


class ChatVectorGeneratorTests(unittest.TestCase):
    def test_case_registry_is_complete_and_unique(self) -> None:
        names = [case[0] for case in chat_vectors.CASES]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(
            set(names),
            {
                "user-only",
                "system-user",
                "multiturn",
                "no-genprompt",
                "think-off",
                "think-on",
            },
        )
        for name, messages, _add_generation_prompt, enable_thinking in (
            chat_vectors.CASES
        ):
            self.assertTrue(messages, name)
            self.assertIn(enable_thinking, (None, False, True), name)


if __name__ == "__main__":
    unittest.main()
