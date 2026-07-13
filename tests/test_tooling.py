#!/usr/bin/env python3
"""Fast, model-free tests for Kipp's artifact tooling."""

from __future__ import annotations

import pathlib
import struct
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import convert_to_gguf as converter  # noqa: E402
import generate_unicode_tables as unicode_tables  # noqa: E402


class ConverterTests(unittest.TestCase):
    def test_fixed_tensor_contract(self) -> None:
        shapes = converter.expected_shapes()
        self.assertEqual(len(shapes), converter.EXPECTED_TENSOR_COUNT)
        self.assertEqual(shapes["model.embed_tokens.weight"], (151936, 2560))
        self.assertEqual(
            shapes["model.layers.35.mlp.down_proj.weight"], (2560, 9728)
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


if __name__ == "__main__":
    unittest.main()
