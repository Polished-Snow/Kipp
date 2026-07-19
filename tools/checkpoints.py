#!/usr/bin/env python3
"""The supported-checkpoint registry, mirroring src/kipp_checkpoints.h.

This module is the tooling-side source of truth for which pinned Qwen3
dense checkpoints Kipp supports. tests/test_tooling.py cross-checks every
entry against the C registry so the two copies cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass

ENDOFTEXT_TOKEN_ID = 151643
IM_END_TOKEN_ID = 151645

# Family-invariant values, validated per checkpoint by the converter.
HEAD_DIM = 128
KV_HEAD_COUNT = 8
VOCAB_SIZE = 151936
RMS_NORM_EPS = 1e-6

VARIANT_BASE = "base"
VARIANT_INSTRUCT = "instruct"
VARIANT_INSTRUCT_2507 = "instruct-2507"
VARIANT_THINKING_2507 = "thinking-2507"


@dataclass(frozen=True)
class CheckpointSpec:
    id: str
    repository: str
    revision: str
    variant: str
    block_count: int
    embedding_length: int
    feed_forward_length: int
    attention_head_count: int
    context_length: int
    rope_theta: float
    tied_embeddings: bool
    eos_token_id: int
    stop_tokens: tuple[int, ...]

    @property
    def attention_width(self) -> int:
        return self.attention_head_count * HEAD_DIM

    @property
    def tensor_count(self) -> int:
        return 2 + self.block_count * 11 + (0 if self.tied_embeddings else 1)


_BASE_STOPS = (ENDOFTEXT_TOKEN_ID,)
_INSTRUCT_STOPS = (IM_END_TOKEN_ID, ENDOFTEXT_TOKEN_ID)

SUPPORTED_CHECKPOINTS: tuple[CheckpointSpec, ...] = (
    CheckpointSpec(
        "qwen3-0.6b-base", "Qwen/Qwen3-0.6B-Base",
        "da87bfb608c14b7cf20ba1ce41287e8de496c0cd", VARIANT_BASE,
        28, 1024, 3072, 16, 32768, 1000000.0, True,
        ENDOFTEXT_TOKEN_ID, _BASE_STOPS),
    CheckpointSpec(
        "qwen3-0.6b", "Qwen/Qwen3-0.6B",
        "c1899de289a04d12100db370d81485cdf75e47ca", VARIANT_INSTRUCT,
        28, 1024, 3072, 16, 40960, 1000000.0, True,
        IM_END_TOKEN_ID, _INSTRUCT_STOPS),
    CheckpointSpec(
        "qwen3-1.7b-base", "Qwen/Qwen3-1.7B-Base",
        "ea980cb0a6c2ae4b936e82123acc929f1cec04c1", VARIANT_BASE,
        28, 2048, 6144, 16, 32768, 1000000.0, True,
        ENDOFTEXT_TOKEN_ID, _BASE_STOPS),
    CheckpointSpec(
        "qwen3-1.7b", "Qwen/Qwen3-1.7B",
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", VARIANT_INSTRUCT,
        28, 2048, 6144, 16, 40960, 1000000.0, True,
        IM_END_TOKEN_ID, _INSTRUCT_STOPS),
    CheckpointSpec(
        "qwen3-4b-base", "Qwen/Qwen3-4B-Base",
        "906bfd4b4dc7f14ee4320094d8b41684abff8539", VARIANT_BASE,
        36, 2560, 9728, 32, 32768, 1000000.0, True,
        ENDOFTEXT_TOKEN_ID, _BASE_STOPS),
    CheckpointSpec(
        "qwen3-4b", "Qwen/Qwen3-4B",
        "1cfa9a7208912126459214e8b04321603b3df60c", VARIANT_INSTRUCT,
        36, 2560, 9728, 32, 40960, 1000000.0, True,
        IM_END_TOKEN_ID, _INSTRUCT_STOPS),
    CheckpointSpec(
        "qwen3-4b-instruct-2507", "Qwen/Qwen3-4B-Instruct-2507",
        "cdbee75f17c01a7cc42f958dc650907174af0554", VARIANT_INSTRUCT_2507,
        36, 2560, 9728, 32, 262144, 5000000.0, True,
        IM_END_TOKEN_ID, _INSTRUCT_STOPS),
    CheckpointSpec(
        "qwen3-4b-thinking-2507", "Qwen/Qwen3-4B-Thinking-2507",
        "768f209d9ea81521153ed38c47d515654e938aea", VARIANT_THINKING_2507,
        36, 2560, 9728, 32, 262144, 5000000.0, True,
        IM_END_TOKEN_ID, _INSTRUCT_STOPS),
    CheckpointSpec(
        "qwen3-8b-base", "Qwen/Qwen3-8B-Base",
        "49e3418fbbbca6ecbdf9608b4d22e5a407081db4", VARIANT_BASE,
        36, 4096, 12288, 32, 32768, 1000000.0, False,
        ENDOFTEXT_TOKEN_ID, _BASE_STOPS),
    CheckpointSpec(
        "qwen3-8b", "Qwen/Qwen3-8B",
        "b968826d9c46dd6066d109eabc6255188de91218", VARIANT_INSTRUCT,
        36, 4096, 12288, 32, 40960, 1000000.0, False,
        IM_END_TOKEN_ID, _INSTRUCT_STOPS),
    CheckpointSpec(
        "qwen3-14b-base", "Qwen/Qwen3-14B-Base",
        "0b0bd3732e2c374d483664439ea334928b65f304", VARIANT_BASE,
        40, 5120, 17408, 40, 32768, 1000000.0, False,
        ENDOFTEXT_TOKEN_ID, _BASE_STOPS),
    CheckpointSpec(
        "qwen3-14b", "Qwen/Qwen3-14B",
        "40c069824f4251a91eefaf281ebe4c544efd3e18", VARIANT_INSTRUCT,
        40, 5120, 17408, 40, 40960, 1000000.0, False,
        IM_END_TOKEN_ID, _INSTRUCT_STOPS),
    # Qwen3-32B-Base was never released publicly; instruct only.
    CheckpointSpec(
        "qwen3-32b", "Qwen/Qwen3-32B",
        "9216db5781bf21249d130ec9da846c4624c16137", VARIANT_INSTRUCT,
        64, 5120, 25600, 64, 40960, 1000000.0, False,
        IM_END_TOKEN_ID, _INSTRUCT_STOPS),
)

BY_ID = {spec.id: spec for spec in SUPPORTED_CHECKPOINTS}


def get(checkpoint_id: str) -> CheckpointSpec:
    try:
        return BY_ID[checkpoint_id]
    except KeyError:
        known = ", ".join(sorted(BY_ID))
        raise SystemExit(
            f"unknown checkpoint '{checkpoint_id}'; supported: {known}"
        ) from None
