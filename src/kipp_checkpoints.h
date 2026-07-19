#ifndef KIPP_CHECKPOINTS_H
#define KIPP_CHECKPOINTS_H

/*
 * The compiled-in registry of supported, pinned Qwen3 dense checkpoints.
 * This table is the ONLY growth point for model support: adding a checkpoint
 * means adding one entry here (and its mirror in tools/checkpoints.py),
 * converting it, generating vectors, and running the gates — no code changes.
 *
 * Every value below was read from the pinned revision's config.json and
 * generation_config.json on Hugging Face. Family-invariant values
 * (head_dim 128, 8 KV heads, vocab 151936, RMS epsilon 1e-6, BF16, no
 * sliding window, no rope scaling) are validated against the KIPP_*
 * constants and are not repeated per entry.
 *
 * tools/checkpoints.py mirrors this table for the converter and download
 * tooling; tests/test_tooling.py cross-checks the two copies.
 */

#include "kipp_backend.h"

#define KIPP_BASE_STOPS {KIPP_ENDOFTEXT_TOKEN_ID, 0}, 1u
#define KIPP_INSTRUCT_STOPS {KIPP_IM_END_TOKEN_ID, KIPP_ENDOFTEXT_TOKEN_ID}, 2u

static const kipp_checkpoint_spec kipp_supported_checkpoints[] = {
    {"qwen3-0.6b-base", "Qwen/Qwen3-0.6B-Base",
     "da87bfb608c14b7cf20ba1ce41287e8de496c0cd", KIPP_VARIANT_BASE,
     28u, 1024u, 3072u, 16u, 32768u, 1000000.0f, true,
     KIPP_ENDOFTEXT_TOKEN_ID, KIPP_BASE_STOPS},
    {"qwen3-0.6b", "Qwen/Qwen3-0.6B",
     "c1899de289a04d12100db370d81485cdf75e47ca", KIPP_VARIANT_INSTRUCT,
     28u, 1024u, 3072u, 16u, 40960u, 1000000.0f, true,
     KIPP_IM_END_TOKEN_ID, KIPP_INSTRUCT_STOPS},
    {"qwen3-1.7b-base", "Qwen/Qwen3-1.7B-Base",
     "ea980cb0a6c2ae4b936e82123acc929f1cec04c1", KIPP_VARIANT_BASE,
     28u, 2048u, 6144u, 16u, 32768u, 1000000.0f, true,
     KIPP_ENDOFTEXT_TOKEN_ID, KIPP_BASE_STOPS},
    {"qwen3-1.7b", "Qwen/Qwen3-1.7B",
     "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", KIPP_VARIANT_INSTRUCT,
     28u, 2048u, 6144u, 16u, 40960u, 1000000.0f, true,
     KIPP_IM_END_TOKEN_ID, KIPP_INSTRUCT_STOPS},
    {"qwen3-4b-base", "Qwen/Qwen3-4B-Base",
     "906bfd4b4dc7f14ee4320094d8b41684abff8539", KIPP_VARIANT_BASE,
     36u, 2560u, 9728u, 32u, 32768u, 1000000.0f, true,
     KIPP_ENDOFTEXT_TOKEN_ID, KIPP_BASE_STOPS},
    {"qwen3-4b", "Qwen/Qwen3-4B",
     "1cfa9a7208912126459214e8b04321603b3df60c", KIPP_VARIANT_INSTRUCT,
     36u, 2560u, 9728u, 32u, 40960u, 1000000.0f, true,
     KIPP_IM_END_TOKEN_ID, KIPP_INSTRUCT_STOPS},
    {"qwen3-4b-instruct-2507", "Qwen/Qwen3-4B-Instruct-2507",
     "cdbee75f17c01a7cc42f958dc650907174af0554", KIPP_VARIANT_INSTRUCT_2507,
     36u, 2560u, 9728u, 32u, 262144u, 5000000.0f, true,
     KIPP_IM_END_TOKEN_ID, KIPP_INSTRUCT_STOPS},
    {"qwen3-4b-thinking-2507", "Qwen/Qwen3-4B-Thinking-2507",
     "768f209d9ea81521153ed38c47d515654e938aea", KIPP_VARIANT_THINKING_2507,
     36u, 2560u, 9728u, 32u, 262144u, 5000000.0f, true,
     KIPP_IM_END_TOKEN_ID, KIPP_INSTRUCT_STOPS},
    {"qwen3-8b-base", "Qwen/Qwen3-8B-Base",
     "49e3418fbbbca6ecbdf9608b4d22e5a407081db4", KIPP_VARIANT_BASE,
     36u, 4096u, 12288u, 32u, 32768u, 1000000.0f, false,
     KIPP_ENDOFTEXT_TOKEN_ID, KIPP_BASE_STOPS},
    {"qwen3-8b", "Qwen/Qwen3-8B",
     "b968826d9c46dd6066d109eabc6255188de91218", KIPP_VARIANT_INSTRUCT,
     36u, 4096u, 12288u, 32u, 40960u, 1000000.0f, false,
     KIPP_IM_END_TOKEN_ID, KIPP_INSTRUCT_STOPS},
    {"qwen3-14b-base", "Qwen/Qwen3-14B-Base",
     "0b0bd3732e2c374d483664439ea334928b65f304", KIPP_VARIANT_BASE,
     40u, 5120u, 17408u, 40u, 32768u, 1000000.0f, false,
     KIPP_ENDOFTEXT_TOKEN_ID, KIPP_BASE_STOPS},
    {"qwen3-14b", "Qwen/Qwen3-14B",
     "40c069824f4251a91eefaf281ebe4c544efd3e18", KIPP_VARIANT_INSTRUCT,
     40u, 5120u, 17408u, 40u, 40960u, 1000000.0f, false,
     KIPP_IM_END_TOKEN_ID, KIPP_INSTRUCT_STOPS},
    /* Qwen3-32B-Base was never released publicly; instruct only. */
    {"qwen3-32b", "Qwen/Qwen3-32B",
     "9216db5781bf21249d130ec9da846c4624c16137", KIPP_VARIANT_INSTRUCT,
     64u, 5120u, 25600u, 64u, 40960u, 1000000.0f, false,
     KIPP_IM_END_TOKEN_ID, KIPP_INSTRUCT_STOPS},
};

#undef KIPP_BASE_STOPS
#undef KIPP_INSTRUCT_STOPS

#define KIPP_SUPPORTED_CHECKPOINT_COUNT \
    (sizeof(kipp_supported_checkpoints) / sizeof(kipp_supported_checkpoints[0]))

#endif
