#if !defined(__APPLE__) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include "kipp.h"
#include "kipp_spec.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define KIPP_CLI_MAX_DRAFT 8u

static uint32_t argmax_logits(const float *logits) {
    uint32_t best = 0;
    for (uint32_t token = 1; token < KIPP_VOCAB_SIZE; ++token) {
        if (logits[token] > logits[best]) {
            best = token;
        }
    }
    return best;
}

static int emit_token(const kipp_model *model, uint32_t token,
                      size_t *decoded, kipp_error *error) {
    char *piece = NULL;
    size_t piece_length = 0;
    if (kipp_detokenize(model, &token, 1, &piece, &piece_length, error) != 0) {
        return -1;
    }
    (void)fwrite(piece, 1, piece_length, stdout);
    (void)fflush(stdout);
    kipp_text_free(piece);
    ++*decoded;
    return 0;
}

/*
 * Greedy self-speculative decode: each step drafts the continuation via
 * prompt-lookup, verifies the whole draft in one multi-logit forward, and
 * accepts the longest greedy-matching prefix. The emitted sequence is exactly
 * the plain greedy sequence — only fewer forward passes produce it. With
 * gate_enabled zero the adaptive gate is bypassed and every step drafts,
 * which is the ungated baseline the speculation benchmark A/B needs.
 */
static int run_spec_decode(kipp_model *model, kipp_session *session,
                           float *logits, const uint32_t *prompt,
                           size_t prompt_len, size_t decode_count,
                           int gate_enabled, size_t *out_decoded,
                           uint64_t *out_drafted, uint64_t *out_accepted,
                           uint64_t *out_draft_steps,
                           uint64_t *out_plain_steps, kipp_error *error) {
    size_t capacity = prompt_len + decode_count + KIPP_CLI_MAX_DRAFT + 1;
    uint32_t *hist = malloc(capacity * sizeof(*hist));
    float *rows = malloc((size_t)(KIPP_CLI_MAX_DRAFT + 1) * KIPP_VOCAB_SIZE *
                         sizeof(*rows));
    uint32_t drafts[KIPP_CLI_MAX_DRAFT];
    uint32_t feed[KIPP_CLI_MAX_DRAFT + 1];
    if (hist == NULL || rows == NULL) {
        free(hist);
        free(rows);
        fprintf(stderr, "kipp: unable to allocate speculative buffers\n");
        (void)error;
        return -1;
    }
    memcpy(hist, prompt, prompt_len * sizeof(*hist));
    size_t hlen = prompt_len;       /* committed tokens (prompt + emitted) */
    uint32_t next_tok = argmax_logits(logits); /* token for position hlen */
    kipp_spec_gate gate;
    kipp_spec_gate_init(&gate);
    int status = -1;

    while (*out_decoded < decode_count) {
        if (kipp_model_is_stop_token(model, next_tok)) {
            break;
        }
        if (emit_token(model, next_tok, out_decoded, error) != 0) {
            goto done;
        }
        hist[hlen++] = next_tok;
        if (*out_decoded >= decode_count) {
            break;
        }
        /* When the gate has suspended drafting, even the history scan is
         * skipped; both paths emit the exact greedy sequence, so gating only
         * moves wall-clock. */
        size_t k = 0;
        if (!gate_enabled || kipp_spec_gate_should_draft(&gate)) {
            k = kipp_spec_prompt_lookup(hist, hlen, 3, 1, KIPP_CLI_MAX_DRAFT,
                                        drafts);
        }
        if (k == 0) {
            /* No draft: a single ordinary step. */
            if (gate_enabled) {
                kipp_spec_gate_tick(&gate);
            }
            *out_plain_steps += 1;
            if (kipp_session_eval(session, &next_tok, 1, logits,
                                  KIPP_VOCAB_SIZE, error) != 0) {
                goto done;
            }
            next_tok = argmax_logits(logits);
            continue;
        }
        feed[0] = next_tok;
        memcpy(feed + 1, drafts, k * sizeof(*drafts));
        if (kipp_session_eval_n(session, feed, k + 1, rows, (uint32_t)(k + 1),
                                error) != 0) {
            goto done;
        }
        *out_drafted += k;
        *out_draft_steps += 1;
        /* Accept drafts while each equals the greedy argmax of its row. */
        size_t accepted = 0;
        int finished = 0;
        for (size_t i = 0; i < k; ++i) {
            uint32_t correct = argmax_logits(rows + i * KIPP_VOCAB_SIZE);
            if (correct != drafts[i]) {
                break;
            }
            if (kipp_model_is_stop_token(model, drafts[i])) {
                finished = 1;
                ++accepted;
                break;
            }
            if (emit_token(model, drafts[i], out_decoded, error) != 0) {
                goto done;
            }
            hist[hlen++] = drafts[i];
            ++accepted;
            *out_accepted += 1;
            if (*out_decoded >= decode_count) {
                finished = 1;
                break;
            }
        }
        /* The verify appended next_tok + all k drafts to the KV; keep only
         * next_tok + the accepted drafts and drop the rest. */
        kipp_session_info info;
        if (kipp_session_get_info(session, &info) != 0) {
            goto done;
        }
        uint32_t keep = info.length - (uint32_t)(k - accepted);
        if (kipp_session_truncate(session, keep, error) != 0) {
            goto done;
        }
        if (gate_enabled) {
            kipp_spec_gate_record(&gate, (uint32_t)k, (uint32_t)accepted);
        }
        if (finished) {
            break;
        }
        /* Next token: the correction at the first mismatch, or the bonus
         * argmax after the last accepted draft. */
        next_tok = argmax_logits(rows + accepted * KIPP_VOCAB_SIZE);
    }
    status = 0;

done:
    free(hist);
    free(rows);
    return status;
}

static void usage(const char *program) {
    fprintf(stderr,
            "Kipp %s\n"
            "Usage: %s --model MODEL.gguf (--prompt TEXT | --ppl FILE)\n"
            "  [--backend cpu|metal|cuda]  execution backend (default cpu)\n"
            "  [--decode N]                greedy/sampled tokens to generate\n"
            "  [--temperature F]           0 = greedy argmax (default)\n"
            "  [--top-p F]                 nucleus mass, 0 < F <= 1 "
            "(default 1)\n"
            "  [--top-k N]                 keep N highest logits "
            "(0 = off)\n"
            "  [--min-p F]                 min prob vs top, 0 <= F <= 1 "
            "(0 = off)\n"
            "  [--seed N]                  nonzero sampling seed "
            "(default 1)\n"
            "  [--top N]                   final top-N logits to print "
            "(default 10)\n"
            "  [--spec]                    greedy self-speculative decoding\n"
            "  [--spec-gate on|off]        adaptive speculation gate "
            "(default on)\n"
            "  [--ppl FILE]                perplexity over LE uint32 tokens\n"
            "  [--ppl-window N]            perplexity window length "
            "(default 2048)\n"
            "  [--ppl-limit N]             score at most N tokens "
            "(0 = all)\n"
            "Decoding stops early when the model emits <|endoftext|>.\n",
            KIPP_VERSION, program);
}

static int parse_float(const char *text, float *result) {
    char *end = NULL;
    errno = 0;
    float value = strtof(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(value)) {
        return -1;
    }
    *result = value;
    return 0;
}

static int parse_count(const char *text, size_t maximum, size_t *result) {
    char *end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value > maximum) {
        return -1;
    }
    *result = (size_t)value;
    return 0;
}

static double elapsed_seconds(struct timespec start, struct timespec finish) {
    return (double)(finish.tv_sec - start.tv_sec) +
           (double)(finish.tv_nsec - start.tv_nsec) / 1000000000.0;
}

/*
 * Tokens evaluated per multi-row call while scoring perplexity. Must not
 * exceed the Metal multi-row token cap (KIPP_METAL_BATCH in
 * src/metal/kipp_metal.m: multi-row logits require token_count <= 32).
 */
#define KIPP_CLI_PPL_CHUNK 32u

/*
 * Perplexity over a little-endian uint32 token file: non-overlapping windows
 * of `window` tokens, a fresh session context per window, no burn-in. Each
 * window scores every position whose target lies inside the same window, so
 * a full window of W tokens scores W-1 targets. This protocol is simple and
 * deterministic but is not numerically comparable to llama.cpp's default
 * perplexity output, which skips a burn-in prefix per chunk.
 */
static int run_perplexity(kipp_model *model, const kipp_model_info *info,
                          const char *path, uint32_t window, size_t limit) {
    FILE *file = fopen(path, "rb");
    uint8_t *raw = NULL;
    uint32_t *stream = NULL;
    float *logits = NULL;
    kipp_session *session = NULL;
    kipp_error error = {0};
    struct timespec started;
    struct timespec finished;
    int status = -1;
    if (file == NULL) {
        fprintf(stderr, "kipp: ppl: open '%s': %s\n", path, strerror(errno));
        return -1;
    }
    if (fseek(file, 0L, SEEK_END) != 0) {
        fprintf(stderr, "kipp: ppl: seek '%s': %s\n", path, strerror(errno));
        goto done;
    }
    long file_size = ftell(file);
    if (file_size < 0 || fseek(file, 0L, SEEK_SET) != 0) {
        fprintf(stderr, "kipp: ppl: seek '%s': %s\n", path, strerror(errno));
        goto done;
    }
    if (file_size == 0 || file_size % 4 != 0) {
        fprintf(stderr, "kipp: ppl: '%s' is not a uint32 token file\n", path);
        goto done;
    }
    size_t token_count = (size_t)file_size / 4;
    raw = malloc((size_t)file_size);
    if (raw == NULL ||
        fread(raw, 1, (size_t)file_size, file) != (size_t)file_size) {
        fprintf(stderr, "kipp: ppl: unable to read '%s'\n", path);
        goto done;
    }
    if (limit != 0 && limit < token_count) {
        token_count = limit;
    }
    if (token_count < 2) {
        fprintf(stderr, "kipp: ppl: need at least two tokens to score\n");
        goto done;
    }
    stream = malloc(token_count * sizeof(*stream));
    if (stream == NULL) {
        fprintf(stderr, "kipp: ppl: unable to allocate the token stream\n");
        goto done;
    }
    for (size_t index = 0; index < token_count; ++index) {
        const uint8_t *bytes = raw + index * 4;
        stream[index] = (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
                        ((uint32_t)bytes[2] << 16) |
                        ((uint32_t)bytes[3] << 24);
        if (stream[index] >= KIPP_VOCAB_SIZE) {
            fprintf(stderr, "kipp: ppl: token %zu is out of range\n", index);
            goto done;
        }
    }
    logits = malloc((size_t)KIPP_CLI_PPL_CHUNK * KIPP_VOCAB_SIZE *
                    sizeof(*logits));
    if (logits == NULL) {
        fprintf(stderr, "kipp: ppl: unable to allocate logits\n");
        goto done;
    }
    if (kipp_session_create(model, window, &session, &error) != 0) {
        fprintf(stderr, "kipp: ppl: session: %s\n", error.message);
        goto done;
    }

    size_t window_total = (token_count + window - 1) / window;
    size_t scored = 0;
    double nll = 0.0;
    (void)clock_gettime(CLOCK_MONOTONIC, &started);
    for (size_t window_index = 0; window_index < window_total;
         ++window_index) {
        size_t window_start = window_index * window;
        size_t window_length = token_count - window_start;
        if (window_length > window) {
            window_length = window;
        }
        if (window_length < 2) {
            break; /* A one-token tail has nothing to score. */
        }
        if (kipp_session_reset(session, &error) != 0) {
            fprintf(stderr, "kipp: ppl: reset: %s\n", error.message);
            goto done;
        }
        for (size_t chunk_start = 0; chunk_start < window_length;
             chunk_start += KIPP_CLI_PPL_CHUNK) {
            size_t chunk_length = window_length - chunk_start;
            if (chunk_length > KIPP_CLI_PPL_CHUNK) {
                chunk_length = KIPP_CLI_PPL_CHUNK;
            }
            /* Scored evaluation: rows held to backend tolerance, which
             * lets Metal keep its matrix kernels instead of the bitwise
             * decode-order vector path speculation requires. */
            if (kipp_session_eval_scored(session,
                                         stream + window_start + chunk_start,
                                         chunk_length, logits,
                                         (uint32_t)chunk_length,
                                         &error) != 0) {
                fprintf(stderr, "kipp: ppl: eval: %s\n", error.message);
                goto done;
            }
            /* Row r predicts the token after chunk position r; its target is
             * the next stream token, scored only when it stays inside this
             * window. */
            for (size_t row = 0; row < chunk_length; ++row) {
                size_t target = chunk_start + row + 1;
                if (target >= window_length) {
                    continue;
                }
                const float *row_logits = logits + row * KIPP_VOCAB_SIZE;
                float maximum = row_logits[0];
                for (uint32_t token = 1; token < KIPP_VOCAB_SIZE; ++token) {
                    if (row_logits[token] > maximum) {
                        maximum = row_logits[token];
                    }
                }
                double sum = 0.0;
                for (uint32_t token = 0; token < KIPP_VOCAB_SIZE; ++token) {
                    sum += exp((double)row_logits[token] - (double)maximum);
                }
                nll += (double)maximum + log(sum) -
                       (double)row_logits[stream[window_start + target]];
                ++scored;
            }
        }
        fprintf(stderr, "ppl: window %u/%u ppl=%.4f\n",
                (unsigned)(window_index + 1), (unsigned)window_total,
                exp(nll / (double)scored));
    }
    (void)clock_gettime(CLOCK_MONOTONIC, &finished);
    fprintf(stderr,
            "KIPP_PPL backend=%s tokens=%zu scored=%zu window=%u "
            "nll=%.9f ppl=%.6f seconds=%.3f\n",
            kipp_backend_name(info->backend), token_count, scored, window,
            nll, exp(nll / (double)scored),
            elapsed_seconds(started, finished));
    status = 0;

done:
    kipp_session_destroy(session);
    free(logits);
    free(stream);
    free(raw);
    (void)fclose(file);
    return status;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *prompt = NULL;
    size_t top_count = 10;
    size_t decode_count = 0;
    float temperature = 0.0f;
    float top_p = 1.0f;
    size_t top_k = 0;
    float min_p = 0.0f;
    int spec = 0;
    int spec_gate = 1;
    const char *ppl_path = NULL;
    size_t ppl_window = 2048;
    size_t ppl_limit = 0;
    uint64_t rng_state = 1;
    kipp_backend_kind backend = KIPP_BACKEND_CPU;
    kipp_model *model = NULL;
    kipp_session *session = NULL;
    kipp_tokens tokens = {0};
    kipp_error error = {0};
    float *logits = NULL;
    uint32_t *top_ids = NULL;
    struct timespec started;
    struct timespec finished;
    double prefill_seconds = 0.0;
    double decode_seconds = 0.0;
    int exit_code = 1;

    for (int index = 1; index < argc; ++index) {
        if (strcmp(argv[index], "--model") == 0 && index + 1 < argc) {
            model_path = argv[++index];
        } else if (strcmp(argv[index], "--prompt") == 0 && index + 1 < argc) {
            prompt = argv[++index];
        } else if (strcmp(argv[index], "--top") == 0 && index + 1 < argc &&
                   parse_count(argv[++index], 100, &top_count) == 0 &&
                   top_count != 0) {
            continue;
        } else if (strcmp(argv[index], "--decode") == 0 &&
                   index + 1 < argc &&
                   parse_count(argv[++index], 1u << 20,
                               &decode_count) == 0) {
            continue;
        } else if (strcmp(argv[index], "--temperature") == 0 &&
                   index + 1 < argc &&
                   parse_float(argv[++index], &temperature) == 0 &&
                   temperature >= 0.0f) {
            continue;
        } else if (strcmp(argv[index], "--top-p") == 0 && index + 1 < argc &&
                   parse_float(argv[++index], &top_p) == 0 &&
                   top_p > 0.0f && top_p <= 1.0f) {
            continue;
        } else if (strcmp(argv[index], "--top-k") == 0 && index + 1 < argc &&
                   parse_count(argv[++index], KIPP_VOCAB_SIZE, &top_k) == 0) {
            continue;
        } else if (strcmp(argv[index], "--min-p") == 0 && index + 1 < argc &&
                   parse_float(argv[++index], &min_p) == 0 &&
                   min_p >= 0.0f && min_p <= 1.0f) {
            continue;
        } else if (strcmp(argv[index], "--spec") == 0) {
            spec = 1;
            continue;
        } else if (strcmp(argv[index], "--spec-gate") == 0 &&
                   index + 1 < argc) {
            const char *mode = argv[++index];
            if (strcmp(mode, "on") == 0) {
                spec_gate = 1;
            } else if (strcmp(mode, "off") == 0) {
                spec_gate = 0;
            } else {
                usage(argv[0]);
                goto cleanup;
            }
            continue;
        } else if (strcmp(argv[index], "--ppl") == 0 && index + 1 < argc) {
            ppl_path = argv[++index];
        } else if (strcmp(argv[index], "--ppl-window") == 0 &&
                   index + 1 < argc &&
                   parse_count(argv[++index], 1u << 20, &ppl_window) == 0 &&
                   ppl_window >= 2) {
            continue;
        } else if (strcmp(argv[index], "--ppl-limit") == 0 &&
                   index + 1 < argc &&
                   parse_count(argv[++index], SIZE_MAX, &ppl_limit) == 0) {
            continue;
        } else if (strcmp(argv[index], "--seed") == 0 && index + 1 < argc) {
            size_t seed = 0;
            if (parse_count(argv[++index], SIZE_MAX, &seed) != 0 ||
                seed == 0) {
                usage(argv[0]);
                goto cleanup;
            }
            rng_state = (uint64_t)seed;
            continue;
        } else if (strcmp(argv[index], "--backend") == 0 &&
                   index + 1 < argc) {
            const char *name = argv[++index];
            if (strcmp(name, "cpu") == 0) {
                backend = KIPP_BACKEND_CPU;
            } else if (strcmp(name, "metal") == 0) {
                backend = KIPP_BACKEND_METAL;
            } else if (strcmp(name, "cuda") == 0) {
                backend = KIPP_BACKEND_CUDA;
            } else {
                usage(argv[0]);
                goto cleanup;
            }
            continue;
        } else {
            usage(argv[0]);
            goto cleanup;
        }
    }
    if (model_path == NULL || (prompt == NULL && ppl_path == NULL)) {
        usage(argv[0]);
        goto cleanup;
    }
    if (ppl_path != NULL &&
        (prompt != NULL || decode_count != 0 || spec)) {
        fprintf(stderr,
                "kipp: --ppl cannot be combined with --prompt, --decode, or "
                "--spec\n");
        goto cleanup;
    }
    if (kipp_model_open_backend(model_path, backend, &model, &error) != 0) {
        fprintf(stderr, "kipp: %s: %s\n", kipp_error_code_name(error.code),
                error.message);
        goto cleanup;
    }
    kipp_model_info info;
    if (kipp_model_get_info(model, &info) != 0) {
        fprintf(stderr, "kipp: unable to read model info\n");
        goto cleanup;
    }
    if (ppl_path != NULL) {
        if (ppl_window > info.context_length) {
            fprintf(stderr, "kipp: --ppl-window must be between 2 and %u\n",
                    info.context_length);
            goto cleanup;
        }
        if (run_perplexity(model, &info, ppl_path, (uint32_t)ppl_window,
                           ppl_limit) != 0) {
            goto cleanup;
        }
        exit_code = 0;
        goto cleanup;
    }
    if (kipp_tokenize(model, prompt, &tokens, &error) != 0) {
        fprintf(stderr, "kipp: tokenize: %s\n", error.message);
        goto cleanup;
    }
    if (tokens.count == 0) {
        fprintf(stderr, "kipp: prompt produced no tokens\n");
        goto cleanup;
    }
    /* Speculation transiently appends up to KIPP_CLI_MAX_DRAFT draft tokens
     * beyond the logical length before rolling them back, so the KV cache
     * needs that much headroom (clamped to the context length). */
    size_t session_capacity = tokens.count + decode_count;
    if (spec) {
        session_capacity += KIPP_CLI_MAX_DRAFT + 1;
        if (session_capacity > info.context_length) {
            session_capacity = info.context_length;
        }
    }
    if (tokens.count > info.context_length ||
        decode_count > info.context_length - tokens.count ||
        kipp_session_create(model, (uint32_t)session_capacity, &session,
                            &error) != 0) {
        fprintf(stderr, "kipp: session: %s\n", error.message);
        goto cleanup;
    }
    logits = malloc(KIPP_VOCAB_SIZE * sizeof(*logits));
    top_ids = malloc(top_count * sizeof(*top_ids));
    if (logits == NULL || top_ids == NULL) {
        fprintf(stderr, "kipp: unable to allocate logits\n");
        goto cleanup;
    }

    fprintf(stderr, "kipp: evaluating %zu token%s of %s on %s\n",
            tokens.count, tokens.count == 1 ? "" : "s", info.checkpoint_id,
            kipp_backend_name(backend));
    (void)clock_gettime(CLOCK_MONOTONIC, &started);
    if (kipp_session_eval(session, tokens.data, tokens.count, logits,
                          KIPP_VOCAB_SIZE, &error) != 0) {
        fprintf(stderr, "kipp: eval: %s\n", error.message);
        goto cleanup;
    }
    (void)clock_gettime(CLOCK_MONOTONIC, &finished);
    prefill_seconds = elapsed_seconds(started, finished);

    size_t decoded_count = 0;
    uint64_t spec_drafted = 0;
    uint64_t spec_accepted = 0;
    uint64_t spec_draft_steps = 0;
    uint64_t spec_plain_steps = 0;
    if (decode_count != 0 && spec) {
        if (temperature > 0.0f) {
            fprintf(stderr,
                    "kipp: --spec requires greedy decoding (temperature 0)\n");
            goto cleanup;
        }
        fputs("generated\t", stdout);
        (void)clock_gettime(CLOCK_MONOTONIC, &started);
        if (run_spec_decode(model, session, logits, tokens.data, tokens.count,
                            decode_count, spec_gate, &decoded_count,
                            &spec_drafted, &spec_accepted, &spec_draft_steps,
                            &spec_plain_steps, &error) != 0) {
            fprintf(stderr, "kipp: spec decode: %s\n", error.message);
            goto cleanup;
        }
        (void)clock_gettime(CLOCK_MONOTONIC, &finished);
        decode_seconds = elapsed_seconds(started, finished);
        putchar('\n');
    } else if (decode_count != 0) {
        fputs("generated\t", stdout);
        (void)clock_gettime(CLOCK_MONOTONIC, &started);
        for (size_t index = 0; index < decode_count; ++index) {
            uint32_t token;
            kipp_sample_params sp = {0};
            sp.temperature = temperature;
            sp.top_p = top_p;
            sp.top_k = (uint32_t)top_k;
            sp.min_p = min_p;
            sp.repetition_penalty = 1.0f;
            if (kipp_sample_ex(logits, KIPP_VOCAB_SIZE, &sp, &rng_state,
                               &token, &error) != 0) {
                fprintf(stderr, "kipp: sample: %s\n", error.message);
                goto cleanup;
            }
            if (kipp_model_is_stop_token(model, token)) {
                break;
            }
            char *piece = NULL;
            size_t piece_length = 0;
            if (kipp_detokenize(model, &token, 1, &piece, &piece_length,
                                &error) != 0) {
                fprintf(stderr, "kipp: detokenize: %s\n", error.message);
                goto cleanup;
            }
            (void)fwrite(piece, 1, piece_length, stdout);
            (void)fflush(stdout);
            kipp_text_free(piece);
            ++decoded_count;
            if (kipp_session_eval(session, &token, 1, logits,
                                  KIPP_VOCAB_SIZE, &error) != 0) {
                fprintf(stderr, "kipp: decode: %s\n", error.message);
                goto cleanup;
            }
        }
        (void)clock_gettime(CLOCK_MONOTONIC, &finished);
        decode_seconds = elapsed_seconds(started, finished);
        putchar('\n');
    }

    for (size_t rank = 0; rank < top_count; ++rank) {
        uint32_t best = UINT32_MAX;
        for (uint32_t token = 0; token < KIPP_VOCAB_SIZE; ++token) {
            int already_selected = 0;
            for (size_t previous = 0; previous < rank; ++previous) {
                if (top_ids[previous] == token) {
                    already_selected = 1;
                    break;
                }
            }
            if (!already_selected &&
                (best == UINT32_MAX || logits[token] > logits[best])) {
                best = token;
            }
        }
        top_ids[rank] = best;
        char *piece = NULL;
        size_t piece_length = 0;
        if (kipp_detokenize(model, &best, 1, &piece, &piece_length, &error) != 0) {
            fprintf(stderr, "kipp: detokenize: %s\n", error.message);
            goto cleanup;
        }
        printf("%zu\t%u\t%.9g\t", rank + 1, best, logits[best]);
        (void)fwrite(piece, 1, piece_length, stdout);
        putchar('\n');
        kipp_text_free(piece);
    }
    fprintf(stderr,
            "KIPP_METRIC backend=%s prefill_tokens=%zu "
            "prefill_seconds=%.9f decode_tokens=%zu decode_seconds=%.9f\n",
            kipp_backend_name(backend), tokens.count, prefill_seconds,
            decoded_count, decode_seconds);
    if (spec) {
        fprintf(stderr,
                "KIPP_SPEC drafted=%llu accepted=%llu accept_rate=%.3f "
                "draft_steps=%llu plain_steps=%llu\n",
                (unsigned long long)spec_drafted,
                (unsigned long long)spec_accepted,
                spec_drafted == 0
                    ? 0.0
                    : (double)spec_accepted / (double)spec_drafted,
                (unsigned long long)spec_draft_steps,
                (unsigned long long)spec_plain_steps);
    }
    exit_code = 0;

cleanup:
    free(top_ids);
    free(logits);
    kipp_tokens_free(&tokens);
    kipp_session_destroy(session);
    if (kipp_model_close(model, &error) != 0) {
        fprintf(stderr, "kipp: close: %s\n", error.message);
        exit_code = 1;
    }
    return exit_code;
}
