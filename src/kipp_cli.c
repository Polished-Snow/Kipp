#if !defined(__APPLE__) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include "kipp.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void usage(const char *program) {
    fprintf(stderr,
            "Kipp %s\n"
            "Usage: %s --model MODEL.gguf --prompt TEXT\n"
            "  [--backend cpu|metal|cuda]  execution backend (default cpu)\n"
            "  [--decode N]                greedy/sampled tokens to generate\n"
            "  [--temperature F]           0 = greedy argmax (default)\n"
            "  [--top-p F]                 nucleus mass, 0 < F <= 1 "
            "(default 1)\n"
            "  [--seed N]                  nonzero sampling seed "
            "(default 1)\n"
            "  [--top N]                   final top-N logits to print "
            "(default 10)\n"
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

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *prompt = NULL;
    size_t top_count = 10;
    size_t decode_count = 0;
    float temperature = 0.0f;
    float top_p = 1.0f;
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
                   parse_count(argv[++index], KIPP_CONTEXT_LENGTH,
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
    if (model_path == NULL || prompt == NULL) {
        usage(argv[0]);
        goto cleanup;
    }
    if (kipp_model_open_backend(model_path, backend, &model, &error) != 0) {
        fprintf(stderr, "kipp: %s: %s\n", kipp_error_code_name(error.code),
                error.message);
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
    if (tokens.count > KIPP_CONTEXT_LENGTH ||
        decode_count > KIPP_CONTEXT_LENGTH - tokens.count ||
        kipp_session_create(model, (uint32_t)(tokens.count + decode_count),
                            &session,
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

    fprintf(stderr, "kipp: evaluating %zu token%s on %s\n", tokens.count,
            tokens.count == 1 ? "" : "s", kipp_backend_name(backend));
    (void)clock_gettime(CLOCK_MONOTONIC, &started);
    if (kipp_session_eval(session, tokens.data, tokens.count, logits,
                          KIPP_VOCAB_SIZE, &error) != 0) {
        fprintf(stderr, "kipp: eval: %s\n", error.message);
        goto cleanup;
    }
    (void)clock_gettime(CLOCK_MONOTONIC, &finished);
    prefill_seconds = elapsed_seconds(started, finished);

    size_t decoded_count = 0;
    if (decode_count != 0) {
        fputs("generated\t", stdout);
        (void)clock_gettime(CLOCK_MONOTONIC, &started);
        for (size_t index = 0; index < decode_count; ++index) {
            uint32_t token;
            if (kipp_sample(logits, KIPP_VOCAB_SIZE, temperature, top_p,
                            &rng_state, &token, &error) != 0) {
                fprintf(stderr, "kipp: sample: %s\n", error.message);
                goto cleanup;
            }
            if (token == KIPP_EOS_TOKEN_ID) {
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
