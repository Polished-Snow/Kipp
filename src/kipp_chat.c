#include "kipp_chat.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* A minimal growable byte buffer, local to this module. */
typedef struct {
    char *data;
    size_t length;
    size_t capacity;
    bool failed;
} chat_builder;

static void chat_append(chat_builder *builder, const char *text) {
    if (builder->failed || text == NULL) {
        return;
    }
    size_t add = strlen(text);
    if (builder->length + add + 1 > builder->capacity) {
        size_t capacity = builder->capacity == 0 ? 256 : builder->capacity;
        while (capacity < builder->length + add + 1) {
            if (capacity > SIZE_MAX / 2) {
                builder->failed = true;
                return;
            }
            capacity *= 2;
        }
        char *resized = realloc(builder->data, capacity);
        if (resized == NULL) {
            builder->failed = true;
            return;
        }
        builder->data = resized;
        builder->capacity = capacity;
    }
    memcpy(builder->data + builder->length, text, add);
    builder->length += add;
    builder->data[builder->length] = '\0';
}

static const char *role_name(kipp_chat_role role) {
    switch (role) {
    case KIPP_ROLE_SYSTEM:
        return "system";
    case KIPP_ROLE_USER:
        return "user";
    case KIPP_ROLE_ASSISTANT:
        return "assistant";
    }
    return NULL;
}

static void append_message(chat_builder *builder,
                           const kipp_chat_message *message) {
    chat_append(builder, "<|im_start|>");
    chat_append(builder, role_name(message->role));
    chat_append(builder, "\n");
    chat_append(builder, message->content != NULL ? message->content : "");
    chat_append(builder, "<|im_end|>\n");
}

int kipp_chat_render(const kipp_chat_message *messages, size_t message_count,
                     const kipp_chat_options *options, char **out_text,
                     kipp_error *error) {
    if (error != NULL) {
        error->code = KIPP_OK;
        error->message[0] = '\0';
    }
    if (messages == NULL || options == NULL || out_text == NULL ||
        message_count == 0) {
        if (error != NULL) {
            error->code = KIPP_ERROR_ARGUMENT;
            (void)snprintf(error->message, sizeof(error->message),
                           "messages and options are required");
        }
        return -1;
    }
    *out_text = NULL;
    if (options->variant == KIPP_VARIANT_BASE) {
        if (error != NULL) {
            error->code = KIPP_ERROR_UNSUPPORTED;
            (void)snprintf(error->message, sizeof(error->message),
                           "base checkpoints have no chat template");
        }
        return -1;
    }
    for (size_t index = 0; index < message_count; ++index) {
        if (role_name(messages[index].role) == NULL) {
            if (error != NULL) {
                error->code = KIPP_ERROR_ARGUMENT;
                (void)snprintf(error->message, sizeof(error->message),
                               "message %zu has an invalid role", index);
            }
            return -1;
        }
    }

    chat_builder builder = {0};
    for (size_t index = 0; index < message_count; ++index) {
        append_message(&builder, &messages[index]);
    }
    if (options->add_generation_prompt) {
        chat_append(&builder, "<|im_start|>assistant\n");
        /*
         * Variant-specific thinking suffix on the generation prompt:
         *  - hybrid instruct: empty <think></think> when thinking is off;
         *  - thinking-2507: a forced opening <think> the model completes;
         *  - non-thinking 2507: nothing (matches HF, which ignores the flag).
         */
        if (options->variant == KIPP_VARIANT_INSTRUCT &&
            !options->enable_thinking) {
            chat_append(&builder, "<think>\n\n</think>\n\n");
        } else if (options->variant == KIPP_VARIANT_THINKING_2507) {
            chat_append(&builder, "<think>\n");
        }
    }

    if (builder.failed) {
        free(builder.data);
        if (error != NULL) {
            error->code = KIPP_ERROR_MEMORY;
            (void)snprintf(error->message, sizeof(error->message),
                           "unable to allocate chat prompt");
        }
        return -1;
    }
    /* An empty render (no messages appended anything) still yields "". */
    *out_text = builder.data != NULL ? builder.data : calloc(1, 1);
    if (*out_text == NULL) {
        if (error != NULL) {
            error->code = KIPP_ERROR_MEMORY;
        }
        return -1;
    }
    return 0;
}
