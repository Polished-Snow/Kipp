.PHONY: all cpu server server-metal server-cuda test test-tools test-sanitize \
	tools-env model convert vectors chat-vectors test-model test-ppl test-phase2 \
	test-paged-cpu test-pooled-cpu test-multilogit metal test-metal-ops \
	test-multilogit-metal test-paged-metal test-pooled-metal test-metal \
	test-server \
	cuda-spark cuda-generic \
	test-cuda-ops test-cuda docs docs-check paper-data paper-check clean

BUILD_DIR := build
TOOLS_DIR := tools
# Any id from the supported-checkpoint registry (src/kipp_checkpoints.h).
CHECKPOINT ?= qwen3-4b-base
CHAT_CHECKPOINT ?= qwen3-4b-instruct-2507
MODEL_DIR := models/$(CHECKPOINT)
MODEL_SOURCE := $(MODEL_DIR)/source
MODEL_GGUF := $(MODEL_DIR)/kipp-$(CHECKPOINT)-bf16.gguf
VECTOR_DIR := tests/test-vectors/$(CHECKPOINT)
CHAT_SOURCE := models/$(CHAT_CHECKPOINT)/source
CHAT_VECTOR := tests/test-vectors/$(CHAT_CHECKPOINT)/chat-cases.json

CPPFLAGS := -Isrc
CFLAGS := -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror
LDLIBS := -lm
METAL_FRAMEWORKS := -framework Foundation -framework Metal
NVCC ?= nvcc
NVCCFLAGS := -std=c++17 -O2 -I./src -I./cuda \
	-Xcompiler=-Wall,-Wextra,-Werror
CUDA_GENERIC_ARCH_FLAGS ?= \
	-gencode arch=compute_80,code=sm_80 \
	-gencode arch=compute_86,code=sm_86 \
	-gencode arch=compute_89,code=sm_89 \
	-gencode arch=compute_90,code=sm_90 \
	-gencode arch=compute_80,code=compute_80
CUDA_SPARK_ARCH_FLAGS ?= -gencode arch=compute_121,code=sm_121
CORE_HEADERS := src/kipp.h src/kipp_backend.h src/kipp_checkpoints.h \
	src/kipp_kv_pool.h src/kipp_unicode.inc
CLI_OBJECTS := $(BUILD_DIR)/kipp_cli.o $(BUILD_DIR)/kipp_spec.o \
	$(BUILD_DIR)/kipp_kv_pool.o
SERVER_OBJECTS := $(BUILD_DIR)/kipp_server.o $(BUILD_DIR)/kipp_chat.o \
	$(BUILD_DIR)/kipp_json.o $(BUILD_DIR)/kipp_http.o \
	$(BUILD_DIR)/kipp_kv_pool.o
TEST_SUPPORT_SOURCES := src/kipp_chat.c src/kipp_spec.c src/kipp_kv_pool.c \
	src/kipp_json.c src/kipp_http.c

all: cpu

cpu: $(BUILD_DIR)/kipp

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/kipp.o: src/kipp.c $(CORE_HEADERS) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/kipp.c -o $@

$(BUILD_DIR)/kipp_cli.o: src/kipp_cli.c src/kipp.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/kipp_cli.c -o $@

$(BUILD_DIR)/kipp: $(BUILD_DIR)/kipp.o $(CLI_OBJECTS)
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/kipp_server.o: src/kipp_server.c src/kipp.h src/kipp_chat.h \
		src/kipp_json.h src/kipp_http.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/kipp_server.c -o $@

$(BUILD_DIR)/kipp-server: $(BUILD_DIR)/kipp.o $(SERVER_OBJECTS)
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/kipp-server-metal: $(BUILD_DIR)/kipp_metal_core.o \
		$(SERVER_OBJECTS) $(BUILD_DIR)/kipp_metal_bridge.o
	xcrun --sdk macosx clang $(CFLAGS) $^ $(LDLIBS) \
		$(METAL_FRAMEWORKS) -o $@

server: $(BUILD_DIR)/kipp-server

server-metal: $(BUILD_DIR)/kipp-server-metal

server-cuda: $(BUILD_DIR)/kipp-server-cuda

$(BUILD_DIR)/kipp_metal_source.inc: metal/kipp_kernels.metal \
		tools/embed_metal_source.py | $(BUILD_DIR)
	python3 tools/embed_metal_source.py $< $@

$(BUILD_DIR)/kipp_metal_core.o: src/kipp.c $(CORE_HEADERS) \
		src/metal/kipp_metal.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DKIPP_ENABLE_METAL -c src/kipp.c -o $@

$(BUILD_DIR)/kipp_metal_bridge.o: src/metal/kipp_metal.m \
		src/metal/kipp_metal.h $(BUILD_DIR)/kipp_metal_source.inc | $(BUILD_DIR)
	xcrun --sdk macosx clang $(CPPFLAGS) -I$(BUILD_DIR) $(CFLAGS) \
		-fobjc-arc -fblocks -c src/metal/kipp_metal.m -o $@

$(BUILD_DIR)/kipp-metal: $(BUILD_DIR)/kipp_metal_core.o \
		$(CLI_OBJECTS) $(BUILD_DIR)/kipp_metal_bridge.o
	xcrun --sdk macosx clang $(CFLAGS) $^ $(LDLIBS) \
		$(METAL_FRAMEWORKS) -o $@

$(BUILD_DIR)/kipp_test: src/kipp.c $(TEST_SUPPORT_SOURCES) \
		$(CORE_HEADERS) src/kipp_chat.h \
		src/kipp_spec.h src/kipp_kv_pool.h src/kipp_unicode.inc \
		tests/kipp_test.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DKIPP_TESTING \
		src/kipp.c $(TEST_SUPPORT_SOURCES) \
		tests/kipp_test.c $(LDLIBS) -o $@

# Mutation-study binary: the CPU oracle with KIPP_FAULT-selected seeded
# paging bugs (see kipp.c's KIPP_FAULT_INJECT block). Research tooling only —
# never a production or default-test target.
$(BUILD_DIR)/kipp_test_fault: src/kipp.c $(TEST_SUPPORT_SOURCES) \
		$(CORE_HEADERS) src/kipp_chat.h \
		src/kipp_spec.h src/kipp_kv_pool.h src/kipp_unicode.inc \
		tests/kipp_test.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DKIPP_TESTING -DKIPP_FAULT_INJECT \
		src/kipp.c $(TEST_SUPPORT_SOURCES) \
		tests/kipp_test.c $(LDLIBS) -o $@

$(BUILD_DIR)/kipp_test_metal: src/kipp.c src/kipp_chat.c src/kipp.h \
		src/kipp_backend.h src/kipp_checkpoints.h src/kipp_chat.h \
		src/kipp_kv_pool.h src/metal/kipp_metal.h src/kipp_unicode.inc \
		tests/kipp_test.c \
		$(BUILD_DIR)/kipp_metal_bridge.o | $(BUILD_DIR)
	xcrun --sdk macosx clang $(CPPFLAGS) $(CFLAGS) \
		-DKIPP_TESTING -DKIPP_ENABLE_METAL src/kipp.c src/kipp_chat.c \
		src/kipp_spec.c src/kipp_kv_pool.c src/kipp_json.c \
		src/kipp_http.c tests/kipp_test.c \
		$(BUILD_DIR)/kipp_metal_bridge.o $(LDLIBS) $(METAL_FRAMEWORKS) -o $@

test: $(BUILD_DIR)/kipp_test test-tools
	$(BUILD_DIR)/kipp_test

test-sanitize: | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -std=c11 -O1 -g -Wall -Wextra -Wpedantic -Werror \
		-DKIPP_TESTING -fsanitize=address,undefined \
		src/kipp.c $(TEST_SUPPORT_SOURCES) \
		tests/kipp_test.c $(LDLIBS) -o $(BUILD_DIR)/kipp_test_sanitize
	$(BUILD_DIR)/kipp_test_sanitize

tools-env:
	uv sync --project $(TOOLS_DIR) --python 3.12 --locked

docs: tools-env
	uv run --project $(TOOLS_DIR) --python 3.12 \
		python tools/render_docs.py

docs-check: tools-env
	uv run --project $(TOOLS_DIR) --python 3.12 \
		python tools/render_docs.py --check

test-tools: docs-check
	uv run --project $(TOOLS_DIR) --python 3.12 \
		python -m unittest tests/test_tooling.py
	python3 tools/paper_data.py --check

paper-data:
	python3 tools/paper_data.py

paper-check:
	python3 tools/paper_data.py --check

model: tools-env
	tools/download_model.sh --checkpoint $(CHECKPOINT)

convert: model
	uv run --project $(TOOLS_DIR) --python 3.12 \
		python tools/convert_to_gguf.py --checkpoint $(CHECKPOINT) \
		--source $(MODEL_SOURCE) --output $(MODEL_GGUF)

vectors: convert
	uv run --project $(TOOLS_DIR) --python 3.12 \
		python tools/generate_test_vectors.py --checkpoint $(CHECKPOINT) \
		--source $(MODEL_SOURCE) --gguf $(MODEL_GGUF) \
		--output $(VECTOR_DIR)

chat-vectors: tools-env
	tools/download_model.sh --checkpoint $(CHAT_CHECKPOINT)
	uv run --project $(TOOLS_DIR) --python 3.12 \
		python tools/generate_chat_vectors.py \
		--checkpoint $(CHAT_CHECKPOINT) --source $(CHAT_SOURCE) \
		--output $(CHAT_VECTOR)

test-model: $(BUILD_DIR)/kipp_test vectors
	$(BUILD_DIR)/kipp_test --model $(MODEL_GGUF) $(VECTOR_DIR)

# Smoke-check the CLI perplexity mode against the pinned prompt tokens
# (already LE uint32, the --ppl input format). Model-gated like test-model.
test-ppl: $(BUILD_DIR)/kipp vectors
	$(BUILD_DIR)/kipp --model $(MODEL_GGUF) \
		--ppl $(VECTOR_DIR)/tokens.u32 --ppl-window 128 2>&1 | \
		grep -E 'KIPP_PPL .* ppl=[0-9]+\.[0-9]+' >/dev/null && \
		echo "PASS test-ppl"

test-phase2: test-paged-cpu test-multilogit
	$(BUILD_DIR)/kipp_test --phase2-model $(MODEL_GGUF) $(VECTOR_DIR)

test-paged-cpu: $(BUILD_DIR)/kipp_test vectors
	$(BUILD_DIR)/kipp_test --paged-cpu $(MODEL_GGUF) $(VECTOR_DIR)

test-pooled-cpu: $(BUILD_DIR)/kipp_test vectors
	$(BUILD_DIR)/kipp_test --pooled-cpu $(MODEL_GGUF) $(VECTOR_DIR)

test-multilogit: $(BUILD_DIR)/kipp_test vectors
	$(BUILD_DIR)/kipp_test --multilogit $(MODEL_GGUF) $(VECTOR_DIR)

metal: $(BUILD_DIR)/kipp-metal

test-metal-ops: $(BUILD_DIR)/kipp_test_metal
	$(BUILD_DIR)/kipp_test_metal --metal-operators

test-multilogit-metal: $(BUILD_DIR)/kipp_test_metal vectors
	$(BUILD_DIR)/kipp_test_metal --multilogit-metal $(MODEL_GGUF) $(VECTOR_DIR)

test-paged-metal: $(BUILD_DIR)/kipp_test_metal vectors
	$(BUILD_DIR)/kipp_test_metal --paged-metal $(MODEL_GGUF) $(VECTOR_DIR)

test-pooled-metal: $(BUILD_DIR)/kipp_test_metal vectors
	$(BUILD_DIR)/kipp_test_metal --pooled-metal $(MODEL_GGUF) $(VECTOR_DIR)

test-metal: test-metal-ops test-multilogit-metal test-paged-metal
	$(BUILD_DIR)/kipp_test_metal --phase3-metal $(MODEL_GGUF) $(VECTOR_DIR)

test-server: $(BUILD_DIR)/kipp-server-metal
	KIPP_SERVER_BINARY=$(BUILD_DIR)/kipp-server-metal \
	KIPP_SERVER_MODEL=$(MODEL_GGUF) \
	KIPP_SERVER_MODEL_ID=$(CHECKPOINT) \
	python3 -m unittest tests/test_server.py

$(BUILD_DIR)/kipp_cuda_core.o: src/kipp.c $(CORE_HEADERS) \
		src/cuda/kipp_cuda.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DKIPP_ENABLE_CUDA -c src/kipp.c -o $@

$(BUILD_DIR)/kipp_cuda_bridge_generic.o: src/cuda/kipp_cuda.cu \
		src/cuda/kipp_cuda.h cuda/kipp_kernels.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CUDA_GENERIC_ARCH_FLAGS) \
		-c src/cuda/kipp_cuda.cu -o $@

$(BUILD_DIR)/kipp_cuda_kernels_generic.o: cuda/kipp_kernels.cu \
		cuda/kipp_kernels.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CUDA_GENERIC_ARCH_FLAGS) \
		-c cuda/kipp_kernels.cu -o $@

$(BUILD_DIR)/kipp-cuda-generic: $(BUILD_DIR)/kipp_cuda_core.o \
		$(CLI_OBJECTS) $(BUILD_DIR)/kipp_cuda_bridge_generic.o \
		$(BUILD_DIR)/kipp_cuda_kernels_generic.o
	$(NVCC) $(NVCCFLAGS) $(CUDA_GENERIC_ARCH_FLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/kipp_cuda_bridge_spark.o: src/cuda/kipp_cuda.cu \
		src/cuda/kipp_cuda.h cuda/kipp_kernels.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CUDA_SPARK_ARCH_FLAGS) \
		-c src/cuda/kipp_cuda.cu -o $@

$(BUILD_DIR)/kipp_cuda_kernels_spark.o: cuda/kipp_kernels.cu \
		cuda/kipp_kernels.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CUDA_SPARK_ARCH_FLAGS) \
		-c cuda/kipp_kernels.cu -o $@

$(BUILD_DIR)/kipp-cuda-spark: $(BUILD_DIR)/kipp_cuda_core.o \
		$(CLI_OBJECTS) $(BUILD_DIR)/kipp_cuda_bridge_spark.o \
		$(BUILD_DIR)/kipp_cuda_kernels_spark.o
	$(NVCC) $(NVCCFLAGS) $(CUDA_SPARK_ARCH_FLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/kipp_cuda_test_core.o: src/kipp.c src/kipp.h \
		src/kipp_backend.h src/kipp_checkpoints.h src/cuda/kipp_cuda.h src/kipp_unicode.inc | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DKIPP_TESTING -DKIPP_ENABLE_CUDA \
		-c src/kipp.c -o $@

$(BUILD_DIR)/kipp_cuda_test.o: tests/kipp_test.c src/kipp.h src/kipp_chat.h \
		src/cuda/kipp_cuda.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DKIPP_TESTING -DKIPP_ENABLE_CUDA \
		-c tests/kipp_test.c -o $@

$(BUILD_DIR)/kipp_chat.o: src/kipp_chat.c src/kipp_chat.h src/kipp.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/kipp_chat.c -o $@

$(BUILD_DIR)/kipp_json.o: src/kipp_json.c src/kipp_json.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/kipp_json.c -o $@

$(BUILD_DIR)/kipp_http.o: src/kipp_http.c src/kipp_http.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/kipp_http.c -o $@

$(BUILD_DIR)/kipp_spec.o: src/kipp_spec.c src/kipp_spec.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/kipp_spec.c -o $@

$(BUILD_DIR)/kipp_kv_pool.o: src/kipp_kv_pool.c src/kipp_kv_pool.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/kipp_kv_pool.c -o $@

$(BUILD_DIR)/kipp-server-cuda: $(BUILD_DIR)/kipp_cuda_core.o \
		$(SERVER_OBJECTS) $(BUILD_DIR)/kipp_cuda_bridge_generic.o \
		$(BUILD_DIR)/kipp_cuda_kernels_generic.o
	$(NVCC) $(NVCCFLAGS) $(CUDA_GENERIC_ARCH_FLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/kipp_test_cuda: $(BUILD_DIR)/kipp_cuda_test_core.o \
		$(BUILD_DIR)/kipp_cuda_test.o $(BUILD_DIR)/kipp_chat.o \
		$(BUILD_DIR)/kipp_spec.o $(BUILD_DIR)/kipp_kv_pool.o \
		$(BUILD_DIR)/kipp_json.o $(BUILD_DIR)/kipp_http.o \
		$(BUILD_DIR)/kipp_cuda_bridge_generic.o \
		$(BUILD_DIR)/kipp_cuda_kernels_generic.o
	$(NVCC) $(NVCCFLAGS) $(CUDA_GENERIC_ARCH_FLAGS) $^ $(LDLIBS) -o $@

cuda-generic: $(BUILD_DIR)/kipp-cuda-generic

cuda-spark: $(BUILD_DIR)/kipp-cuda-spark

test-cuda-ops: $(BUILD_DIR)/kipp_test_cuda
	$(BUILD_DIR)/kipp_test_cuda --cuda-operators

test-cuda: $(BUILD_DIR)/kipp_test_cuda vectors
	$(BUILD_DIR)/kipp_test_cuda --cuda-operators
	$(BUILD_DIR)/kipp_test_cuda --phase4-cuda $(MODEL_GGUF) $(VECTOR_DIR)

clean:
	rm -rf $(BUILD_DIR)
