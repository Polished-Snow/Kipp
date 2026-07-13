.PHONY: all cpu server server-metal test test-tools test-sanitize tools-env \
	model convert vectors test-model test-phase2 metal test-metal-ops \
	test-metal test-server cuda-spark cuda-generic test-cuda-ops test-cuda \
	clean

BUILD_DIR := build
TOOLS_DIR := tools
MODEL_DIR := models/qwen3-4b-base
MODEL_SOURCE := $(MODEL_DIR)/source
MODEL_GGUF := $(MODEL_DIR)/kipp-qwen3-4b-base-bf16.gguf
VECTOR_DIR := tests/test-vectors/qwen3-4b-base

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

all: cpu

cpu: $(BUILD_DIR)/kipp

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/kipp.o: src/kipp.c src/kipp.h src/kipp_backend.h \
		src/kipp_unicode.inc | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/kipp.c -o $@

$(BUILD_DIR)/kipp_cli.o: src/kipp_cli.c src/kipp.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/kipp_cli.c -o $@

$(BUILD_DIR)/kipp: $(BUILD_DIR)/kipp.o $(BUILD_DIR)/kipp_cli.o
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/kipp_server.o: src/kipp_server.c src/kipp.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/kipp_server.c -o $@

$(BUILD_DIR)/kipp-server: $(BUILD_DIR)/kipp.o $(BUILD_DIR)/kipp_server.o
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/kipp-server-metal: $(BUILD_DIR)/kipp_metal_core.o \
		$(BUILD_DIR)/kipp_server.o $(BUILD_DIR)/kipp_metal_bridge.o
	xcrun --sdk macosx clang $(CFLAGS) $^ $(LDLIBS) \
		$(METAL_FRAMEWORKS) -o $@

server: $(BUILD_DIR)/kipp-server

server-metal: $(BUILD_DIR)/kipp-server-metal

$(BUILD_DIR)/kipp_metal_source.inc: metal/kipp_kernels.metal \
		tools/embed_metal_source.py | $(BUILD_DIR)
	python3 tools/embed_metal_source.py $< $@

$(BUILD_DIR)/kipp_metal_core.o: src/kipp.c src/kipp.h src/kipp_backend.h \
		src/metal/kipp_metal.h src/kipp_unicode.inc | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DKIPP_ENABLE_METAL -c src/kipp.c -o $@

$(BUILD_DIR)/kipp_metal_bridge.o: src/metal/kipp_metal.m \
		src/metal/kipp_metal.h $(BUILD_DIR)/kipp_metal_source.inc | $(BUILD_DIR)
	xcrun --sdk macosx clang $(CPPFLAGS) -I$(BUILD_DIR) $(CFLAGS) \
		-fobjc-arc -fblocks -c src/metal/kipp_metal.m -o $@

$(BUILD_DIR)/kipp-metal: $(BUILD_DIR)/kipp_metal_core.o \
		$(BUILD_DIR)/kipp_cli.o $(BUILD_DIR)/kipp_metal_bridge.o
	xcrun --sdk macosx clang $(CFLAGS) $^ $(LDLIBS) \
		$(METAL_FRAMEWORKS) -o $@

$(BUILD_DIR)/kipp_test: src/kipp.c src/kipp.h src/kipp_backend.h \
		src/kipp_unicode.inc tests/kipp_test.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DKIPP_TESTING \
		src/kipp.c tests/kipp_test.c $(LDLIBS) -o $@

$(BUILD_DIR)/kipp_test_metal: src/kipp.c src/kipp.h src/kipp_backend.h \
		src/metal/kipp_metal.h src/kipp_unicode.inc tests/kipp_test.c \
		$(BUILD_DIR)/kipp_metal_bridge.o | $(BUILD_DIR)
	xcrun --sdk macosx clang $(CPPFLAGS) $(CFLAGS) \
		-DKIPP_TESTING -DKIPP_ENABLE_METAL src/kipp.c tests/kipp_test.c \
		$(BUILD_DIR)/kipp_metal_bridge.o $(LDLIBS) $(METAL_FRAMEWORKS) -o $@

test: $(BUILD_DIR)/kipp_test test-tools
	$(BUILD_DIR)/kipp_test

test-tools:
	python3 -m unittest tests/test_tooling.py

test-sanitize: | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) -std=c11 -O1 -g -Wall -Wextra -Wpedantic -Werror \
		-DKIPP_TESTING -fsanitize=address,undefined \
		src/kipp.c tests/kipp_test.c $(LDLIBS) -o $(BUILD_DIR)/kipp_test_sanitize
	$(BUILD_DIR)/kipp_test_sanitize

tools-env:
	uv sync --project $(TOOLS_DIR) --python 3.12 --locked

model: tools-env
	tools/download_model.sh

convert: model
	uv run --project $(TOOLS_DIR) --python 3.12 \
		python tools/convert_to_gguf.py \
		--source $(MODEL_SOURCE) --output $(MODEL_GGUF)

vectors: convert
	uv run --project $(TOOLS_DIR) --python 3.12 \
		python tools/generate_test_vectors.py \
		--source $(MODEL_SOURCE) --gguf $(MODEL_GGUF) \
		--output $(VECTOR_DIR)

test-model: $(BUILD_DIR)/kipp_test vectors
	$(BUILD_DIR)/kipp_test --model $(MODEL_GGUF) $(VECTOR_DIR)

test-phase2: $(BUILD_DIR)/kipp_test vectors
	$(BUILD_DIR)/kipp_test --phase2-model $(MODEL_GGUF) $(VECTOR_DIR)

metal: $(BUILD_DIR)/kipp-metal

test-metal-ops: $(BUILD_DIR)/kipp_test_metal
	$(BUILD_DIR)/kipp_test_metal --metal-operators

test-metal: $(BUILD_DIR)/kipp_test_metal vectors
	$(BUILD_DIR)/kipp_test_metal --metal-operators
	$(BUILD_DIR)/kipp_test_metal --phase3-metal $(MODEL_GGUF) $(VECTOR_DIR)

test-server: $(BUILD_DIR)/kipp-server-metal
	KIPP_SERVER_BINARY=$(BUILD_DIR)/kipp-server-metal \
	KIPP_SERVER_MODEL=$(MODEL_GGUF) \
	python3 -m unittest tests/test_server.py

$(BUILD_DIR)/kipp_cuda_core.o: src/kipp.c src/kipp.h src/kipp_backend.h \
		src/cuda/kipp_cuda.h src/kipp_unicode.inc | $(BUILD_DIR)
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
		$(BUILD_DIR)/kipp_cli.o $(BUILD_DIR)/kipp_cuda_bridge_generic.o \
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
		$(BUILD_DIR)/kipp_cli.o $(BUILD_DIR)/kipp_cuda_bridge_spark.o \
		$(BUILD_DIR)/kipp_cuda_kernels_spark.o
	$(NVCC) $(NVCCFLAGS) $(CUDA_SPARK_ARCH_FLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/kipp_cuda_test_core.o: src/kipp.c src/kipp.h \
		src/kipp_backend.h src/cuda/kipp_cuda.h src/kipp_unicode.inc | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DKIPP_TESTING -DKIPP_ENABLE_CUDA \
		-c src/kipp.c -o $@

$(BUILD_DIR)/kipp_cuda_test.o: tests/kipp_test.c src/kipp.h \
		src/cuda/kipp_cuda.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DKIPP_TESTING -DKIPP_ENABLE_CUDA \
		-c tests/kipp_test.c -o $@

$(BUILD_DIR)/kipp_test_cuda: $(BUILD_DIR)/kipp_cuda_test_core.o \
		$(BUILD_DIR)/kipp_cuda_test.o $(BUILD_DIR)/kipp_cuda_bridge_generic.o \
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
