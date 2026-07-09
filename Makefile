.PHONY: all metal cuda-spark cuda-generic cpu clean

BUILD_DIR := build

all:
	@echo "Kipp is a scaffold; select an explicit backend target."

metal:
	@echo "Metal backend is not implemented yet."

cuda-spark:
	@echo "CUDA Spark backend is not implemented yet."

cuda-generic:
	@echo "Generic CUDA backend is not implemented yet."

cpu:
	@mkdir -p $(BUILD_DIR)
	$(CC) -std=c11 -Wall -Wextra -Wpedantic -c src/kipp.c -o $(BUILD_DIR)/kipp.o
	$(CC) -std=c11 -Wall -Wextra -Wpedantic -c tests/kipp_test.c -o $(BUILD_DIR)/kipp_test.o
	@echo "Built CPU reference-path placeholders only."

clean:
	rm -rf $(BUILD_DIR)
