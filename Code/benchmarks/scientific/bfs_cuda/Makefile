include ../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

# OpenMP flags for hybrid CUDA+OpenMP compilation
OPENMP_FLAGS = -Xcompiler -fopenmp

SRC = bfs.cu
SRC_GPU_CPU = bfs_gpu_cpu.cu

EXE = bfs
EXE_GPU_CPU = bfs_gpu_cpu

all: release gpu_cpu

release: $(SRC)
	$(CC) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

# GPU-CPU mode: CUDA for memory management, OpenMP for computation
gpu_cpu: $(SRC_GPU_CPU)
	$(CC) $(OPENMP_FLAGS) $(SRC_GPU_CPU) -o $(EXE_GPU_CPU) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lgomp

enum: $(SRC)
	$(CC) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

debug: $(SRC)
	$(CC) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

debugenum: $(SRC)
	$(CC) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

clean:
	rm -f $(EXE) $(EXE_GPU_CPU) $(EXE).linkinfo result.txt
