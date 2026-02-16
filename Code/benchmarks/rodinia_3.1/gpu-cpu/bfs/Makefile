include ../../common/make.config

NVCC = $(CUDA_DIR)/bin/nvcc
INCLUDE = $(CUDA_DIR)/include

SRC = bfs_gpu_cpu.cu
EXE = bfs_gpu_cpu

all: $(EXE)

$(EXE): $(SRC)
	$(NVCC) -Xcompiler -fopenmp $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

clean:
	rm -f $(EXE) *.linkinfo
