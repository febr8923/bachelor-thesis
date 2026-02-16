include ../../common/make.config

NVCC = $(CUDA_DIR)/bin/nvcc
INCLUDE = $(CUDA_DIR)/include

SRC = bfs_gpu_gpu.cu
EXE = bfs_gpu_gpu

all: $(EXE)

$(EXE): $(SRC) kernel.cu kernel2.cu
	$(NVCC) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

clean:
	rm -f $(EXE) *.linkinfo
