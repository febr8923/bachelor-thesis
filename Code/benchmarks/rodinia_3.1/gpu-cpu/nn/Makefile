include ../../common/make.config

NVCC = $(CUDA_DIR)/bin/nvcc
INCLUDE = $(CUDA_DIR)/include

SRC = nn_gpu_cpu.cu
EXE = nn_gpu_cpu

all: $(EXE)

$(EXE): $(SRC)
	$(NVCC) -O3 -Xcompiler -fopenmp -Xcompiler -O3 $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

clean:
	rm -f $(EXE) *.linkinfo
