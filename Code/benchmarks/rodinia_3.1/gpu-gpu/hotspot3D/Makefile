include ../../common/make.config

NVCC = $(CUDA_DIR)/bin/nvcc
INCLUDE = $(CUDA_DIR)/include

SRC = 3D_gpu_gpu.cu
EXE = 3D_gpu_gpu

all: $(EXE)

$(EXE): $(SRC)
	$(NVCC) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lm

clean:
	rm -f $(EXE) *.linkinfo *.out
