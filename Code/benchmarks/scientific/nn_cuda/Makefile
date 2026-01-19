include ../common/make.config

LOCAL_CC = gcc -g -O3 -Wall
CC := $(CUDA_DIR)/bin/nvcc

# OpenMP flags for hybrid CUDA+OpenMP compilation
OPENMP_FLAGS = -Xcompiler -fopenmp

all : nn nn_gpu_cpu hurricane_gen

clean :
	rm -rf *.o nn nn_gpu_cpu hurricane_gen

nn : nn_cuda.cu
	$(CC) -cuda nn_cuda.cu
	$(CC) -o nn nn_cuda.cu

# GPU-CPU mode: CUDA for memory management, OpenMP for computation
nn_gpu_cpu : nn_gpu_cpu.cu
	$(CC) $(OPENMP_FLAGS) -o nn_gpu_cpu nn_gpu_cpu.cu -lgomp

hurricane_gen : hurricane_gen.c
	$(LOCAL_CC) -o $@ $< -lm

#data :
#	mkdir data
#	./gen_dataset.sh
