include ../../common/make.config

# C compiler
CC = gcc
CC_FLAGS = -g -O3 -Wall -fopenmp -I$(OMP_LEUK_DIR) -I$(MATRIX_DIR)

# CUDA compiler
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -I$(CUDA_DIR)/include -I$(OMP_LEUK_DIR) -I$(MATRIX_DIR)

OMP_LEUK_DIR = ../../openmp/leukocyte/OpenMP
MATRIX_DIR = ../../openmp/leukocyte/meschach_lib

EXE = leukocyte_gpu_cpu

OBJS = detect_main_gpu_cpu.o avilib.o find_ellipse.o track_ellipse.o misc_math.o

all: $(MATRIX_DIR)/meschach.a $(EXE)

$(EXE): $(OBJS) $(MATRIX_DIR)/meschach.a
	$(NVCC) -Xcompiler -fopenmp $(OBJS) -o $(EXE) $(MATRIX_DIR)/meschach.a -lm

detect_main_gpu_cpu.o: detect_main_gpu_cpu.cu
	$(NVCC) $(NVCC_FLAGS) -Xcompiler -fopenmp -c detect_main_gpu_cpu.cu

avilib.o: $(OMP_LEUK_DIR)/avilib.c
	$(CC) $(CC_FLAGS) -c $(OMP_LEUK_DIR)/avilib.c

find_ellipse.o: $(OMP_LEUK_DIR)/find_ellipse.c
	$(CC) $(CC_FLAGS) -c $(OMP_LEUK_DIR)/find_ellipse.c

track_ellipse.o: $(OMP_LEUK_DIR)/track_ellipse.c
	$(CC) $(CC_FLAGS) -c $(OMP_LEUK_DIR)/track_ellipse.c

misc_math.o: $(OMP_LEUK_DIR)/misc_math.c
	$(CC) $(CC_FLAGS) -c $(OMP_LEUK_DIR)/misc_math.c

$(MATRIX_DIR)/meschach.a:
	cd $(MATRIX_DIR); ./configure --with-all; make all; make clean

clean:
	rm -f *.o $(EXE) *.linkinfo
