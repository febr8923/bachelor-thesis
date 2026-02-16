include ../../common/make.config

# C compiler
CC = gcc
CC_FLAGS = -g -O3 -Wall -I$(MATRIX_DIR) -I$(CUDA_LEUK_DIR)

# CUDA compiler
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -I$(CUDA_DIR)/include

ifeq ($(dbg),1)
	NVCC_FLAGS += -g -O0
else
	NVCC_FLAGS += -O3
endif

CUDA_LEUK_DIR = ../../cuda/leukocyte/CUDA
MATRIX_DIR = ../../cuda/leukocyte/meschach_lib

EXE = leukocyte_gpu_gpu

OBJS = detect_main_gpu_gpu.o avilib.o find_ellipse.o find_ellipse_kernel.o \
       track_ellipse.o track_ellipse_kernel.o misc_math.o

all: $(MATRIX_DIR)/meschach.a $(EXE)

$(EXE): $(OBJS) $(MATRIX_DIR)/meschach.a
	$(NVCC) -Xcompiler "$(CC_FLAGS)" $(OBJS) -o $(EXE) $(MATRIX_DIR)/meschach.a -lm

detect_main_gpu_gpu.o: detect_main_gpu_gpu.c $(CUDA_LEUK_DIR)/find_ellipse.h $(CUDA_LEUK_DIR)/track_ellipse.h
	$(CC) $(CC_FLAGS) detect_main_gpu_gpu.c -c

avilib.o: $(CUDA_LEUK_DIR)/avilib.c
	$(CC) $(CC_FLAGS) $(CUDA_LEUK_DIR)/avilib.c -c

find_ellipse.o: $(CUDA_LEUK_DIR)/find_ellipse.c
	$(CC) $(CC_FLAGS) $(CUDA_LEUK_DIR)/find_ellipse.c -c

find_ellipse_kernel.o: $(CUDA_LEUK_DIR)/find_ellipse_kernel.cu
	$(NVCC) $(NVCC_FLAGS) -c $(CUDA_LEUK_DIR)/find_ellipse_kernel.cu

track_ellipse.o: $(CUDA_LEUK_DIR)/track_ellipse.c
	$(CC) $(CC_FLAGS) $(CUDA_LEUK_DIR)/track_ellipse.c -c

track_ellipse_kernel.o: $(CUDA_LEUK_DIR)/track_ellipse_kernel.cu
	$(NVCC) $(NVCC_FLAGS) -I$(MATRIX_DIR) -c $(CUDA_LEUK_DIR)/track_ellipse_kernel.cu

misc_math.o: $(CUDA_LEUK_DIR)/misc_math.c
	$(CC) $(CC_FLAGS) $(CUDA_LEUK_DIR)/misc_math.c -c

$(MATRIX_DIR)/meschach.a:
	cd $(MATRIX_DIR); ./configure --with-all; make all; make clean

clean:
	rm -f *.o *~ $(EXE) *.linkinfo
