# C compiler
CC = g++
CC_FLAGS = -g -fopenmp -O2

all: bfs

bfs: bfs.cpp
	$(CC) $(CC_FLAGS) bfs.cpp -o bfs

# bfs_offload requires Intel compiler (icc) for OpenMP offloading
# Uncomment and set ICC if available:
# ICC = icc
# bfs_offload: bfs.cpp
# 	$(ICC) $(CC_FLAGS) -DOMP_OFFLOAD bfs.cpp -o bfs_offload

clean:
	rm -f bfs bfs_offload result.txt
