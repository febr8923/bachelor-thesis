LOCAL_CC = gcc -g -O3 -Wall
CC = gcc
CFLAGS = -lm -fopenmp -Wall


all : nn nn_walltime

clean :
	rm -rf *.o nn nn_walltime

nn : nn_openmp.c
	$(CC) -o $@ $< $(LDFLAGS) $(CFLAGS)

nn_walltime : nn_openmp_walltime.c
	$(CC) -o $@ $< $(LDFLAGS) $(CFLAGS)

hurricane_gen : hurricane_gen.c
	$(LOCAL_CC) -o $@ $< -lm

