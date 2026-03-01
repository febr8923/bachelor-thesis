/*
 * nn_gpu_cpu.cu
 * GPU-CPU Execution variant for Nearest Neighbor
 * Same structure as nn_openmp.c but copies data to GPU and back before
 * CPU computation. Measures data transfer time, computation time, and
 * total time separately.
 *
 * Usage: ./nn_gpu_cpu <filelist> <num> <target latitude> <target longitude>
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <cuda.h>

#define MAX_ARGS 10
#define REC_LENGTH 49	// size of a record in db
#define REC_WINDOW 10	// number of records to read at a time
#define LATITUDE_POS 28	// location of latitude coordinates in input record
#define OPEN 10000	// initial value of nearest neighbors

struct neighbor {
	char entry[REC_LENGTH];
	double dist;
};

int main(int argc, char* argv[]) {
    FILE   *flist,*fp;
	int    i=0,j=0, k=0, rec_count=0, done=0;
	char   sandbox[REC_LENGTH * REC_WINDOW], *rec_iter;
	char   dbname[64];
	struct neighbor *neighbors = NULL;
	float target_lat, target_long;

	if(argc < 5) {
		fprintf(stderr, "Invalid set of arguments\n");
		exit(-1);
	}

	flist = fopen(argv[1], "r");
	if(!flist) {
		printf("error opening flist\n");
		exit(1);
	}

	k = atoi(argv[2]);
	target_lat = atof(argv[3]);
	target_long = atof(argv[4]);

	neighbors = (struct neighbor*) malloc(k*sizeof(struct neighbor));

	if(neighbors == NULL) {
		fprintf(stderr, "no room for neighbors\n");
		exit(0);
	}

	for( j = 0 ; j < k ; j++ ) {
		neighbors[j].dist = OPEN;
	}

	/**** main processing ****/
	if(fscanf(flist, "%s\n", dbname) != 1) {
		fprintf(stderr, "error reading filelist\n");
		exit(0);
	}

	fp = fopen(dbname, "r");
	if(!fp) {
		printf("error opening flist\n");
		exit(1);
	}

	float *z;
	z  = (float *) malloc(REC_WINDOW * sizeof(float));

	// ===================== GPU MEMORY ALLOCATION =====================
	char *d_sandbox;
	float *d_z;
	cudaMalloc((void **) &d_sandbox, REC_LENGTH * REC_WINDOW);
	cudaMalloc((void **) &d_z, REC_WINDOW * sizeof(float));

	// ===================== START TIMERS =====================
	double total_start_time = omp_get_wtime();
	double transfer_time_accum = 0.0;
	double compute_time_accum = 0.0;

	while(!done) {
		//Read in REC_WINDOW number of records
		rec_count = fread(sandbox, REC_LENGTH, REC_WINDOW, fp);
		if( rec_count != REC_WINDOW ) {
			if(!ferror(flist)) {// an eof occured
				fclose(fp);

				if(feof(flist))
					done = 1;
				else {
					if(fscanf(flist, "%s\n", dbname) != 1) {
						fprintf(stderr, "error reading filelist\n");
						exit(0);
					}

					fp = fopen(dbname, "r");

					if(!fp) {
						printf("error opening a db\n");
						exit(1);
					}
				}
			} else {
				perror("Error");
				exit(0);
			}
		}

		// ===================== GPU ROUND-TRIP (H2D then D2H) =====================
		double t0 = omp_get_wtime();

		cudaMemcpy(d_sandbox, sandbox, REC_LENGTH * rec_count, cudaMemcpyHostToDevice);
		cudaMemcpy(sandbox, d_sandbox, REC_LENGTH * rec_count, cudaMemcpyDeviceToHost);

		double t1 = omp_get_wtime();
		transfer_time_accum += (t1 - t0);

		// ===================== CPU COMPUTATION (OpenMP) =====================
		double c0 = omp_get_wtime();

		#pragma omp parallel for shared(z, target_lat, target_long) private(i, rec_iter)
		for (i = 0; i < rec_count; i++){
			rec_iter = sandbox+(i * REC_LENGTH + LATITUDE_POS - 1);
			float tmp_lat = atof(rec_iter);
			float tmp_long = atof(rec_iter+5);
			z[i] = sqrt(( (tmp_lat-target_lat) * (tmp_lat-target_lat) )+( (tmp_long-target_long) * (tmp_long-target_long) ));
		}
		#pragma omp barrier

		for( i = 0 ; i < rec_count ; i++ ) {
			float max_dist = -1;
			int max_idx = 0;
			for( j = 0 ; j < k ; j++ ) {
				if( neighbors[j].dist > max_dist ) {
					max_dist = neighbors[j].dist;
					max_idx = j;
				}
			}
			if( z[i] < neighbors[max_idx].dist ) {
				sandbox[(i+1)*REC_LENGTH-1] = '\0';
				strcpy(neighbors[max_idx].entry, sandbox +i*REC_LENGTH);
				neighbors[max_idx].dist = z[i];
			}
		}

		double c1 = omp_get_wtime();
		compute_time_accum += (c1 - c0);
	}//End while loop

	double total_end_time = omp_get_wtime();

	fprintf(stderr, "The %d nearest neighbors are:\n", k);
	for( j = 0 ; j < k ; j++ ) {
		if( !(neighbors[j].dist == OPEN) )
			fprintf(stderr, "%s --> %f\n", neighbors[j].entry, neighbors[j].dist);
	}

	fclose(flist);

	// ===================== PRINT TIMING RESULTS =====================
	printf("\n===== GPU-CPU Execution Timing =====\n");
	printf("Data transfer time (D2H): %lf seconds\n", transfer_time_accum);
	printf("Computation time (CPU):   %lf seconds\n", compute_time_accum);
	printf("Total time:               %lf seconds\n", total_end_time - total_start_time);

	// cleanup
	free(neighbors);
	free(z);
	cudaFree(d_sandbox);
	cudaFree(d_z);

	return 0;
}
