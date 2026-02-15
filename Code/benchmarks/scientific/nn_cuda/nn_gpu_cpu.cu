/*
 * nn_gpu_cpu.cu
 * Nearest Neighbor - GPU-CPU Mode
 *
 * This variant simulates the gpu-cpu scenario:
 * - Data is initially allocated on GPU memory
 * - Data is transferred from GPU to CPU
 * - Computation happens on CPU using OpenMP
 *
 * The CPU computation matches nn_openmp_walltime.c exactly:
 * same record format, same chunked processing, same neighbor update logic.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include "cuda.h"

// These constants match nn_openmp_walltime.c exactly
#define REC_LENGTH 49	// size of a record in db
#define REC_WINDOW 10	// number of records to read at a time
#define LATITUDE_POS 28	// location of latitude coordinates in input record
#define OPEN 10000	// initial value of nearest neighbors

struct neighbor {
	char entry[REC_LENGTH];
	double dist;
};

// Helper function to get time in milliseconds
double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000.0) + (tv.tv_usec / 1000.0);
}

// Load all raw record data from files into a flat buffer
// Returns total number of records loaded
int loadRawData(char *filelist, char **out_buffer) {
    FILE *flist, *fp;
    char dbname[64];
    int totalRecords = 0;
    int bufferCapacity = 1024;
    char *buffer = (char *)malloc(bufferCapacity * REC_LENGTH);

    flist = fopen(filelist, "r");
    if (!flist) {
        printf("error opening flist\n");
        exit(1);
    }

    // First file
    if (fscanf(flist, "%s\n", dbname) != 1) {
        fprintf(stderr, "error reading filelist\n");
        exit(0);
    }

    fp = fopen(dbname, "r");
    if (!fp) {
        printf("error opening flist\n");
        exit(1);
    }

    int done = 0;
    while (!done) {
        // Ensure buffer has room for REC_WINDOW more records
        if (totalRecords + REC_WINDOW > bufferCapacity) {
            bufferCapacity *= 2;
            buffer = (char *)realloc(buffer, bufferCapacity * REC_LENGTH);
        }

        // Read REC_WINDOW records at a time (same as cpu-cpu)
        int rec_count = fread(buffer + totalRecords * REC_LENGTH, REC_LENGTH, REC_WINDOW, fp);
        totalRecords += rec_count;

        if (rec_count != REC_WINDOW) {
            if (!ferror(flist)) {
                fclose(fp);
                if (feof(flist))
                    done = 1;
                else {
                    if (fscanf(flist, "%s\n", dbname) != 1) {
                        fprintf(stderr, "error reading filelist\n");
                        exit(0);
                    }
                    fp = fopen(dbname, "r");
                    if (!fp) {
                        printf("error opening a db\n");
                        exit(1);
                    }
                }
            } else {
                perror("Error");
                exit(0);
            }
        }
    }

    fclose(flist);
    *out_buffer = buffer;
    return totalRecords;
}

void printUsage() {
    printf("Nearest Neighbor Usage (GPU-CPU Mode)\n");
    printf("\n");
    printf("nn_gpu_cpu [filename] -r [int] -lat [float] -lng [float] [-hqt] [-p [int] -d [int]]\n");
    printf("\n");
    printf("filename     the filename that lists the data input files\n");
    printf("-r [int]     the number of records to return (default: 10)\n");
    printf("-lat [float] the latitude for nearest neighbors (default: 0)\n");
    printf("-lng [float] the longitude for nearest neighbors (default: 0)\n");
    printf("\n");
}

int parseCommandline(int argc, char *argv[], char* filename, int *r, float *lat, float *lng,
                     int *q, int *t, int *p, int *d) {
    int i;
    if (argc < 2) return 1;
    strncpy(filename, argv[1], 100);
    char flag;

    for (i = 1; i < argc; i++) {
        if (argv[i][0] == '-') {
            flag = argv[i][1];
            switch (flag) {
                case 'r': i++; *r = atoi(argv[i]); break;
                case 'l':
                    if (argv[i][2] == 'a') *lat = atof(argv[i+1]);
                    else *lng = atof(argv[i+1]);
                    i++;
                    break;
                case 'h': return 1;
                case 'q': *q = 1; break;
                case 't': *t = 1; break;
                case 'p': i++; *p = atoi(argv[i]); break;
                case 'd': i++; *d = atoi(argv[i]); break;
            }
        }
    }
    if ((*d >= 0 && *p < 0) || (*p >= 0 && *d < 0))
        return 1;
    return 0;
}

int main(int argc, char* argv[])
{
	int i = 0, j = 0, k = 0;
	float target_lat, target_long;
	int quiet = 0, timing = 0, platform = 0, device = 0;
	char filename[100];
	int resultsCount = 10;

	// Parse command line
	if (parseCommandline(argc, argv, filename, &resultsCount, &target_lat, &target_long,
	                     &quiet, &timing, &platform, &device)) {
		printUsage();
		return 0;
	}

	k = resultsCount;

	// === Load raw record data from files (not timed) ===
	char *h_raw_records = NULL;
	int numRecords = loadRawData(filename, &h_raw_records);
	int bufferSize = numRecords * REC_LENGTH;

	// Allocate neighbor array (same as cpu-cpu)
	struct neighbor *neighbors = (struct neighbor *)malloc(k * sizeof(struct neighbor));
	if (neighbors == NULL) {
		fprintf(stderr, "no room for neighbors\n");
		exit(0);
	}
	for (j = 0; j < k; j++) {
		neighbors[j].dist = OPEN;
	}

	// Create CUDA events for GPU timing
	cudaEvent_t transfer_to_gpu_start, transfer_to_gpu_end;
	cudaEvent_t transfer_to_cpu_start, transfer_to_cpu_end;
	cudaEventCreate(&transfer_to_gpu_start);
	cudaEventCreate(&transfer_to_gpu_end);
	cudaEventCreate(&transfer_to_cpu_start);
	cudaEventCreate(&transfer_to_cpu_end);

	// Allocate GPU memory for raw record buffer
	char *d_raw_records;
	cudaMalloc((void **)&d_raw_records, bufferSize);

	// Allocate host memory for receiving data from GPU
	char *h_raw_records_from_gpu = (char *)malloc(bufferSize);

	// === PHASE 1: Simulate data being on GPU ===
	cudaEventRecord(transfer_to_gpu_start, 0);
	cudaMemcpy(d_raw_records, h_raw_records, bufferSize, cudaMemcpyHostToDevice);
	cudaEventRecord(transfer_to_gpu_end, 0);
	cudaEventSynchronize(transfer_to_gpu_end);

	// Start total time measurement (covers only transfer + computation)
	double total_start = get_time_ms();

	// === PHASE 2: GPU->CPU Transfer (THIS IS WHAT WE MEASURE) ===
	cudaEventRecord(transfer_to_cpu_start, 0);
	cudaMemcpy(h_raw_records_from_gpu, d_raw_records, bufferSize, cudaMemcpyDeviceToHost);
	cudaEventRecord(transfer_to_cpu_end, 0);
	cudaEventSynchronize(transfer_to_cpu_end);

	// === PHASE 3: CPU Computation using OpenMP ===
	// This matches nn_openmp_walltime.c exactly: chunked processing with
	// atof parsing, parallel distance computation, serial neighbor update.
	double compute_start = get_time_ms();

	float *z = (float *)malloc(REC_WINDOW * sizeof(float));
	int total_processed = 0;

	while (total_processed < numRecords) {
		// Determine how many records in this chunk
		int rec_count = REC_WINDOW;
		if (total_processed + rec_count > numRecords)
			rec_count = numRecords - total_processed;

		char *sandbox = h_raw_records_from_gpu + total_processed * REC_LENGTH;

		// Parallel distance computation (same as cpu-cpu)
		#pragma omp parallel for shared(z, target_lat, target_long) private(i)
		for (i = 0; i < rec_count; i++) {
			char *rec_iter = sandbox + (i * REC_LENGTH + LATITUDE_POS - 1);
			float tmp_lat = atof(rec_iter);
			float tmp_long = atof(rec_iter + 5);
			z[i] = sqrt(((tmp_lat - target_lat) * (tmp_lat - target_lat)) +
			            ((tmp_long - target_long) * (tmp_long - target_long)));
		}
		#pragma omp barrier

		// Serial neighbor update (same as cpu-cpu)
		for (i = 0; i < rec_count; i++) {
			float max_dist = -1;
			int max_idx = 0;
			for (j = 0; j < k; j++) {
				if (neighbors[j].dist > max_dist) {
					max_dist = neighbors[j].dist;
					max_idx = j;
				}
			}
			if (z[i] < neighbors[max_idx].dist) {
				sandbox[(i + 1) * REC_LENGTH - 1] = '\0';
				strcpy(neighbors[max_idx].entry, sandbox + i * REC_LENGTH);
				neighbors[max_idx].dist = z[i];
			}
		}

		total_processed += rec_count;
	}

	double compute_end = get_time_ms();

	// End total time measurement
	double total_end = get_time_ms();

	// Calculate elapsed times
	float transfer_to_gpu_time = 0.0f;
	float transfer_to_cpu_time = 0.0f;
	cudaEventElapsedTime(&transfer_to_gpu_time, transfer_to_gpu_start, transfer_to_gpu_end);
	cudaEventElapsedTime(&transfer_to_cpu_time, transfer_to_cpu_start, transfer_to_cpu_end);
	double computation_time = compute_end - compute_start;
	double total_time = total_end - total_start;

	// Print timing results
	printf("Total time:                %.3f ms\n", total_time);
	printf("GPU->CPU data transfer:    %.3f ms\n", transfer_to_cpu_time);
	printf("CPU computation (OpenMP):  %.3f ms\n", computation_time);

	// Destroy CUDA events
	cudaEventDestroy(transfer_to_gpu_start);
	cudaEventDestroy(transfer_to_gpu_end);
	cudaEventDestroy(transfer_to_cpu_start);
	cudaEventDestroy(transfer_to_cpu_end);

	free(z);
	free(neighbors);
	free(h_raw_records);
	free(h_raw_records_from_gpu);
	cudaFree(d_raw_records);

	return 0;
}
