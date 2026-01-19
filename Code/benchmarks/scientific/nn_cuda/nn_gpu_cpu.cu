/*
 * nn_gpu_cpu.cu
 * Nearest Neighbor - GPU-CPU Mode
 *
 * This variant simulates the gpu-cpu scenario:
 * - Data is initially allocated on GPU memory
 * - Data is transferred from GPU to CPU
 * - Computation happens on CPU using OpenMP
 *
 * This measures the GPU->CPU transfer overhead when data originates on GPU.
 */

#include <stdio.h>
#include <sys/time.h>
#include <float.h>
#include <vector>
#include <math.h>
#include <omp.h>
#include "cuda.h"

#define min( a, b )			a > b ? b : a
#define ceilDiv( a, b )		( a + b - 1 ) / b
#define print( x )			printf( #x ": %lu\n", (unsigned long) x )
#define DEBUG				false

#define MAX_ARGS 10
#define REC_LENGTH 53 // size of a record in db
#define LATITUDE_POS 28	// character position of the latitude value in each record
#define OPEN 10000	// initial value of nearest neighbors


typedef struct latLong
{
  float lat;
  float lng;
} LatLong;

typedef struct record
{
  char recString[REC_LENGTH];
  float distance;
} Record;

int loadData(char *filename,std::vector<Record> &records,std::vector<LatLong> &locations);
void findLowest(std::vector<Record> &records,float *distances,int numRecords,int topN);
void printUsage();
int parseCommandline(int argc, char *argv[], char* filename,int *r,float *lat,float *lng,
                     int *q, int *t, int *p, int *d);

// Helper function to get time in milliseconds
double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000.0) + (tv.tv_usec / 1000.0);
}

/**
 * CPU kernel using OpenMP
 * Calculates the Euclidean distance from each record to the target position
 */
void euclid_cpu(LatLong *locations, float *distances, int numRecords, float lat, float lng)
{
    #pragma omp parallel for
    for (int i = 0; i < numRecords; i++) {
        float dlat = lat - locations[i].lat;
        float dlng = lng - locations[i].lng;
        distances[i] = sqrtf(dlat * dlat + dlng * dlng);
    }
}

/**
* This program finds the k-nearest neighbors
* GPU-CPU mode: Data on GPU, computation on CPU
**/

int main(int argc, char* argv[])
{
	int    i=0;
	float lat, lng;
	int quiet=0,timing=0,platform=0,device=0;

    std::vector<Record> records;
	std::vector<LatLong> locations;
	char filename[100];
	int resultsCount=10;

    // parse command line
    if (parseCommandline(argc, argv, filename,&resultsCount,&lat,&lng,
                     &quiet, &timing, &platform, &device)) {
      printUsage();
      return 0;
    }

    int numRecords = loadData(filename,records,locations);
    if (resultsCount > numRecords) resultsCount = numRecords;

    // Pointers to host memory
	float *h_distances;
	LatLong *h_locations;

	// Pointers to device memory
	LatLong *d_locations;

	// Start total time measurement
	double total_start = get_time_ms();

	// Create CUDA events for GPU timing
	cudaEvent_t transfer_to_gpu_start, transfer_to_gpu_end;
	cudaEvent_t transfer_to_cpu_start, transfer_to_cpu_end;
	cudaEventCreate(&transfer_to_gpu_start);
	cudaEventCreate(&transfer_to_gpu_end);
	cudaEventCreate(&transfer_to_cpu_start);
	cudaEventCreate(&transfer_to_cpu_end);

	/**
	* Allocate memory on host and device
	*/
	h_distances = (float *)malloc(sizeof(float) * numRecords);
	h_locations = (LatLong *)malloc(sizeof(LatLong) * numRecords);

	// Allocate GPU memory
	cudaMalloc((void **) &d_locations, sizeof(LatLong) * numRecords);

	// === PHASE 1: Simulate data being on GPU ===
	// In a real scenario, data would already be on GPU from a previous computation.
	// We simulate this by copying data to GPU first (not timed as part of benchmark).
	cudaEventRecord(transfer_to_gpu_start, 0);
	cudaMemcpy(d_locations, &locations[0], sizeof(LatLong) * numRecords, cudaMemcpyHostToDevice);
	cudaEventRecord(transfer_to_gpu_end, 0);
	cudaEventSynchronize(transfer_to_gpu_end);

	// === PHASE 2: GPU->CPU Transfer (THIS IS WHAT WE MEASURE) ===
	cudaEventRecord(transfer_to_cpu_start, 0);
	cudaMemcpy(h_locations, d_locations, sizeof(LatLong) * numRecords, cudaMemcpyDeviceToHost);
	cudaEventRecord(transfer_to_cpu_end, 0);
	cudaEventSynchronize(transfer_to_cpu_end);

	// === PHASE 3: CPU Computation using OpenMP ===
	double compute_start = get_time_ms();
	euclid_cpu(h_locations, h_distances, numRecords, lat, lng);
	double compute_end = get_time_ms();

	// find the resultsCount least distances
    findLowest(records, h_distances, numRecords, resultsCount);

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
	printf("\n=== TIMING RESULTS (GPU-CPU Mode) ===\n");
	printf("Total time:                %.3f ms\n", total_time);
	printf("GPU->CPU data transfer:    %.3f ms\n", transfer_to_cpu_time);
	printf("CPU computation (OpenMP):  %.3f ms\n", computation_time);
	printf("(Setup: CPU->GPU transfer: %.3f ms - not counted)\n", transfer_to_gpu_time);
	printf("=====================================\n\n");

    // print out results
    if (!quiet)
    for(i=0;i<resultsCount;i++) {
      printf("%s --> Distance=%f\n",records[i].recString,records[i].distance);
    }

	// Destroy CUDA events
	cudaEventDestroy(transfer_to_gpu_start);
	cudaEventDestroy(transfer_to_gpu_end);
	cudaEventDestroy(transfer_to_cpu_start);
	cudaEventDestroy(transfer_to_cpu_end);

    free(h_distances);
    free(h_locations);
	cudaFree(d_locations);

	return 0;
}

int loadData(char *filename,std::vector<Record> &records,std::vector<LatLong> &locations){
    FILE   *flist,*fp;
	int    i=0;
	char dbname[64];
	int recNum=0;

    /**Main processing **/

    flist = fopen(filename, "r");
	while(!feof(flist)) {
		/**
		* Read in all records of length REC_LENGTH
		* If this is the last file in the filelist, then done
		* else open next file to be read next iteration
		*/
		if(fscanf(flist, "%s\n", dbname) != 1) {
            fprintf(stderr, "error reading filelist\n");
            exit(0);
        }
        fp = fopen(dbname, "r");
        if(!fp) {
            printf("error opening a db\n");
            exit(1);
        }
        // read each record
        while(!feof(fp)){
            Record record;
            LatLong latLong;
            fgets(record.recString,49,fp);
            fgetc(fp); // newline
            if (feof(fp)) break;

            // parse for lat and long
            char substr[6];

            for(i=0;i<5;i++) substr[i] = *(record.recString+i+28);
            substr[5] = '\0';
            latLong.lat = atof(substr);

            for(i=0;i<5;i++) substr[i] = *(record.recString+i+33);
            substr[5] = '\0';
            latLong.lng = atof(substr);

            locations.push_back(latLong);
            records.push_back(record);
            recNum++;
        }
        fclose(fp);
    }
    fclose(flist);
    return recNum;
}

void findLowest(std::vector<Record> &records,float *distances,int numRecords,int topN){
  int i,j;
  float val;
  int minLoc;
  Record *tempRec;
  float tempDist;

  for(i=0;i<topN;i++) {
    minLoc = i;
    for(j=i;j<numRecords;j++) {
      val = distances[j];
      if (val < distances[minLoc]) minLoc = j;
    }
    // swap locations and distances
    tempRec = &records[i];
    records[i] = records[minLoc];
    records[minLoc] = *tempRec;

    tempDist = distances[i];
    distances[i] = distances[minLoc];
    distances[minLoc] = tempDist;

    // add distance to the min we just found
    records[i].distance = distances[i];
  }
}

int parseCommandline(int argc, char *argv[], char* filename,int *r,float *lat,float *lng,
                     int *q, int *t, int *p, int *d){
    int i;
    if (argc < 2) return 1; // error
    strncpy(filename,argv[1],100);
    char flag;

    for(i=1;i<argc;i++) {
      if (argv[i][0]=='-') {// flag
        flag = argv[i][1];
          switch (flag) {
            case 'r': // number of results
              i++;
              *r = atoi(argv[i]);
              break;
            case 'l': // lat or lng
              if (argv[i][2]=='a') {//lat
                *lat = atof(argv[i+1]);
              }
              else {//lng
                *lng = atof(argv[i+1]);
              }
              i++;
              break;
            case 'h': // help
              return 1;
            case 'q': // quiet
              *q = 1;
              break;
            case 't': // timing
              *t = 1;
              break;
            case 'p': // platform
              i++;
              *p = atoi(argv[i]);
              break;
            case 'd': // device
              i++;
              *d = atoi(argv[i]);
              break;
        }
      }
    }
    if ((*d >= 0 && *p<0) || (*p>=0 && *d<0)) // both p and d must be specified if either are specified
      return 1;
    return 0;
}

void printUsage(){
  printf("Nearest Neighbor Usage (GPU-CPU Mode)\n");
  printf("\n");
  printf("nn_gpu_cpu [filename] -r [int] -lat [float] -lng [float] [-hqt] [-p [int] -d [int]]\n");
  printf("\n");
  printf("This mode simulates data starting on GPU, transferring to CPU for computation.\n");
  printf("\n");
  printf("example:\n");
  printf("$ ./nn_gpu_cpu filelist.txt -r 5 -lat 30 -lng 90\n");
  printf("\n");
  printf("filename     the filename that lists the data input files\n");
  printf("-r [int]     the number of records to return (default: 10)\n");
  printf("-lat [float] the latitude for nearest neighbors (default: 0)\n");
  printf("-lng [float] the longitude for nearest neighbors (default: 0)\n");
  printf("\n");
  printf("-h, --help   Display the help file\n");
  printf("-q           Quiet mode. Suppress all text output.\n");
  printf("-t           Print timing information.\n");
  printf("\n");
  printf("-p [int]     Choose the platform (must choose both platform and device)\n");
  printf("-d [int]     Choose the device (must choose both platform and device)\n");
  printf("\n");
}
