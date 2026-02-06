/***********************************************************************************
  Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

  GPU-CPU Mode: Data on GPU, computation on CPU

  This variant simulates the gpu-cpu scenario:
  - Data is initially allocated on GPU memory
  - Data is transferred from GPU to CPU
  - Computation happens on CPU using OpenMP

  Copyright (c) 2008 International Institute of Information Technology - Hyderabad.
  All rights reserved.
 ************************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <sys/time.h>
#include <omp.h>

#define MAX_THREADS_PER_BLOCK 512

// Helper function to get time in milliseconds
double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000.0) + (tv.tv_usec / 1000.0);
}

int no_of_nodes;
int edge_list_size;
FILE *fp;

//Structure to hold a node information
struct Node
{
	int starting;
	int no_of_edges;
};

void BFSGraph(int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv)
{
	no_of_nodes=0;
	edge_list_size=0;
	BFSGraph( argc, argv);
}

void Usage(int argc, char**argv){
	fprintf(stderr,"Usage: %s <input_file> [num_threads]\n", argv[0]);
}

////////////////////////////////////////////////////////////////////////////////
// BFS on CPU using OpenMP (after data transfer from GPU)
////////////////////////////////////////////////////////////////////////////////
void BFS_CPU(Node* h_graph_nodes, int* h_graph_edges, bool* h_graph_mask,
             bool* h_updating_graph_mask, bool* h_graph_visited, int* h_cost,
             int no_of_nodes, int num_threads)
{
    omp_set_num_threads(num_threads);

    bool stop;
    int k = 0;

    do {
        stop = false;

        #pragma omp parallel for
        for(int tid = 0; tid < no_of_nodes; tid++) {
            if (h_graph_mask[tid] == true) {
                h_graph_mask[tid] = false;
                for(int i = h_graph_nodes[tid].starting;
                    i < (h_graph_nodes[tid].no_of_edges + h_graph_nodes[tid].starting); i++) {
                    int id = h_graph_edges[i];
                    if(!h_graph_visited[id]) {
                        h_cost[id] = h_cost[tid] + 1;
                        h_updating_graph_mask[id] = true;
                    }
                }
            }
        }

        #pragma omp parallel for
        for(int tid = 0; tid < no_of_nodes; tid++) {
            if (h_updating_graph_mask[tid] == true) {
                h_graph_mask[tid] = true;
                h_graph_visited[tid] = true;
                stop = true;
                h_updating_graph_mask[tid] = false;
            }
        }
        k++;
    } while(stop);

    printf("BFS executed %d iterations\n", k);
}

////////////////////////////////////////////////////////////////////////////////
// Apply BFS on a Graph - GPU-CPU Mode
////////////////////////////////////////////////////////////////////////////////
void BFSGraph( int argc, char** argv)
{
    char *input_f;
    int num_threads = 4;  // Default number of threads

	if(argc < 2){
        Usage(argc, argv);
        exit(0);
	}

	input_f = argv[1];
    if (argc >= 3) {
        num_threads = atoi(argv[2]);
    }

	printf("Reading File\n");
	//Read in Graph from a file
	fp = fopen(input_f,"r");
	if(!fp)
	{
		printf("Error Reading graph file\n");
		return;
	}

	int source = 0;

	fscanf(fp,"%d",&no_of_nodes);

	// allocate host memory
	Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
	bool *h_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_updating_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_graph_visited = (bool*) malloc(sizeof(bool)*no_of_nodes);

	int start, edgeno;
	// initalize the memory
	for( unsigned int i = 0; i < no_of_nodes; i++)
	{
		fscanf(fp,"%d %d",&start,&edgeno);
		h_graph_nodes[i].starting = start;
		h_graph_nodes[i].no_of_edges = edgeno;
		h_graph_mask[i]=false;
		h_updating_graph_mask[i]=false;
		h_graph_visited[i]=false;
	}

	//read the source node from the file
	fscanf(fp,"%d",&source);
	source=0;

	//set the source node as true in the mask
	h_graph_mask[source]=true;
	h_graph_visited[source]=true;

	fscanf(fp,"%d",&edge_list_size);

	int id,cost;
	int* h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
	for(int i=0; i < edge_list_size ; i++)
	{
		fscanf(fp,"%d",&id);
		fscanf(fp,"%d",&cost);
		h_graph_edges[i] = id;
	}

	if(fp)
		fclose(fp);

	printf("Read File\n");

	// allocate mem for the result on host side
	int* h_cost = (int*) malloc( sizeof(int)*no_of_nodes);
	for(int i=0;i<no_of_nodes;i++)
		h_cost[i]=-1;
	h_cost[source]=0;


	// Create CUDA events for GPU timing
	cudaEvent_t transfer_to_gpu_start, transfer_to_gpu_end;
	cudaEvent_t transfer_to_cpu_start, transfer_to_cpu_end;
	cudaEventCreate(&transfer_to_gpu_start);
	cudaEventCreate(&transfer_to_gpu_end);
	cudaEventCreate(&transfer_to_cpu_start);
	cudaEventCreate(&transfer_to_cpu_end);

	// === PHASE 1: Simulate data being on GPU ===
	// Allocate and copy data to GPU (simulating data already on GPU)
	Node* d_graph_nodes;
	int* d_graph_edges;
	bool* d_graph_mask;
	bool* d_updating_graph_mask;
	bool* d_graph_visited;
	int* d_cost;




	cudaEventRecord(transfer_to_gpu_start, 0);

	cudaMalloc( (void**) &d_graph_nodes, sizeof(Node)*no_of_nodes);
	cudaMalloc( (void**) &d_graph_edges, sizeof(int)*edge_list_size);
	cudaMalloc( (void**) &d_graph_mask, sizeof(bool)*no_of_nodes);
	cudaMalloc( (void**) &d_updating_graph_mask, sizeof(bool)*no_of_nodes);
	cudaMalloc( (void**) &d_graph_visited, sizeof(bool)*no_of_nodes);
	cudaMalloc( (void**) &d_cost, sizeof(int)*no_of_nodes);

	cudaMemcpy( d_graph_nodes, h_graph_nodes, sizeof(Node)*no_of_nodes, cudaMemcpyHostToDevice);
	cudaMemcpy( d_graph_edges, h_graph_edges, sizeof(int)*edge_list_size, cudaMemcpyHostToDevice);
	cudaMemcpy( d_graph_mask, h_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice);
	cudaMemcpy( d_updating_graph_mask, h_updating_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice);
	cudaMemcpy( d_graph_visited, h_graph_visited, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice);
	cudaMemcpy( d_cost, h_cost, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice);

	cudaEventRecord(transfer_to_gpu_end, 0);
	cudaEventSynchronize(transfer_to_gpu_end);

	printf("Data copied to GPU (simulating data origin on GPU)\n");

	// === PHASE 2: GPU->CPU Transfer (THIS IS WHAT WE MEASURE) ===

	// Start total time measurement
	double total_start = get_time_ms();
	// Allocate fresh host memory for receiving data from GPU
	Node* h_graph_nodes_from_gpu = (Node*) malloc(sizeof(Node)*no_of_nodes);
	int* h_graph_edges_from_gpu = (int*) malloc(sizeof(int)*edge_list_size);
	bool* h_graph_mask_from_gpu = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool* h_updating_graph_mask_from_gpu = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool* h_graph_visited_from_gpu = (bool*) malloc(sizeof(bool)*no_of_nodes);
	int* h_cost_from_gpu = (int*) malloc(sizeof(int)*no_of_nodes);

	cudaEventRecord(transfer_to_cpu_start, 0);

	cudaMemcpy( h_graph_nodes_from_gpu, d_graph_nodes, sizeof(Node)*no_of_nodes, cudaMemcpyDeviceToHost);
	cudaMemcpy( h_graph_edges_from_gpu, d_graph_edges, sizeof(int)*edge_list_size, cudaMemcpyDeviceToHost);
	cudaMemcpy( h_graph_mask_from_gpu, d_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyDeviceToHost);
	cudaMemcpy( h_updating_graph_mask_from_gpu, d_updating_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyDeviceToHost);
	cudaMemcpy( h_graph_visited_from_gpu, d_graph_visited, sizeof(bool)*no_of_nodes, cudaMemcpyDeviceToHost);
	cudaMemcpy( h_cost_from_gpu, d_cost, sizeof(int)*no_of_nodes, cudaMemcpyDeviceToHost);

	cudaEventRecord(transfer_to_cpu_end, 0);
	cudaEventSynchronize(transfer_to_cpu_end);

	printf("Data transferred from GPU to CPU\n");

	// === PHASE 3: CPU Computation using OpenMP ===
	printf("Start traversing the tree on CPU with %d threads\n", num_threads);

	double compute_start = get_time_ms();
	BFS_CPU(h_graph_nodes_from_gpu, h_graph_edges_from_gpu, h_graph_mask_from_gpu,
	        h_updating_graph_mask_from_gpu, h_graph_visited_from_gpu, h_cost_from_gpu,
	        no_of_nodes, num_threads);
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
	printf("\n=== TIMING RESULTS (GPU-CPU Mode) ===\n");
	printf("Total time:                %.3f ms\n", total_time);
	printf("GPU->CPU data transfer:    %.3f ms\n", transfer_to_cpu_time);
	printf("CPU computation (OpenMP):  %.3f ms\n", computation_time);
	printf("(Setup: CPU->GPU transfer: %.3f ms - not counted)\n", transfer_to_gpu_time);
	printf("=====================================\n\n");

	//Store the result into a file
	FILE *fpo = fopen("result.txt","w");
	for(int i=0;i<no_of_nodes;i++)
		fprintf(fpo,"%d) cost:%d\n",i,h_cost_from_gpu[i]);
	fclose(fpo);
	printf("Result stored in result.txt\n");

	// Destroy CUDA events
	cudaEventDestroy(transfer_to_gpu_start);
	cudaEventDestroy(transfer_to_gpu_end);
	cudaEventDestroy(transfer_to_cpu_start);
	cudaEventDestroy(transfer_to_cpu_end);

	// cleanup memory
	free( h_graph_nodes);
	free( h_graph_edges);
	free( h_graph_mask);
	free( h_updating_graph_mask);
	free( h_graph_visited);
	free( h_cost);

	free( h_graph_nodes_from_gpu);
	free( h_graph_edges_from_gpu);
	free( h_graph_mask_from_gpu);
	free( h_updating_graph_mask_from_gpu);
	free( h_graph_visited_from_gpu);
	free( h_cost_from_gpu);

	cudaFree(d_graph_nodes);
	cudaFree(d_graph_edges);
	cudaFree(d_graph_mask);
	cudaFree(d_updating_graph_mask);
	cudaFree(d_graph_visited);
	cudaFree(d_cost);
}
