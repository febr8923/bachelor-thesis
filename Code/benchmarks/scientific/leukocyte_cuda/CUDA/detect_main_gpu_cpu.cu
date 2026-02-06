/*
 * detect_main_gpu_cpu.cu
 * Leukocyte Detection - GPU-CPU Mode
 *
 * This variant simulates the gpu-cpu scenario:
 * - Data (image gradients) is initially allocated on GPU memory
 * - Data is transferred from GPU to CPU
 * - Computation happens on CPU using OpenMP
 *
 * This measures the GPU->CPU transfer overhead when data originates on GPU.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {
#include "avilib.h"
#include "find_ellipse.h"
#include "track_ellipse.h"
}

// The number of sample points per ellipse
#define NPOINTS 150
// The expected radius (in pixels) of a cell
#define RADIUS 10
// The range of acceptable radii
#define MIN_RAD (RADIUS - 2)
#define MAX_RAD (RADIUS * 2)
// The number of different sample ellipses to try
#define NCIRCLES 7

int omp_num_threads = 4;

// Helper function to get time in milliseconds
double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000.0) + (tv.tv_usec / 1000.0);
}

// Returns the current system time in microseconds (for compatibility)
long long get_time_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

// CPU implementation of GICOV computation using OpenMP
MAT* ellipsematching_cpu(MAT* grad_x, MAT* grad_y) {
    int i, n, k;

    // Compute the sine and cosine of the angle to each point
    double sin_angle[NPOINTS], cos_angle[NPOINTS], theta[NPOINTS];
    for (n = 0; n < NPOINTS; n++) {
        theta[n] = (double)n * 2.0 * PI / (double)NPOINTS;
        sin_angle[n] = sin(theta[n]);
        cos_angle[n] = cos(theta[n]);
    }

    // Compute the (x,y) pixel offsets of each sample point
    int tX[NCIRCLES][NPOINTS], tY[NCIRCLES][NPOINTS];
    for (k = 0; k < NCIRCLES; k++) {
        double rad = (double)(MIN_RAD + 2 * k);
        for (n = 0; n < NPOINTS; n++) {
            tX[k][n] = (int)(cos(theta[n]) * rad);
            tY[k][n] = (int)(sin(theta[n]) * rad);
        }
    }

    int MaxR = MAX_RAD + 2;
    int height = grad_x->m, width = grad_x->n;
    MAT* gicov = m_get(height, width);

    // OpenMP parallelization
    #pragma omp parallel for num_threads(omp_num_threads)
    for (i = MaxR; i < width - MaxR; i++) {
        double Grad[NPOINTS];
        int j, k, n, x, y;

        for (j = MaxR; j < height - MaxR; j++) {
            double max_GICOV = 0;

            for (k = 0; k < NCIRCLES; k++) {
                for (n = 0; n < NPOINTS; n++) {
                    y = j + tY[k][n];
                    x = i + tX[k][n];
                    Grad[n] = m_get_val(grad_x, y, x) * cos_angle[n] +
                              m_get_val(grad_y, y, x) * sin_angle[n];
                }

                double sum = 0.0;
                for (n = 0; n < NPOINTS; n++) sum += Grad[n];
                double mean = sum / (double)NPOINTS;

                double var = 0.0;
                for (n = 0; n < NPOINTS; n++) {
                    sum = Grad[n] - mean;
                    var += sum * sum;
                }
                var = var / (double)(NPOINTS - 1);

                if (var > 0 && mean * mean / var > max_GICOV) {
                    max_GICOV = mean * mean / var;
                }
            }
            m_set_val(gicov, j, i, max_GICOV);
        }
    }

    return gicov;
}

// CPU implementation of dilation using OpenMP
MAT* dilate_cpu(MAT* gicov) {
    int height = gicov->m, width = gicov->n;
    int radius = 12;
    MAT* result = m_get(height, width);

    #pragma omp parallel for num_threads(omp_num_threads)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            double max_val = m_get_val(gicov, i, j);
            for (int di = -radius; di <= radius; di++) {
                for (int dj = -radius; dj <= radius; dj++) {
                    int ni = i + di;
                    int nj = j + dj;
                    if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
                        double val = m_get_val(gicov, ni, nj);
                        if (val > max_val) max_val = val;
                    }
                }
            }
            m_set_val(result, i, j, max_val);
        }
    }

    return result;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input_file> <num_frames> [num_threads]\n", argv[0]);
        exit(1);
    }

    char* video_file_name = argv[1];
    int num_frames = atoi(argv[2]);
    if (argc >= 4) {
        omp_num_threads = atoi(argv[3]);
    }

    printf("Leukocyte GPU-CPU Mode\n");
    printf("Num threads: %d\n", omp_num_threads);

    // Open video file
    avi_t* cell_file = AVI_open_input_file(video_file_name, 1);
    if (cell_file == NULL) {
        AVI_print_error("Error with AVI_open_input_file");
        return -1;
    }

    // Get video dimensions
    int vid_width = AVI_video_width(cell_file);
    int vid_height = AVI_video_height(cell_file);
    int frame_size = vid_width * vid_height;

    printf("Video: %dx%d, %d frames\n", vid_width, vid_height, num_frames);

    // Create CUDA events for timing
    cudaEvent_t transfer_to_gpu_start, transfer_to_gpu_end;
    cudaEvent_t transfer_to_cpu_start, transfer_to_cpu_end;
    cudaEventCreate(&transfer_to_gpu_start);
    cudaEventCreate(&transfer_to_gpu_end);
    cudaEventCreate(&transfer_to_cpu_start);
    cudaEventCreate(&transfer_to_cpu_end);

    // Extract and process first frame
    printf("Detecting cells in frame 0\n");
    MAT* image_chopped = get_frame(cell_file, 0, 1, 0);

    // Compute gradients on CPU first
    MAT* grad_x = gradient_x(image_chopped);
    MAT* grad_y = gradient_y(image_chopped);
    m_free(image_chopped);

    int grad_height = grad_x->m;
    int grad_width = grad_x->n;
    int grad_size = grad_height * grad_width;

    // Flatten gradient matrices for GPU transfer
    double* h_grad_x = (double*)malloc(grad_size * sizeof(double));
    double* h_grad_y = (double*)malloc(grad_size * sizeof(double));
    for (int i = 0; i < grad_height; i++) {
        for (int j = 0; j < grad_width; j++) {
            h_grad_x[i * grad_width + j] = m_get_val(grad_x, i, j);
            h_grad_y[i * grad_width + j] = m_get_val(grad_y, i, j);
        }
    }

    // === PHASE 1: Simulate data being on GPU ===
    double* d_grad_x;
    double* d_grad_y;

    cudaEventRecord(transfer_to_gpu_start, 0);
    cudaMalloc((void**)&d_grad_x, grad_size * sizeof(double));
    cudaMalloc((void**)&d_grad_y, grad_size * sizeof(double));
    cudaMemcpy(d_grad_x, h_grad_x, grad_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_y, h_grad_y, grad_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaEventRecord(transfer_to_gpu_end, 0);
    cudaEventSynchronize(transfer_to_gpu_end);

    printf("Data copied to GPU (simulating data origin on GPU)\n");

    // Start total time measurement
    double total_start = get_time_ms();

    // === PHASE 2: GPU->CPU Transfer (THIS IS WHAT WE MEASURE) ===
    double* h_grad_x_from_gpu = (double*)malloc(grad_size * sizeof(double));
    double* h_grad_y_from_gpu = (double*)malloc(grad_size * sizeof(double));

    cudaEventRecord(transfer_to_cpu_start, 0);
    cudaMemcpy(h_grad_x_from_gpu, d_grad_x, grad_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grad_y_from_gpu, d_grad_y, grad_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventRecord(transfer_to_cpu_end, 0);
    cudaEventSynchronize(transfer_to_cpu_end);

    // Reconstruct MAT structures from transferred data
    MAT* grad_x_cpu = m_get(grad_height, grad_width);
    MAT* grad_y_cpu = m_get(grad_height, grad_width);
    for (int i = 0; i < grad_height; i++) {
        for (int j = 0; j < grad_width; j++) {
            m_set_val(grad_x_cpu, i, j, h_grad_x_from_gpu[i * grad_width + j]);
            m_set_val(grad_y_cpu, i, j, h_grad_y_from_gpu[i * grad_width + j]);
        }
    }

    printf("Data transferred from GPU to CPU\n");

    // === PHASE 3: CPU Computation using OpenMP ===
    double compute_start = get_time_ms();

    // GICOV computation
    MAT* gicov = ellipsematching_cpu(grad_x_cpu, grad_y_cpu);

    // Square GICOV values
    MAT* max_gicov = m_get(gicov->m, gicov->n);
    for (int i = 0; i < gicov->m; i++) {
        for (int j = 0; j < gicov->n; j++) {
            double val = m_get_val(gicov, i, j);
            m_set_val(max_gicov, i, j, val * val);
        }
    }

    // Dilation
    MAT* img_dilated = dilate_cpu(max_gicov);

    // Find cell centers (matching cpu-cpu version)
    int pair_counter = 0;
    int* crow = (int*)malloc(max_gicov->m * max_gicov->n * sizeof(int));
    int* ccol = (int*)malloc(max_gicov->m * max_gicov->n * sizeof(int));

    for (int i = 0; i < max_gicov->m; i++) {
        for (int j = 0; j < max_gicov->n; j++) {
            if (!(m_get_val(max_gicov, i, j) == 0.0) &&
                (m_get_val(img_dilated, i, j) == m_get_val(max_gicov, i, j))) {
                crow[pair_counter] = i;
                ccol[pair_counter] = j;
                pair_counter++;
            }
        }
    }

    // Get GICOV values for detected spots
    double* GICOV_spots = (double*)malloc(sizeof(double) * pair_counter);
    for (int i = 0; i < pair_counter; i++) {
        GICOV_spots[i] = m_get_val(gicov, crow[i], ccol[i]);
    }

    // Filter results and compute cell centers (matching cpu-cpu version)
    double* G = (double*)calloc(pair_counter, sizeof(double));
    double* x_result = (double*)calloc(pair_counter, sizeof(double));
    double* y_result = (double*)calloc(pair_counter, sizeof(double));

    int x_result_len = 0;
    for (int i = 0; i < pair_counter; i++) {
        if ((crow[i] > 29) && (crow[i] < BOTTOM - TOP + 39)) {
            x_result[x_result_len] = ccol[i];
            y_result[x_result_len] = crow[i] - 40;
            G[x_result_len] = GICOV_spots[i];
            x_result_len++;
        }
    }

    // Create cell boundaries and find valid cell centers
    double threshold = 1.8;
    double radius = 10.0;
    double b = 5.0;

    double* QAX_CENTERS = (double*)malloc(sizeof(double) * pair_counter);
    double* QAY_CENTERS = (double*)malloc(sizeof(double) * pair_counter);
    memset(QAX_CENTERS, 0, sizeof(double) * pair_counter);
    memset(QAY_CENTERS, 0, sizeof(double) * pair_counter);

    int k_count = 0;
    for (int n = 0; n < x_result_len; n++) {
        if ((G[n] < -1 * threshold) || G[n] > threshold) {
            // Check if cell is not too close to edge
            double min_x = x_result[n] - radius;
            double max_x = x_result[n] + radius;
            double min_y = y_result[n] - radius;
            double max_y = y_result[n] + radius;

            if ((min_x > b) && (min_y > b) &&
                (max_x < cell_file->width - b) && (max_y < cell_file->height - b)) {
                QAX_CENTERS[k_count] = x_result[n];
                QAY_CENTERS[k_count] = y_result[n] + TOP;
                k_count++;
            }
        }
    }

    printf("Cells detected: %d\n", k_count);

    // === PHASE 4: Track cells across frames (matching cpu-cpu version) ===
    if (num_frames > 1) printf("\nTracking cells across %d frames\n", num_frames);
    else                printf("\nTracking cells across 1 frame\n");

    int num_snaxels = 20;
    ellipsetrack(cell_file, QAX_CENTERS, QAY_CENTERS, k_count, radius, num_snaxels, num_frames);

    double compute_end = get_time_ms();
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

    // Cleanup
    cudaEventDestroy(transfer_to_gpu_start);
    cudaEventDestroy(transfer_to_gpu_end);
    cudaEventDestroy(transfer_to_cpu_start);
    cudaEventDestroy(transfer_to_cpu_end);

    free(h_grad_x);
    free(h_grad_y);
    free(h_grad_x_from_gpu);
    free(h_grad_y_from_gpu);
    free(crow);
    free(ccol);
    free(GICOV_spots);
    free(G);
    free(x_result);
    free(y_result);
    free(QAX_CENTERS);
    free(QAY_CENTERS);

    m_free(grad_x);
    m_free(grad_y);
    m_free(grad_x_cpu);
    m_free(grad_y_cpu);
    m_free(gicov);
    m_free(max_gicov);
    m_free(img_dilated);

    cudaFree(d_grad_x);
    cudaFree(d_grad_y);

    AVI_close(cell_file);

    return 0;
}
