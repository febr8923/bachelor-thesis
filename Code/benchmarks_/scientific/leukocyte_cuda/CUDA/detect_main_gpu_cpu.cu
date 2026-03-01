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
// (matches ellipsematching() in OpenMP/find_ellipse.c exactly)
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

                // Keep track of the maximal GICOV value seen so far
                if (mean * mean / var > max_GICOV) {
                    m_set_val(gicov, j, i, mean / sqrt(var));
                    max_GICOV = mean * mean / var;
                }
            }
        }
    }

    return gicov;
}

// CPU implementation of dilation using OpenMP
// (matches dilate_f() with structuring_element() in OpenMP/find_ellipse.c exactly)
MAT* dilate_cpu(MAT* img_in) {
    int strel_radius = 12;
    int strel_m = strel_radius * 2 + 1;
    int strel_n = strel_radius * 2 + 1;

    // Build circular structuring element (matching structuring_element())
    MAT* strel = m_get(strel_m, strel_n);
    for (int si = 0; si < strel_m; si++) {
        for (int sj = 0; sj < strel_n; sj++) {
            if (sqrt((float)((si - strel_radius) * (si - strel_radius) +
                             (sj - strel_radius) * (sj - strel_radius))) <= strel_radius)
                m_set_val(strel, si, sj, 1.0);
            else
                m_set_val(strel, si, sj, 0.0);
        }
    }

    MAT* dilated = m_get(img_in->m, img_in->n);
    int el_center_i = strel->m / 2, el_center_j = strel->n / 2;

    #pragma omp parallel for num_threads(omp_num_threads)
    for (int i = 0; i < img_in->m; i++) {
        int j, el_i, el_j, x, y;
        for (j = 0; j < img_in->n; j++) {
            double max = 0.0, temp;
            for (el_i = 0; el_i < strel->m; el_i++) {
                for (el_j = 0; el_j < strel->n; el_j++) {
                    y = i - el_center_i + el_i;
                    x = j - el_center_j + el_j;
                    if (y >= 0 && x >= 0 && y < img_in->m && x < img_in->n &&
                        m_get_val(strel, el_i, el_j) != 0) {
                        temp = m_get_val(img_in, y, x);
                        if (temp > max) max = temp;
                    }
                }
            }
            m_set_val(dilated, i, j, max);
        }
    }

    m_free(strel);
    return dilated;
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

    // Create CUDA events for timing
    cudaEvent_t transfer_to_gpu_start, transfer_to_gpu_end;
    cudaEvent_t transfer_to_cpu_start, transfer_to_cpu_end;
    cudaEventCreate(&transfer_to_gpu_start);
    cudaEventCreate(&transfer_to_gpu_end);
    cudaEventCreate(&transfer_to_cpu_start);
    cudaEventCreate(&transfer_to_cpu_end);

    // Extract and process first frame
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

    // Find cell centers (matching cpu-cpu version exactly)
    int i, j, n, pair_counter = 0, x_result_len = 0, Iter = 20, ns = 4, k_count = 0;
    int* crow = (int*)malloc(max_gicov->m * max_gicov->n * sizeof(int));
    int* ccol = (int*)malloc(max_gicov->m * max_gicov->n * sizeof(int));

    for (i = 0; i < max_gicov->m; i++) {
        for (j = 0; j < max_gicov->n; j++) {
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
    for (i = 0; i < pair_counter; i++) {
        GICOV_spots[i] = m_get_val(gicov, crow[i], ccol[i]);
    }

    double* G = (double*)calloc(pair_counter, sizeof(double));
    double* x_result = (double*)calloc(pair_counter, sizeof(double));
    double* y_result = (double*)calloc(pair_counter, sizeof(double));

    x_result_len = 0;
    for (i = 0; i < pair_counter; i++) {
        if ((crow[i] > 29) && (crow[i] < BOTTOM - TOP + 39)) {
            x_result[x_result_len] = ccol[i];
            y_result[x_result_len] = crow[i] - 40;
            G[x_result_len] = GICOV_spots[i];
            x_result_len++;
        }
    }

    // Make an array t which holds each "time step" for the possible cells
    double* t = (double*)malloc(sizeof(double) * 36);
    for (i = 0; i < 36; i++) {
        t[i] = (double)i * 2.0 * PI / 36.0;
    }

    // Store cell boundaries (as simple circles) for all cells
    MAT* cellx = m_get(x_result_len, 36);
    MAT* celly = m_get(x_result_len, 36);
    double radius = 10.0;
    for (i = 0; i < x_result_len; i++) {
        for (j = 0; j < 36; j++) {
            m_set_val(cellx, i, j, x_result[i] + radius * cos(t[j]));
            m_set_val(celly, i, j, y_result[i] + radius * sin(t[j]));
        }
    }

    MAT* A = TMatrix(9, 4);

    double threshold = 1.8;
    double delta = 3.0;
    double dt = 0.01;
    double b = 5.0;

    double* V = (double*)malloc(sizeof(double) * pair_counter);
    double* QAX_CENTERS = (double*)malloc(sizeof(double) * pair_counter);
    double* QAY_CENTERS = (double*)malloc(sizeof(double) * pair_counter);
    memset(V, 0, sizeof(double) * pair_counter);
    memset(QAX_CENTERS, 0, sizeof(double) * pair_counter);
    memset(QAY_CENTERS, 0, sizeof(double) * pair_counter);

    // For all possible results, find the ones that are feasibly leukocytes and store their centers
    k_count = 0;
    for (n = 0; n < x_result_len; n++) {
        if ((G[n] < -1 * threshold) || G[n] > threshold) {
            MAT *x, *y;
            VEC *x_row, *y_row;
            x = m_get(1, 36);
            y = m_get(1, 36);

            x_row = v_get(36);
            y_row = v_get(36);

            // Get current values of possible cells from cellx/celly matrices
            x_row = get_row(cellx, n, x_row);
            y_row = get_row(celly, n, y_row);
            uniformseg(x_row, y_row, x, y);

            // Make sure that the possible leukocytes are not too close to the edge of the frame
            if ((m_min(x) > b) && (m_min(y) > b) && (m_max(x) < cell_file->width - b) && (m_max(y) < cell_file->height - b)) {
                MAT *Cx, *Cy, *Cy_temp, *Ix1, *Iy1;
                VEC *Xs, *Ys, *W, *Nx, *Ny, *X, *Y;
                Cx = m_get(1, 36);
                Cy = m_get(1, 36);
                Cx = mmtr_mlt(A, x, Cx);
                Cy = mmtr_mlt(A, y, Cy);

                Cy_temp = m_get(Cy->m, Cy->n);

                for (i = 0; i < 9; i++)
                    m_set_val(Cy, i, 0, m_get_val(Cy, i, 0) + 40.0);

                // Iteratively refine the snake/spline
                for (i = 0; i < Iter; i++) {
                    int typeofcell;

                    if (G[n] > 0.0) typeofcell = 0;
                    else typeofcell = 1;

                    splineenergyform01(Cx, Cy, grad_x_cpu, grad_y_cpu, ns, delta, 2.0 * dt, typeofcell);
                }

                X = getsampling(Cx, ns);
                for (i = 0; i < Cy->m; i++)
                    m_set_val(Cy_temp, i, 0, m_get_val(Cy, i, 0) - 40.0);
                Y = getsampling(Cy_temp, ns);

                Ix1 = linear_interp2(grad_x_cpu, X, Y);
                Iy1 = linear_interp2(grad_x_cpu, X, Y);
                Xs = getfdriv(Cx, ns);
                Ys = getfdriv(Cy, ns);

                Nx = v_get(Ys->dim);
                for (i = 0; i < (int)Ys->dim; i++)
                    v_set_val(Nx, i, v_get_val(Ys, i) / sqrt(v_get_val(Xs, i)*v_get_val(Xs, i) + v_get_val(Ys, i)*v_get_val(Ys, i)));

                Ny = v_get(Xs->dim);
                for (i = 0; i < (int)Xs->dim; i++)
                    v_set_val(Ny, i, -1.0 * v_get_val(Xs, i) / sqrt(v_get_val(Xs, i)*v_get_val(Xs, i) + v_get_val(Ys, i)*v_get_val(Ys, i)));

                W = v_get(Nx->dim);
                for (i = 0; i < (int)Nx->dim; i++)
                    v_set_val(W, i, m_get_val(Ix1, 0, i) * v_get_val(Nx, i) + m_get_val(Iy1, 0, i) * v_get_val(Ny, i));

                V[n] = mean(W) / std_dev(W);

                // Get means of X and Y values for all "snaxels" of the spline contour
                QAX_CENTERS[k_count] = mean(X);
                QAY_CENTERS[k_count] = mean(Y) + TOP;

                k_count++;

                // Free memory
                v_free(W);
                v_free(Ny);
                v_free(Nx);
                v_free(Ys);
                v_free(Xs);
                m_free(Iy1);
                m_free(Ix1);
                v_free(Y);
                v_free(X);
                m_free(Cy_temp);
                m_free(Cy);
                m_free(Cx);
            }

            // Free memory
            v_free(y_row);
            v_free(x_row);
            m_free(y);
            m_free(x);
        }
    }

    // === PHASE 4: Track cells across frames (matching cpu-cpu version) ===
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
    printf("Total time:                %.3f ms\n", total_time);
    printf("GPU->CPU data transfer:    %.3f ms\n", transfer_to_cpu_time);
    printf("CPU computation (OpenMP):  %.3f ms\n", computation_time);

    // Cleanup
    cudaEventDestroy(transfer_to_gpu_start);
    cudaEventDestroy(transfer_to_gpu_end);
    cudaEventDestroy(transfer_to_cpu_start);
    cudaEventDestroy(transfer_to_cpu_end);

    free(h_grad_x);
    free(h_grad_y);
    free(h_grad_x_from_gpu);
    free(h_grad_y_from_gpu);
    free(V);
    free(crow);
    free(ccol);
    free(GICOV_spots);
    free(t);
    free(G);
    free(x_result);
    free(y_result);
    free(QAX_CENTERS);
    free(QAY_CENTERS);

    m_free(A);
    m_free(celly);
    m_free(cellx);
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
