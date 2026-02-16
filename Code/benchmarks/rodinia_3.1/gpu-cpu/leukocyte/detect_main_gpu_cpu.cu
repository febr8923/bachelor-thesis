/***********************************************************************************
  GPU-CPU Execution variant for Leukocyte detection and tracking.
  Copies data to GPU and back, then performs computation on CPU (OpenMP).
  Measures data transfer time, computation time, and total time separately.

  This file replaces detect_main.c. Compile with the OpenMP versions of
  find_ellipse.c, track_ellipse.c, and supporting libraries, plus CUDA runtime.

  Example compilation:
    nvcc -Xcompiler -fopenmp -o leukocyte_gpu_cpu detect_main_gpu_cpu.cu \
      ../../openmp/leukocyte/OpenMP/find_ellipse.c \
      ../../openmp/leukocyte/OpenMP/track_ellipse.c \
      ../../openmp/leukocyte/OpenMP/avilib.c \
      ../../openmp/leukocyte/OpenMP/misc_math.c \
      -I../../openmp/leukocyte/OpenMP \
      -I../../openmp/leukocyte/OpenMP/meschach_lib \
      -L../../openmp/leukocyte/OpenMP/meschach_lib -lmeschach -lm
 ************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

extern "C" {
#include "find_ellipse.h"
#include "track_ellipse.h"
}

int omp_num_threads = 1;

int main(int argc, char ** argv) {

	// Keep track of the start time of the program
	long long program_start_time = get_time();

	// Let the user specify the number of frames to process
	int num_frames = 1;

	if (argc !=4){
		fprintf(stderr, "usage: %s <num of frames> <num of threads> <input file>", argv[0]);
		exit(1);
	}

	if (argc > 1){
		num_frames = atoi(argv[1]);
		omp_num_threads = atoi(argv[2]);
	}
	printf("Num of threads: %d\n", omp_num_threads);

	// Open video file
	char *video_file_name;
	video_file_name = argv[3];

	avi_t *cell_file = AVI_open_input_file(video_file_name, 1);
	if (cell_file == NULL)	{
		AVI_print_error("Error with AVI_open_input_file");
		return -1;
	}

	int i, j, *crow, *ccol, pair_counter = 0, x_result_len = 0, Iter = 20, ns = 4, k_count = 0, n;
	MAT *cellx, *celly, *A;
	double *GICOV_spots, *t, *G, *x_result, *y_result, *V, *QAX_CENTERS, *QAY_CENTERS;
	double threshold = 1.8, radius = 10.0, delta = 3.0, dt = 0.01, b = 5.0;

	// Extract a cropped version of the first frame from the video file
	MAT *image_chopped = get_frame(cell_file, 0, 1, 0);
	printf("Detecting cells in frame 0\n");

	// Get gradient matrices in x and y directions
	MAT *grad_x = gradient_x(image_chopped);
	MAT *grad_y = gradient_y(image_chopped);

	m_free(image_chopped);

	int height = grad_x->m;
	int width = grad_x->n;
	int matrix_size = height * width;

	// ===================== Extract MAT data into flat arrays =====================
	double *h_grad_x = (double *) malloc(sizeof(double) * matrix_size);
	double *h_grad_y = (double *) malloc(sizeof(double) * matrix_size);

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			h_grad_x[i * width + j] = m_get_val(grad_x, i, j);
			h_grad_y[i * width + j] = m_get_val(grad_y, i, j);
		}
	}

	// ===================== GPU MEMORY ALLOCATION =====================
	double *d_grad_x, *d_grad_y;
	cudaMalloc((void **) &d_grad_x, sizeof(double) * matrix_size);
	cudaMalloc((void **) &d_grad_y, sizeof(double) * matrix_size);

	// ===================== COPY HOST -> DEVICE =====================
	cudaMemcpy(d_grad_x, h_grad_x, sizeof(double) * matrix_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_grad_y, h_grad_y, sizeof(double) * matrix_size, cudaMemcpyHostToDevice);

	printf("Copied gradient data to GPU memory (%d x %d)\n", height, width);

	// ===================== START TIMERS =====================
	double total_start_time = omp_get_wtime();
	double transfer_start_time = omp_get_wtime();

	// ===================== COPY DEVICE -> HOST =====================
	cudaMemcpy(h_grad_x, d_grad_x, sizeof(double) * matrix_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_grad_y, d_grad_y, sizeof(double) * matrix_size, cudaMemcpyDeviceToHost);

	double transfer_end_time = omp_get_wtime();

	// ===================== CPU COMPUTATION (OpenMP) =====================
	double compute_start_time = omp_get_wtime();

	// Copy data back into MAT structures
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			m_set_val(grad_x, i, j, h_grad_x[i * width + j]);
			m_set_val(grad_y, i, j, h_grad_y[i * width + j]);
		}
	}

	// Get GICOV matrix corresponding to image gradients
	long long GICOV_start_time = get_time();
	MAT *gicov = ellipsematching(grad_x, grad_y);

	// Square GICOV values
	MAT *max_gicov = m_get(gicov->m, gicov->n);
	for (i = 0; i < gicov->m; i++) {
		for (j = 0; j < gicov->n; j++) {
			double val = m_get_val(gicov, i, j);
			m_set_val(max_gicov, i, j, val * val);
		}
	}

	long long GICOV_end_time = get_time();

	// Dilate the GICOV matrix
	long long dilate_start_time = get_time();
	MAT *strel = structuring_element(12);
	MAT *img_dilated = dilate_f(max_gicov, strel);
	long long dilate_end_time = get_time();

	// Find possible matches for cell centers
	pair_counter = 0;
	crow = (int *) malloc(max_gicov->m * max_gicov->n * sizeof(int));
	ccol = (int *) malloc(max_gicov->m * max_gicov->n * sizeof(int));
	for (i = 0; i < max_gicov->m; i++) {
		for (j = 0; j < max_gicov->n; j++) {
			if (!(m_get_val(max_gicov,i,j) == 0.0) && (m_get_val(img_dilated,i,j) == m_get_val(max_gicov,i,j))) {
				crow[pair_counter] = i;
				ccol[pair_counter] = j;
				pair_counter++;
			}
		}
	}

	GICOV_spots = (double *) malloc(sizeof(double)*pair_counter);
	for (i = 0; i < pair_counter; i++)
		GICOV_spots[i] = m_get_val(gicov, crow[i], ccol[i]);

	G = (double *) calloc(pair_counter, sizeof(double));
	x_result = (double *) calloc(pair_counter, sizeof(double));
	y_result = (double *) calloc(pair_counter, sizeof(double));

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
	t = (double *) malloc(sizeof(double) * 36);
	for (i = 0; i < 36; i++) {
		t[i] = (double)i * 2.0 * PI / 36.0;
	}

	// Store cell boundaries
	cellx = m_get(x_result_len, 36);
	celly = m_get(x_result_len, 36);
	for(i = 0; i < x_result_len; i++) {
		for(j = 0; j < 36; j++) {
			m_set_val(cellx, i, j, x_result[i] + radius * cos(t[j]));
			m_set_val(celly, i, j, y_result[i] + radius * sin(t[j]));
		}
	}

	A = TMatrix(9,4);

	V = (double *) malloc(sizeof(double) * pair_counter);
	QAX_CENTERS = (double * )malloc(sizeof(double) * pair_counter);
	QAY_CENTERS = (double *) malloc(sizeof(double) * pair_counter);
	memset(V, 0, sizeof(double) * pair_counter);
	memset(QAX_CENTERS, 0, sizeof(double) * pair_counter);
	memset(QAY_CENTERS, 0, sizeof(double) * pair_counter);

	// For all possible results, find the ones that are feasibly leukocytes
	k_count = 0;
	for (n = 0; n < x_result_len; n++) {
		if ((G[n] < -1 * threshold) || G[n] > threshold) {
			MAT * x, *y;
			VEC * x_row, * y_row;
			x = m_get(1, 36);
			y = m_get(1, 36);

			x_row = v_get(36);
			y_row = v_get(36);

			x_row = get_row(cellx, n, x_row);
			y_row = get_row(celly, n, y_row);
			uniformseg(x_row, y_row, x, y);

			if ((m_min(x) > b) && (m_min(y) > b) && (m_max(x) < cell_file->width - b) && (m_max(y) < cell_file->height - b)) {
				MAT * Cx, * Cy, *Cy_temp, * Ix1, * Iy1;
				VEC  *Xs, *Ys, *W, *Nx, *Ny, *X, *Y;
				Cx = m_get(1, 36);
				Cy = m_get(1, 36);
				Cx = mmtr_mlt(A, x, Cx);
				Cy = mmtr_mlt(A, y, Cy);

				Cy_temp = m_get(Cy->m, Cy->n);

				for (i = 0; i < 9; i++)
					m_set_val(Cy, i, 0, m_get_val(Cy, i, 0) + 40.0);

				for (i = 0; i < Iter; i++) {
					int typeofcell;

					if(G[n] > 0.0) typeofcell = 0;
					else typeofcell = 1;

					splineenergyform01(Cx, Cy, grad_x, grad_y, ns, delta, 2.0 * dt, typeofcell);
				}

				X = getsampling(Cx, ns);
				for (i = 0; i < Cy->m; i++)
					m_set_val(Cy_temp, i, 0, m_get_val(Cy, i, 0) - 40.0);
				Y = getsampling(Cy_temp, ns);

				Ix1 = linear_interp2(grad_x, X, Y);
				Iy1 = linear_interp2(grad_x, X, Y);
				Xs = getfdriv(Cx, ns);
				Ys = getfdriv(Cy, ns);

				Nx = v_get(Ys->dim);
				for (i = 0; i < Ys->dim; i++)
					v_set_val(Nx, i, v_get_val(Ys, i) / sqrt(v_get_val(Xs, i)*v_get_val(Xs, i) + v_get_val(Ys, i)*v_get_val(Ys, i)));

				Ny = v_get(Xs->dim);
				for (i = 0; i < Xs->dim; i++)
					v_set_val(Ny, i, -1.0 * v_get_val(Xs, i) / sqrt(v_get_val(Xs, i)*v_get_val(Xs, i) + v_get_val(Ys, i)*v_get_val(Ys, i)));

				W = v_get(Nx->dim);
				for (i = 0; i < Nx->dim; i++)
					v_set_val(W, i, m_get_val(Ix1, 0, i) * v_get_val(Nx, i) + m_get_val(Iy1, 0, i) * v_get_val(Ny, i));

				V[n] = mean(W) / std_dev(W);

				QAX_CENTERS[k_count] = mean(X);
				QAY_CENTERS[k_count] = mean(Y) + TOP;

				k_count++;

				v_free(W); v_free(Ny); v_free(Nx); v_free(Ys); v_free(Xs);
				m_free(Iy1); m_free(Ix1); v_free(Y); v_free(X);
				m_free(Cy_temp); m_free(Cy); m_free(Cx);
			}

			v_free(y_row); v_free(x_row);
			m_free(y); m_free(x);
		}
	}

	// Track ellipses through subsequent frames (CPU/OpenMP)
	if (num_frames > 1) printf("\nTracking cells across %d frames\n", num_frames);
	else                printf("\nTracking cells across 1 frame\n");
	long long tracking_start_time = get_time();
	int num_snaxels = 20;
	ellipsetrack(cell_file, QAX_CENTERS, QAY_CENTERS, k_count, radius, num_snaxels, num_frames);

	double compute_end_time = omp_get_wtime();
	double total_end_time = omp_get_wtime();

	// ===================== PRINT TIMING RESULTS =====================
	double data_transfer_time = transfer_end_time - transfer_start_time;
	double computation_time = compute_end_time - compute_start_time;
	double total_time = total_end_time - total_start_time;

	// Report the total number of cells detected
	printf("Cells detected: %d\n\n", k_count);

	// Report the breakdown of the detection runtime
	printf("Detection runtime\n");
	printf("-----------------\n");
	printf("GICOV computation: %.5f seconds\n", ((float) (GICOV_end_time - GICOV_start_time)) / (1000*1000));
	printf("   GICOV dilation: %.5f seconds\n", ((float) (dilate_end_time - dilate_start_time)) / (1000*1000));
	printf("Tracking per frame: %.5f seconds\n", ((float) (get_time() - tracking_start_time)) / (float) (1000*1000*num_frames));

	printf("\n===== GPU-CPU Execution Timing =====\n");
	printf("Data transfer time (D2H): %lf seconds\n", data_transfer_time);
	printf("Computation time (CPU):   %lf seconds\n", computation_time);
	printf("Total time:               %lf seconds\n", total_time);

	printf("\nTotal application run time: %.5f seconds\n", ((float) (get_time() - program_start_time)) / (1000*1000));

	// Free memory
	free(h_grad_x);
	free(h_grad_y);
	cudaFree(d_grad_x);
	cudaFree(d_grad_y);
	free(V); free(ccol); free(crow); free(GICOV_spots);
	free(t); free(G); free(x_result); free(y_result);
	m_free(A); m_free(celly); m_free(cellx);
	m_free(img_dilated); m_free(max_gicov); m_free(gicov);
	m_free(grad_y); m_free(grad_x);

	return 0;
}
