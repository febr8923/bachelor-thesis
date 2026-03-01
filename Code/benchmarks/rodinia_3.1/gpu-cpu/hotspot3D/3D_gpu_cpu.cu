/***********************************************************************************
  GPU-CPU Execution variant for Hotspot3D
  Copies data to GPU and back, then performs computation on CPU (OpenMP).
  Measures data transfer time, computation time, and total time separately.

  Usage: ./3D_gpu_cpu <rows/cols> <layers> <iterations> <powerFile> <tempFile> <outputFile>
 ************************************************************************************/
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <cuda.h>
#include <omp.h>

#define STR_SIZE (256)
#define MAX_PD	(3.0e6)
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
#define FACTOR_CHIP	0.5

/* chip parameters */
float t_chip = 0.0005;
float chip_height = 0.016;
float chip_width = 0.016;
float amb_temp = 80.0;

void fatal(const char *s)
{
    fprintf(stderr, "Error: %s\n", s);
}

void readinput(float *vect, int grid_rows, int grid_cols, int layers, char *file) {
    int i, j, k;
    FILE *fp;
    char str[STR_SIZE];
    float val;

    if ((fp = fopen(file, "r")) == 0)
        fatal("The file was not opened");

    for (i = 0; i <= grid_rows - 1; i++)
        for (j = 0; j <= grid_cols - 1; j++)
            for (k = 0; k <= layers - 1; k++) {
                if (fgets(str, STR_SIZE, fp) == NULL) fatal("Error reading file\n");
                if (feof(fp))
                    fatal("not enough lines in file");
                if ((sscanf(str, "%f", &val) != 1))
                    fatal("invalid file format");
                vect[i * grid_cols + j + k * grid_rows * grid_cols] = val;
            }

    fclose(fp);
}

void writeoutput(float *vect, int grid_rows, int grid_cols, int layers, char *file) {
    int i, j, k, index = 0;
    FILE *fp;
    char str[STR_SIZE];

    if ((fp = fopen(file, "w")) == 0)
        printf("The file was not opened\n");

    for (i = 0; i < grid_rows; i++)
        for (j = 0; j < grid_cols; j++)
            for (k = 0; k < layers; k++) {
                sprintf(str, "%d\t%g\n", index, vect[i * grid_cols + j + k * grid_rows * grid_cols]);
                fputs(str, fp);
                index++;
            }

    fclose(fp);
}

void computeTempOMP(float *pIn, float *tIn, float *tOut,
        int nx, int ny, int nz, float Cap,
        float Rx, float Ry, float Rz,
        float dt, int numiter)
{
    float ce, cw, cn, cs, ct, cb, cc;
    float stepDivCap = dt / Cap;
    ce = cw = stepDivCap / Rx;
    cn = cs = stepDivCap / Ry;
    ct = cb = stepDivCap / Rz;

    cc = 1.0 - (2.0 * ce + 2.0 * cn + 3.0 * ct);

#pragma omp parallel
    {
        int count = 0;
        float *tIn_t = tIn;
        float *tOut_t = tOut;

        do {
            int z;
#pragma omp for
            for (z = 0; z < nz; z++) {
                int y;
                for (y = 0; y < ny; y++) {
                    int x;
                    for (x = 0; x < nx; x++) {
                        int c, w, e, n, s, b, t;
                        c = x + y * nx + z * nx * ny;
                        w = (x == 0)      ? c : c - 1;
                        e = (x == nx - 1) ? c : c + 1;
                        n = (y == 0)      ? c : c - nx;
                        s = (y == ny - 1) ? c : c + nx;
                        b = (z == 0)      ? c : c - nx * ny;
                        t = (z == nz - 1) ? c : c + nx * ny;
                        tOut_t[c] = cc * tIn_t[c] + cw * tIn_t[w] + ce * tIn_t[e]
                            + cs * tIn_t[s] + cn * tIn_t[n] + cb * tIn_t[b] + ct * tIn_t[t]
                            + (dt / Cap) * pIn[c] + ct * amb_temp;
                    }
                }
            }
            float *tmp = tIn_t;
            tIn_t = tOut_t;
            tOut_t = tmp;
            count++;
        } while (count < numiter);
    }
    return;
}

void usage(int argc, char **argv)
{
    fprintf(stderr, "Usage: %s <rows/cols> <layers> <iterations> <powerFile> <tempFile> <outputFile>\n", argv[0]);
    fprintf(stderr, "\t<rows/cols>  - number of rows/cols in the grid (positive integer)\n");
    fprintf(stderr, "\t<layers>  - number of layers in the grid (positive integer)\n");
    fprintf(stderr, "\t<iteration> - number of iterations\n");
    fprintf(stderr, "\t<powerFile>  - name of the file containing the initial power values of each cell\n");
    fprintf(stderr, "\t<tempFile>  - name of the file containing the initial temperature values of each cell\n");
    fprintf(stderr, "\t<outputFile - output file\n");
    exit(1);
}

int main(int argc, char **argv)
{
    if (argc != 7) {
        usage(argc, argv);
    }

    char *pfile, *tfile, *ofile;
    int iterations = atoi(argv[3]);

    pfile = argv[4];
    tfile = argv[5];
    ofile = argv[6];
    int numCols = atoi(argv[1]);
    int numRows = atoi(argv[1]);
    int layers = atoi(argv[2]);

    /* calculating parameters */
    float dx = chip_height / numRows;
    float dy = chip_width / numCols;
    float dz = t_chip / layers;

    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;
    float Rx = dy / (2.0 * K_SI * t_chip * dx);
    float Ry = dx / (2.0 * K_SI * t_chip * dy);
    float Rz = dz / (K_SI * dx * dy);

    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float dt = PRECISION / max_slope;

    int size = numCols * numRows * layers;

    float *powerIn = (float *)calloc(size, sizeof(float));
    float *tempIn  = (float *)calloc(size, sizeof(float));
    float *tempOut = (float *)calloc(size, sizeof(float));

    readinput(powerIn, numRows, numCols, layers, pfile);
    readinput(tempIn, numRows, numCols, layers, tfile);

    // ===================== GPU MEMORY ALLOCATION =====================
    size_t s = sizeof(float) * size;
    float *d_powerIn, *d_tempIn, *d_tempOut;

    cudaMalloc((void **)&d_powerIn, s);
    cudaMalloc((void **)&d_tempIn, s);
    cudaMalloc((void **)&d_tempOut, s);

    // ===================== COPY HOST -> DEVICE (untimed, simulates data on GPU) =====================
    cudaMemcpy(d_powerIn, powerIn, s, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tempIn, tempIn, s, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tempOut, tempOut, s, cudaMemcpyHostToDevice);

    printf("Copied data to GPU memory\n");

    // ===================== START TIMERS =====================
    double total_start_time = omp_get_wtime();
    double transfer_start_time = omp_get_wtime();

    // ===================== COPY DEVICE -> HOST (timed) =====================
    cudaMemcpy(powerIn, d_powerIn, s, cudaMemcpyDeviceToHost);
    cudaMemcpy(tempIn, d_tempIn, s, cudaMemcpyDeviceToHost);
    cudaMemcpy(tempOut, d_tempOut, s, cudaMemcpyDeviceToHost);

    double transfer_end_time = omp_get_wtime();

    // ===================== CPU COMPUTATION (OpenMP) =====================
    double compute_start_time = omp_get_wtime();

    computeTempOMP(powerIn, tempIn, tempOut, numCols, numRows, layers, Cap, Rx, Ry, Rz, dt, iterations);

    double compute_end_time = omp_get_wtime();
    double total_end_time = omp_get_wtime();

    // ===================== PRINT TIMING RESULTS =====================
    printf("\n===== GPU-CPU Execution Timing =====\n");
    printf("Data transfer time (D2H): %lf seconds\n", transfer_end_time - transfer_start_time);
    printf("Execution time: %lf seconds\n", compute_end_time - compute_start_time);
    printf("Total time: %lf seconds\n", total_end_time - total_start_time);

    writeoutput(tempOut, numRows, numCols, layers, ofile);

    // cleanup
    free(powerIn);
    free(tempIn);
    free(tempOut);
    cudaFree(d_powerIn);
    cudaFree(d_tempIn);
    cudaFree(d_tempOut);

    return 0;
}
