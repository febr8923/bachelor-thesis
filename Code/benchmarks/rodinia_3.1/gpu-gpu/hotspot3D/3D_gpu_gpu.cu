/***********************************************************************************
  GPU-GPU Execution variant for Hotspot3D
  Data starts on GPU, computation on GPU. No data transfer is timed.
  Measures kernel execution time only.

  Usage: ./3D_gpu_gpu <rows/cols> <layers> <iterations> <powerFile> <tempFile> <outputFile>
 ************************************************************************************/
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>

#define BLOCK_SIZE 16
#define STR_SIZE 256

#define block_x_ 128
#define block_y_ 2
#define block_z_ 1
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

__global__ void hotspotOpt1(float *p, float *tIn, float *tOut, float sdc,
        int nx, int ny, int nz,
        float ce, float cw,
        float cn, float cs,
        float ct, float cb,
        float cc)
{
    float amb_temp = 80.0;

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    int c = i + j * nx;
    int xy = nx * ny;

    int W = (i == 0)      ? c : c - 1;
    int E = (i == nx - 1) ? c : c + 1;
    int N = (j == 0)      ? c : c - nx;
    int S = (j == ny - 1) ? c : c + nx;

    float temp1, temp2, temp3;
    temp1 = temp2 = tIn[c];
    temp3 = tIn[c + xy];
    tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
        + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;
    c += xy;
    W += xy;
    E += xy;
    N += xy;
    S += xy;

    for (int k = 1; k < nz - 1; ++k) {
        temp1 = temp2;
        temp2 = temp3;
        temp3 = tIn[c + xy];
        tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
            + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;
        c += xy;
        W += xy;
        E += xy;
        N += xy;
        S += xy;
    }
    temp1 = temp2;
    temp2 = temp3;
    tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
        + cn * tIn[N] + cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;
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
    size_t s = sizeof(float) * size;

    float *powerIn = (float *)calloc(size, sizeof(float));
    float *tempIn  = (float *)calloc(size, sizeof(float));
    float *tempOut = (float *)calloc(size, sizeof(float));

    readinput(powerIn, numRows, numCols, layers, pfile);
    readinput(tempIn, numRows, numCols, layers, tfile);

    // ===================== GPU MEMORY ALLOCATION & H2D (untimed, simulates data on GPU) =====================
    float *d_powerIn, *d_tempIn, *d_tempOut;

    cudaMalloc((void **)&d_powerIn, s);
    cudaMalloc((void **)&d_tempIn, s);
    cudaMalloc((void **)&d_tempOut, s);

    cudaMemcpy(d_powerIn, powerIn, s, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tempIn, tempIn, s, cudaMemcpyHostToDevice);

    printf("Copied data to GPU memory\n");

    // Compute coefficients
    float ce, cw, cn, cs, ct, cb, cc;
    float stepDivCap = dt / Cap;
    ce = cw = stepDivCap / Rx;
    cn = cs = stepDivCap / Ry;
    ct = cb = stepDivCap / Rz;
    cc = 1.0 - (2.0 * ce + 2.0 * cn + 3.0 * ct);

    cudaFuncSetCacheConfig(hotspotOpt1, cudaFuncCachePreferL1);

    dim3 block_dim(64, 4, 1);
    dim3 grid_dim(numCols / 64, numRows / 4, 1);

    // ===================== START TIMER (after all alloc + H2D) =====================
    struct timeval t_start, t_end;
    cudaDeviceSynchronize();
    gettimeofday(&t_start, NULL);

    for (int i = 0; i < iterations; ++i) {
        hotspotOpt1<<<grid_dim, block_dim>>>
            (d_powerIn, d_tempIn, d_tempOut, stepDivCap, numCols, numRows, layers,
             ce, cw, cn, cs, ct, cb, cc);
        float *tmp = d_tempIn;
        d_tempIn = d_tempOut;
        d_tempOut = tmp;
    }

    cudaDeviceSynchronize();
    gettimeofday(&t_end, NULL);

    // ===================== STOP TIMER =====================
    double execution_time = (t_end.tv_sec - t_start.tv_sec) + (t_end.tv_usec - t_start.tv_usec) / 1000000.0;

    // Copy result back (not timed — just to retrieve results)
    cudaMemcpy(tempOut, d_tempOut, s, cudaMemcpyDeviceToHost);

    printf("\n===== GPU-GPU Execution Timing =====\n");
    printf("Execution time: %lf seconds\n", execution_time);

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
