#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h> // for cpu time measurement

#define KERNEL_SIZE 3
#define WIDTH 1024
#define HEIGHT 1024
#define TILE_SIZE 16

__constant__ float c_kernel[9];

// kernels - functions
void convolutionCPU(float *in, float *kernel, float *out, int w, int h) {
    int r = KERNEL_SIZE / 2;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float sum = 0.0f;
            for (int ky = -r; ky <= r; ky++) {
                for (int kx = -r; kx <= r; kx++) {
                    int iy = y + ky; int ix = x + kx;
                    if (iy >= 0 && iy < h && ix >= 0 && ix < w)
                        sum += in[iy * w + ix] * kernel[(ky + r) * KERNEL_SIZE + (kx + r)];
                }
            }
            out[y * w + x] = sum;
        }
    }
}

__global__ void convolutionNaive(float *in, float *kernel, float *out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) {
        int r = KERNEL_SIZE / 2;
        float sum = 0.0f;
        for (int ky = -r; ky <= r; ky++) {
            for (int kx = -r; kx <= r; kx++) {
                int iy = y + ky; int ix = x + kx;
                if (iy >= 0 && iy < h && ix >= 0 && ix < w)
                    sum += in[iy * w + ix] * kernel[(ky + r) * KERNEL_SIZE + (kx + r)];
            }
        }
        out[y * w + x] = sum;
    }
}

__global__ void convolutionShared(float *in, float *out, int w, int h) {
    __shared__ float tile[TILE_SIZE + KERNEL_SIZE - 1][TILE_SIZE + KERNEL_SIZE - 1];
    int r = KERNEL_SIZE / 2;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int x = blockIdx.x * TILE_SIZE + tx; int y = blockIdx.y * TILE_SIZE + ty;

    for (int i = ty; i < TILE_SIZE + KERNEL_SIZE - 1; i += TILE_SIZE) {
        for (int j = tx; j < TILE_SIZE + KERNEL_SIZE - 1; j += TILE_SIZE) {
            int iy = blockIdx.y * TILE_SIZE + i - r;
            int ix = blockIdx.x * TILE_SIZE + j - r;
            if (iy >= 0 && iy < h && ix >= 0 && ix < w) tile[i][j] = in[iy * w + ix];
            else tile[i][j] = 0.0f;
        }
    }
    __syncthreads();
    if (x < w && y < h) {
        float sum = 0.0f;
        for (int ky = 0; ky < KERNEL_SIZE; ky++) {
            for (int kx = 0; kx < KERNEL_SIZE; kx++)
                sum += tile[ty + ky][tx + kx] * c_kernel[ky * KERNEL_SIZE + kx];
        }
        out[y * w + x] = sum;
    }
}

int main() {
    size_t size = WIDTH * HEIGHT * sizeof(float);
    float *h_in = (float*)malloc(size), *h_out = (float*)malloc(size), *h_k = (float*)malloc(9*sizeof(float));
    float *d_in, *d_out, *d_k_naive;
    cudaMalloc(&d_in, size); cudaMalloc(&d_out, size); cudaMalloc(&d_k_naive, 9*sizeof(float));

    for(int i=0; i<WIDTH*HEIGHT; i++) h_in[i] = 1.0f;
    for(int i=0; i<9; i++) h_k[i] = 1.0f/9.0f;

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_naive, h_k, 9*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_kernel, h_k, 9*sizeof(float));

    // 1. cpu Ölçümü
    clock_t start_c = clock();
    convolutionCPU(h_in, h_k, h_out, WIDTH, HEIGHT);
    clock_t end_c = clock();
    double cpu_ms = ((double)(end_c - start_c) / CLOCKS_PER_SEC) * 1000;

    // naive gpu 
    cudaEvent_t s1, e1; cudaEventCreate(&s1); cudaEventCreate(&e1);
    cudaEventRecord(s1);
    convolutionNaive<<<dim3((WIDTH+15)/16, (HEIGHT+15)/16), dim3(16,16)>>>(d_in, d_k_naive, d_out, WIDTH, HEIGHT);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float naive_ms; cudaEventElapsedTime(&naive_ms, s1, e1);

    // optimized gpu 
    cudaEventRecord(s1);
    convolutionShared<<<dim3((WIDTH+15)/16, (HEIGHT+15)/16), dim3(16,16)>>>(d_in, d_out, WIDTH, HEIGHT);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float opt_ms; cudaEventElapsedTime(&opt_ms, s1, e1);

    printf("CPU Time: %f ms\n", cpu_ms);
    printf("Naive GPU Time: %f ms\n", naive_ms);
    printf("Optimized GPU Time: %f ms\n", opt_ms);
    printf("Total Speedup: %fx\n", cpu_ms / opt_ms);

    return 0;
}
