#include <stdio.h>
#include <cuda_runtime.h>

#define KERNEL_SIZE 3
#define WIDTH 1024
#define HEIGHT 1024
#define TILE_SIZE 16

//  constant memory hızlı erişim
__constant__ float c_kernel[KERNEL_SIZE * KERNEL_SIZE];

// shared memory kullanan optimize kernel
__global__ void convolutionShared(float *in, float *out, int w, int h) {
    __shared__ float tile[TILE_SIZE + KERNEL_SIZE - 1][TILE_SIZE + KERNEL_SIZE - 1];

    int r = KERNEL_SIZE / 2;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int x = blockIdx.x * TILE_SIZE + tx;
    int y = blockIdx.y * TILE_SIZE + ty;

    // shared memory ye veri yükleme - coalesced access
    for (int i = ty; i < TILE_SIZE + KERNEL_SIZE - 1; i += TILE_SIZE) {
        for (int j = tx; j < TILE_SIZE + KERNEL_SIZE - 1; j += TILE_SIZE) {
            int iy = blockIdx.y * TILE_SIZE + i - r;
            int ix = blockIdx.x * TILE_SIZE + j - r;

            if (iy >= 0 && iy < h && ix >= 0 && ix < w)
                tile[i][j] = in[iy * w + ix];
            else
                tile[i][j] = 0.0f; 
        }
    }

    __syncthreads(); // wait for all the pixels

    if (x < w && y < h) {
        float sum = 0.0f;
        for (int ky = 0; ky < KERNEL_SIZE; ky++) {
            for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                sum += tile[ty + ky][tx + kx] * c_kernel[ky * KERNEL_SIZE + kx];
            }
        }
        out[y * w + x] = sum;
    }
}

int main() {
    size_t size = WIDTH * HEIGHT * sizeof(float);
    float *h_in, *h_out, *h_kernel, *d_in, *d_out;

    h_in = (float*)malloc(size);
    h_out = (float*)malloc(size);
    h_kernel = (float*)malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(float));

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    for(int i=0; i<WIDTH*HEIGHT; i++) h_in[i] = (float)(i % 255);
    for(int i=0; i<9; i++) h_kernel[i] = 1.0f/9.0f;

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    // copy to constant memory 
    cudaMemcpyToSymbol(c_kernel, h_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((WIDTH + TILE_SIZE - 1) / TILE_SIZE, (HEIGHT + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    convolutionShared<<<blocks, threads>>>(d_in, d_out, WIDTH, HEIGHT);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Optimized GPU Time (Shared Memory): %f ms\n", ms);

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out); free(h_kernel);
    return 0;
}