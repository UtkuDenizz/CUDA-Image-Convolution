# CUDA Image Convolution Optimization

This project implements and optimizes a 2D Image Convolution filter using NVIDIA CUDA. The implementation compares a **Sequential CPU** version, a **Naive GPU** version, and an **Optimized GPU** version using **Shared Memory** and **Constant Memory**.

## 🚀 Performance Results
Tests were conducted on a system with an **NVIDIA GeForce GTX 1070**.

| Implementation | Execution Time (ms) | Speedup (vs CPU) |
| :--- | :--- | :--- |
| CPU Sequential | ~500.000 ms (Est.) | 1x |
| Naive CUDA | ~5.800 ms (Est.) | ~86x |
| **Optimized (Shared + Constant)** | **0.688 ms** | **~726x** |

## 🛠️ Optimization Techniques
- **Constant Memory:** The convolution kernel (filter) is stored in `__constant__` memory. This allows for fast, cached broadcasting of filter weights to all threads simultaneously.
- **Shared Memory Tiling:** Image pixels are loaded into high-speed Shared Memory tiles. This significantly reduces Global Memory traffic since each pixel is accessed multiple times by neighboring threads.
- **Halo Handling:** The shared memory allocation includes a "halo" region (`TILE_SIZE + KERNEL_SIZE - 1`) to ensure boundary pixels are available for the convolution calculation within each block.
- **Coalesced Access:** Memory transfers from Global to Shared memory are designed to be coalesced, maximizing the available bandwidth of the GTX 1070.

## 💻 Compilation & Execution
```bash
nvcc -o convolution convolution_filter.cu
./convolution
