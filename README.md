# 🖼️ High-Performance Image Convolution with CUDA

This project implements and analyzes a 2D Image Convolution filter using NVIDIA CUDA. The implementation provides a comprehensive performance comparison between a **Sequential CPU** approach, a **Naive GPU** kernel, and an **Optimized GPU** version utilizing **Shared Memory** and **Constant Memory**.

## 🚀 Performance Results
The following benchmarks were recorded on a system with an **NVIDIA GeForce GTX 1070** for a **1024x1024** image using a **3x3** kernel.

| Implementation | Execution Time (ms) | Speedup (vs CPU) |
| :--- | :--- | :--- |
| Sequential CPU | 478.0000 ms | 1.0x |
| Naive CUDA Kernel | 4.4736 ms | 106.8x |
| **Optimized (Shared + Constant)** | **0.6881 ms** | **694.6x** |

---

## 💡 Technical Analysis & Optimizations

### 1. Constant Memory for Filter Kernels
The convolution filter (3x3 kernel) is stored in the GPU's `__constant__` memory. Unlike global memory, constant memory is cached and optimized for cases where all threads in a warp read the same address simultaneously. This "broadcasting" mechanism significantly reduces memory latency during the convolution sum.

### 2. Shared Memory Tiling (The Tiled Approach)
In a standard convolution, each pixel is read multiple times by its neighbors. To avoid redundant and slow Global Memory accesses, we load the image into **Shared Memory tiles**. 
- Each thread block loads its own "tile" of the image into the high-speed, on-chip Shared Memory.
- This transition reduces the bandwidth bottleneck and allows the GTX 1070 to process pixels at near-L1 cache speeds.

### 3. Halo Region & Boundary Handling
To compute the convolution for pixels at the edge of a tile, data from neighboring tiles is required. We implemented a **"Halo" region** in our Shared Memory allocation:
- **Tile Size:** 16x16
- **Shared Memory Size:** `(16 + KernelRadius*2) x (16 + KernelRadius*2)`
- This ensures that every thread has local access to all required neighbors without needing to go back to Global Memory.

### 4. Coalesced Memory Access
The data loading process from Global Memory to Shared Memory is designed to be **coalesced**. Threads in a warp access contiguous memory addresses, allowing the hardware to combine multiple memory requests into a single transaction, maximizing the effective bandwidth.

---

## 🛠️ Requirements & Setup

- **Hardware:** NVIDIA GPU (Tested on GTX 1070)
- **Toolchain:** CUDA Toolkit (nvcc compiler)

### Compilation:
```bash
nvcc -o convolution convolution_filter.cu
