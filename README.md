# High-Performance CUDA SpMV: Architectural Optimization Journey

![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?style=flat-square&logo=nvidia)
![Platform](https://img.shields.io/badge/Platform-RTX%204050%20(Ada)-blue?style=flat-square)
![Memory Bandwidth](https://img.shields.io/badge/Effective%20Bandwidth-~186%20GB/s-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> **"In sparse computation, the bottleneck is rarely the arithmetic; it is the efficiency of data movement across the memory hierarchy."**

## 📖 Overview

This repository documents a systematic optimization of the **Sparse Matrix-Vector Multiplication (SpMV)** operator on the NVIDIA Ada Lovelace architecture (RTX 4050). 

Starting from a baseline scalar implementation, we progress through six evolutionary stages—addressing **Memory Coalescing**, **Thread Idling**, **Load Imbalance**, and **Cache Thrashing**. By leveraging hardware-specific features like **Warp Shuffle**, **Shared Memory Tiling**, and **2D Indexing**, we achieve nearly **95% of the theoretical DRAM bandwidth**.

---

## 🚀 Optimization Roadmap

This project organizes optimizations into six source kernels, each targeting a specific hardware bottleneck.

| Kernel | Source | Bottleneck Solved | Architectural Principle |
| :--- | :--- | :--- | :--- |
| **V1: Scalar** | [`01_scalar.cu`](./src/01_scalar.cu) | Baseline | **Thread-per-Row** (Naive mapping) |
| **V2: Vector** | [`02_vector.cu`](./src/02_vector.cu) | DRAM Throughput | **Warp-per-Row** (Memory Coalescing) |
| **V3: Sub-Warp** | [`03_subwarp.cu`](./src/03_subwarp.cu) | SIMT Under-utilization | **Sub-Warp Partitioning** (SWS) |
| **V4: Merge-Path** | [`04_mergepath.cu`](./src/04_mergepath.cu) | Extreme Load Imbalance | **Work-Stealing / 2D Partition** |
| **V5: Shared-Mem** | [`05_shared_mem.cu`](./src/05_shared_mem.cu) | Random $x$ Access | **Shared Memory Tiling** ($x$ vector reuse) |
| **V6: Indexed-2D** | [`06_indexed_2d.cu`](./src/06_indexed_2d.cu) | Redundant Scanning | **2D Index Mapping** (Optimized CSR) |

---

## 📊 Performance & Sensitivity Analysis

A core value of this repository is the rigorous characterization of kernel performance across varying matrix scales ($M$) and row densities ($L$).

### 🧠 Analysis 1: The $L=256$ Capacity Wall (Scalar vs. Vector)

We observed a distinct performance bifurcation at $L=256$ in banded matrices.
* **The Amortization Phase ($L < 256$):** Effective bandwidth grows as instruction overhead is amortized over more math operations.
* **The DRAM Collapse ($L > 256$):** As the working set exceeds the **L2 Cache capacity**, the system becomes DRAM-bound. In this regime, the **Vector Kernel** dominates due to strict **Memory Coalescing**, while the Scalar kernel collapses due to fragmented memory transactions.

### ⚡ Analysis 2: Finding the Sub-Warp "Sweet Spot"

For short rows typical in PDE solvers ($L=4 \sim 16$), standard Warp-32 kernels suffer from thread idling. Our Sub-warp sensitivity analysis identifies the optimal **Sub-Warp Size (SWS)**:
* **$L=8, SWS=2$**: Achieved **23.35 GFLOPS**, significantly outperforming standard warp mapping by maximizing concurrency per physical warp.

### 🏗️ Analysis 3: Shared Memory vs. 2D Indexing
Traditional CSR scanning can be redundant for certain sparsity patterns. 
* **V5 (Shared Memory)** focuses on caching the $x$-vector to mitigate L2 cache thrashing during unstructured access.
* **V6 (2D-Indexed)** provides a sophisticated mapping to resolve the trade-off between **Thread-Level Parallelism (TLP)** and **Instruction-Level Parallelism (ILP)**.

---

## 🛠️ Key Architectural Insights

### 1. Instruction Amortization
Longer rows provide higher **Arithmetic Intensity**. Increasing $L$ allows the GPU to overlap memory wait times with arithmetic execution, saturating the deep instruction pipelines of the Ada architecture.

### 2. Memory Coalescing & Transaction Fragmentation

Once data spills from L2 to DRAM, the cost of "non-coalesced" access increases 10x. Our Vector and Sub-warp kernels ensure that threads within a warp access contiguous 128-byte segments, saturating the memory bus at **186 GB/s**.

### 3. Load Balancing via Merge-Path
For matrices with skewed row distributions (e.g., social networks), static row-mapping fails. The **Merge-Path** algorithm implements a 2D binary search on the matrix coordinates to ensure every thread processes an identical number of non-zero elements, regardless of row length.

---

## 💻 Getting Started

### Prerequisites
* **Hardware:** NVIDIA GPU (RTX 40-series recommended for Ada-specific results)
* **Software:** CUDA Toolkit 12.0+, CMake 3.18+, C++17

### Build & Run
```bash
# Clone the repository
git clone [https://github.com/MinkyRay/cuda_spmv_optimization.git](https://github.com/MinkyRay/cuda_spmv_optimization.git)
cd cuda_spmv_optimization

# Build
mkdir build && cd build
cmake ..
make -j

# Run sensitivity benchmarks
./bin/spmv_bench --kernel subwarp --L 8
