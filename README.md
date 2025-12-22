# High-Performance SpMV: A Hardware-Software Co-design Journey

## 🚀 Overview
Sparse Matrix-Vector Multiplication (SpMV) is a critical kernel in scientific computing, graph analytics, and Large Language Model (LLM) compression. Mathematically represented as $y = Ax$ where $A$ is sparse, its performance on modern GPUs is primarily bottlenecked by **memory bandwidth** and **irregular sparsity patterns**.

This repository documents my systematic approach to optimizing SpMV on NVIDIA GPUs (Ada Lovelace architecture, RTX 4050/4060), progressing from a naive baseline to architecture-aware, state-of-the-art kernels.

---

## 🛠 Kernel Evolution & Optimization Taxonomy

### V1: Scalar CSR (The Baseline)
* **Strategy**: One thread per row.
* **Bottleneck**: **Uncoalesced memory access**. Threads within a warp access `values` and `col_indices` at non-contiguous offsets, leading to excessive memory transactions.
* **Insight**: Identified that the "one-thread-per-row" model fails to exploit the 128-byte L1/L2 cache line granularity.

### V2: Vector CSR (Warp-level Parallelism)
* **Strategy**: One warp (32 threads) per row using `__shfl_down_sync` for intra-warp reduction.
* **Improvement**: Achieved **100% memory coalescing** for matrix data. 
* **Hardware Mapping**: Aligned memory requests with the SIMT execution model, significantly increasing effective bandwidth.

### V3: Adaptive Sub-Warp (Heuristic Balancing)
* **Strategy**: Dynamically assign $2^n$ threads (sub-warps) per row based on the matrix's average Non-Zero (NNZ) count.
* **Optimization**: Reduced resource under-utilization for short rows while maintaining high throughput for medium-length rows.

### V4: Merge-Path SpMV (The State-of-the-Art)
* **Strategy**: Treats the SpMV problem as a 2D search in a decision space of size $(M + NNZ)$. It partitions the "staircase" path equally across SMs using binary search intersection.
* **Scientific Value**: Effectively eliminates **Load Imbalance** in power-law/skewed matrices (e.g., social graphs), where a few rows contain the majority of non-zeros.
* **Technical Challenge**: Implemented a robust "Staircase Walk" logic to handle cross-thread reductions via hardware-level atomic primitives.



### V5: Tiled CSR with Shared Memory (Cache-Aware)
* **Strategy**: Vertical tiling of the column space to cache segments of vector $x$ into **Shared Memory (On-chip SRAM)**.
* **Optimization**: Mitigates **L2 Cache Thrashing** caused by the random access pattern of $x[col\_indices[j]]$.
* **Architecture Insight**: Manually managed the storage hierarchy to bypass the limitations of hardware-managed LRU cache policies.

---

## 📊 Performance Analysis

I evaluate kernel efficiency using the **Roofline Model**, focusing on the relationship between Arithmetic Intensity and Peak Bandwidth.

$$Arithmetic\ Intensity = \frac{2 \times NNZ}{Bytes(A) + Bytes(x) + Bytes(y)}$$

| Kernel Version | GFLOPS | Bandwidth Utilization | Optimization Focus |
| :--- | :--- | :--- | :--- |
| V1: Scalar | ~5-10 | < 5% | Baseline |
| V2: Vector | ~150-200 | ~60-70% | Coalescing |
| V4: Merge-Path | ~15-30 | Stability | Load Balancing |
| V5: Tiled | TBD | Latency Hiding | Cache Management |



---

## 🧠 Research Context & Academic Aspirations

As a Master's student in **Computational Mathematics at Nankai University**, I view High-Performance Computing as the bridge between abstract numerical analysis and physical hardware limits. 

This project reflects my core research interests:
1.  **Numerical Linear Algebra**: Optimizing iterative solvers (CG, GMRES) for large-scale systems.
2.  **Architecture-Aware Design**: Exploiting L1/Shared Memory and Tensor Cores for scientific kernels.
3.  **LLM Acceleration**: Applying sparse optimization techniques to low-bit quantization (GPTQ/AWQ) inference.

I am actively seeking PhD opportunities in **HPC, Computational Mathematics, or Numerical Analysis** for Fall 2026/2027, with a specific interest in research that pushes the boundaries of hardware-efficient scientific computing.

---

## 🏗 Setup & Build
```bash
# Environment: CUDA 12.x+ / NVCC
# Compilation
nvcc -O3 -arch=sm_89 main.cu -o spmv_benchmark

# Running the benchmark
./spmv_benchmark --matrix=west0497.mtx
