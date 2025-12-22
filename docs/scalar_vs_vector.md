# SpMV Performance Analysis: Scalar vs. Vector Kernels

## 1. Overview
This report presents a sensitivity analysis of Sparse Matrix-Vector Multiplication (SpMV) on an NVIDIA GPU (RTX 4050). We evaluate two primary thread-mapping strategies using a **Banded Matrix** (structured sparsity):
1. **Scalar Kernel (Thread-per-Row):** A naive approach where one CUDA thread processes one entire row.
2. **Vector Kernel (Warp-per-Row):** A more advanced approach where one Warp (32 threads) cooperatively processes a single row using Warp Shuffle instructions.

The goal is to analyze how matrix scale ($M$) and row width ($L$) affect GFLOPS and Effective Bandwidth.

---

## 2. Experimental Data

### Scalar Kernel (Thread-per-Row) Sensitivity
| M (Scale) | L (Width) | Time (ms) | GFLOPS | Bandwidth (GB/s) |
| :--- | :--- | :--- | :--- | :--- |
| 10000 | 8 | 0.0112 | 14.23 | 67.59 |
| 10000 | 32 | 0.0191 | 33.55 | 140.48 |
| 10000 | 64 | 0.0319 | 40.09 | 164.13 |
| 10000 | 128 | 0.0599 | 42.77 | 173.10 |
| 10000 | 256 | 0.1173 | 43.65 | 175.61 |
| 50000 | 8 | 0.0114 | 70.32 | 334.02 |
| 50000 | 32 | 0.0747 | 42.86 | 179.46 |
| 50000 | 64 | 0.1467 | 43.62 | 178.56 |
| 50000 | 128 | 0.3734 | 34.28 | 138.74 |
| 50000 | 256 | 1.2705 | 20.15 | 81.07 |
| 100000 | 8 | 0.0173 | 92.30 | 438.41 |
| 100000 | 32 | 0.1358 | 47.12 | 197.30 |
| 100000 | 64 | 0.2980 | 42.96 | 175.86 |
| 100000 | 128 | 0.6902 | 37.09 | 150.10 |
| 100000 | 256 | 2.3370 | 21.91 | 88.15 |
| 400000 | 8 | 0.1669 | 38.34 | 182.11 |
| 400000 | 32 | 0.5907 | 43.34 | 181.48 |
| 400000 | 64 | 1.1492 | 44.55 | 182.38 |
| 400000 | 128 | 2.7338 | 37.46 | 151.59 |
| 400000 | 256 | 9.6250 | 21.28 | 85.61 |

### Vector Kernel (Warp-per-Row) Sensitivity
| M (Scale) | L (Width) | Time (ms) | GFLOPS | Bandwidth (GB/s) |
| :--- | :--- | :--- | :--- | :--- |
| 10000 | 8 | 0.0117 | 13.66 | 64.89 |
| 10000 | 32 | 0.0094 | 67.79 | 283.86 |
| 10000 | 64 | 0.0139 | 91.84 | 375.99 |
| 10000 | 128 | 0.0144 | 177.72 | 719.22 |
| 10000 | 256 | 0.0224 | 228.55 | 919.54 |
| 50000 | 8 | 0.0375 | 21.31 | 101.20 |
| 50000 | 32 | 0.0371 | 86.23 | 361.09 |
| 50000 | 64 | 0.0931 | 68.78 | 281.57 |
| 50000 | 128 | 0.2832 | 45.20 | 182.91 |
| 50000 | 256 | 0.5554 | 46.09 | 185.45 |
| 100000 | 8 | 0.0663 | 24.12 | 114.56 |
| 100000 | 32 | 0.1268 | 50.48 | 211.38 |
| 100000 | 64 | 0.2890 | 44.29 | 181.29 |
| 100000 | 128 | 0.5620 | 45.56 | 184.36 |
| 100000 | 256 | 1.1073 | 46.24 | 186.03 |
| 400000 | 8 | 0.4470 | 14.32 | 68.02 |
| 400000 | 32 | 0.6880 | 37.21 | 155.82 |
| 400000 | 64 | 1.1469 | 44.64 | 182.76 |
| 400000 | 128 | 2.2352 | 45.81 | 185.40 |
| 400000 | 256 | 4.4222 | 46.31 | 186.33 |

---

## 3. Comparative Analysis

### A. The "Cache Illusion" at Small Scales
At $M = 10,000$, we observe bandwidth values as high as **919 GB/s**, which exceeds the physical limit of VRAM (approx. 192 GB/s for RTX 4050).
* **Cause:** The entire matrix and vectors fit within the **L2 Cache**. The kernel is measuring SRAM throughput rather than DRAM bandwidth.
* **Observation:** As $M$ increases to $400,000$, data spills into DRAM, and bandwidth stabilizes at realistic hardware limits.

### B. The Failure of the Vector Kernel for Short Rows ($L=8$)
At large scales ($M=400k$), for $L=8$:
* **Scalar:** 182.11 GB/s
* **Vector:** 68.02 GB/s
* **Analysis:** The Vector kernel assigns 32 threads per row. If $L=8$, **24 threads (75%) remain idle** after a single iteration. This leads to massive under-utilization of the SIMT hardware. The Scalar kernel avoids this overhead, and because the banded matrix is structured, the L2 cache mitigates the impact of uncoalesced memory access.

### C. The Dominance of the Vector Kernel for Long Rows ($L \ge 64$)
At large scales ($M=400k$), for $L=256$:
* **Scalar:** 85.61 GB/s (Significant drop)
* **Vector:** 186.33 GB/s (Close to theoretical peak)
* **Analysis:** As rows get longer, **Memory Coalescing** becomes the dominant factor. The Vector kernel ensures that all 32 threads in a warp access contiguous memory locations in the `values` and `col_indices` arrays. Conversely, the Scalar kernel suffers from **Cache Thrashing**; threads in the same warp compete for cache lines as they stride across different rows, leading to high miss rates.

---

## 4. Conclusion
The optimal SpMV kernel depends heavily on the matrix sparsity pattern:
1. **For very short rows ($L < 32$):** Use the Scalar kernel or a Sub-warp strategy to maintain high thread utilization.
2. **For standard or long rows ($L \ge 32$):** Use the Vector kernel to leverage memory coalescing and maximize DRAM bandwidth.
3. **Hardware Constraint:** The experimental peak bandwidth of ~186 GB/s reflects the physical saturation of the RTX 4050's memory bus.

---
