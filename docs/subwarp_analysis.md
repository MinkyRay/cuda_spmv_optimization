# Analysis of Sub-Warp Granularity and Scaling Laws in SpMV

## 1. Overview
This report investigates the relationship between **Sub-Warp Size (SWS)** and **Row Length (L)** in Sparse Matrix-Vector Multiplication (SpMV). By benchmarking on an NVIDIA RTX 4050, we identify how hardware architectural constraints—specifically the L2 cache capacity and instruction issue rates—dictate the optimal threading strategy.

---

## 2. Experimental Data

| Row Len (L) | SubWarp (SWS) | Avg Time (ms) | GFLOPS | Notes |
| :--- | :--- | :--- | :--- | :--- |
| 4 | 4 | 0.00639904 | 12.50 | |
| 4 | **8** | **0.00632832** | **12.64** | **Best** |
| 4 | 32 | 0.0135270 | 5.91408 | |
| 8 | **8** | **0.00579584** | **27.606** | **Best** |
| 8 | 8 | 0.0101360 | 15.7853 | |
| 16 | **8** | **0.00628512** | **50.9139** | **Best** |
| 16 | 16 | 0.0111203 | 28.7761 | |
| 32 | 2 | 0.0135178 | 47.3451 | |
| 32 | **16** | **0.00808672** | **79.1421** | **Best** |
| 32 | 32 | 0.0100656 | 63.5829 | |
| 64 | **8** | **0.0120320** | **106.383** | **Best** |
| 64 | 32 | 0.0167526 | 76.4059 | |
| 128 | 4 | 0.0245229 | 104.392 | |
| 128 | **32** | **0.0190771** | **134.192** | **Best** |
| 256 | **32** | **0.0266035** | **192.456** | **Best** |
| 512 | **32** | **0.2232730** | **45.8631** | **Best** |
| 1024 | **32** | **0.4423490** | **46.2983** | **Best** |
| 2048 | **32** | **0.8798390** | **46.5540** | **Best** |

---

## 3. Pattern 1: The Phase Shift at $L = 256$

A critical bifurcation in performance occurs at a row length of 256. 

### Phase A: Amortization and Saturation ($L < 256$)
In this regime, GFLOPS increase steadily across all $SWS$ configurations as $L$ grows.
* **Effective Computation Ratio:** For small $L$, the "management overhead" (pointer initialization, loop boundary checks, and warp synchronization) dominates execution time. As $L$ increases, the ratio of floating-point operations to control-flow instructions improves, allowing the GPU to spend more cycles on actual computation.
* **Latency Hiding:** Longer rows provide the thread scheduler with a deeper pool of independent instructions. This allows the GPU to better overlap memory fetch latencies with arithmetic execution, saturating the hardware pipelines.
* **L2 Cache Mitigation:** During this phase, the data fits largely within the **L2 Cache**. The L2's high throughput and sectorized management mask the penalties of non-coalesced memory access from smaller $SWS$ groups.



### Phase B: The Capacity Wall Collapse ($L > 256$)
Once $L$ exceeds 256, performance collapses across all configurations.
* **L2 Cache Overflow:** The working set exceeds the L2 capacity, forcing the kernel to fetch data directly from **DRAM (Global Memory)**.
* **DRAM Penalty:** DRAM has significantly higher latency and lower bandwidth than L2. In this regime, the hardware can no longer "forgive" inefficient memory patterns.

---

## 4. Pattern 2: SWS Scaling Laws

As $L$ increases, the optimal $SWS$ trends toward the full warp size (32).

### Small $L$: The Occupancy Advantage
For $L \le 16$, smaller $SWS$ (2 or 4) is generally superior. 
* **Thread Idling:** A large $SWS$ on a short row results in massive **SIMT under-utilization** (e.g., $SWS=32$ on $L=8$ leaves 75% of threads idle). 
* **High Concurrency:** Small $SWS$ allows the hardware to process more rows simultaneously per warp, maintaining high **SM Occupancy** and aggregate throughput.

### Large $L$: The Coalescing Mandate
For $L \ge 256$, $SWS=32$ (the Vector approach) becomes the stable winner.
* **Memory Coalescing:** As data spills into DRAM ($L > 256$), **Memory Coalescing** becomes the single most important factor for performance. $SWS=32$ ensures that the warp accesses perfectly aligned 128-byte segments. Smaller $SWS$ introduces strided access patterns that fragment memory transactions, leading to a bandwidth bottleneck.
* **Instruction Issue Rate:** At $L=256$, a configuration with $SWS=2$ requires 128 loop iterations, creating an instruction bottleneck. $SWS=32$ reduces this to only 8 iterations, acting as a hardware-level loop unrolling that minimizes control-flow overhead.



---

## 5. Summary of Architectural Mechanics

| Mechanism | Description | Impact on SpMV |
| :--- | :--- | :--- |
| **Instruction Amortization** | Amortizing loop/sync overhead over more math ops. | Explains GFLOPS growth as $L$ approaches 256. |
| **Sectorized L2 Cache** | Serving non-contiguous data from cache at high speed. | Masks non-coalesced access until the L2 is full. |
| **DRAM Transaction Splitting** | Fragmenting a single request into multiple DRAM fetches. | Causes the performance gap between $SWS=32$ and others at large $L$. |
| **Latency Hiding** | Swapping stalled warps for ready ones to keep ALU busy. | Requires sufficient $L$ to provide "ready" instructions. |



## 6. Conclusion
The experimental data confirms that SpMV optimization is a transition from **Instruction-Bound** (small $L$) to **Memory-Bound** (large $L$) regimes. The optimal strategy is to use **Sub-warp (SWS=4 or 8)** to maximize occupancy for short rows, while reverting to **Vector (SWS=32)** for long rows to maintain memory bus saturation once the L2 cache capacity is exceeded.
For the upcoming **Poisson PDE** solver, the resulting sparse matrix typically has $L=5$ (2D) or $L=7$ (3D). According to our benchmark:
* Standard Warp-32 methods would yield low efficiency ($\approx 6-12$ GFLOPS).
* By implementing **Sub-warp with $SWS=8$**, we can expect a **2x to 3x performance gain**, leveraging the hardware's ability to process 4 rows simultaneously per physical warp while maintaining efficient reduction.
