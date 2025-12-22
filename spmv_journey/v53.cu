#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>

#define TILE_SIZE 1024

// --- V5-Pro Kernel: 引入 Early Exit 优化 ---
__global__ void spmv_csr_tiled_sorted_kernel(int num_rows, int num_cols, int *row_ptr, int *col_indices, float *values, float *x, float *y) {
    __shared__ float x_shared[TILE_SIZE];

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row = tid / 32;
    int lane_id = tid % 32;
    float row_sum = 0.0f;

    for (int tile_start = 0; tile_start < num_cols; tile_start += TILE_SIZE) {
        // 1. 协作搬运 (合并访问)
        for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
            int idx = tile_start + i;
            x_shared[i] = (idx < num_cols) ? x[idx] : 0.0f;
        }
        __syncthreads();

        // 2. 局部计算 (带 Early Exit)
        if (row < num_rows) {
            int start = row_ptr[row];
            int end = row_ptr[row + 1];

            for (int i = start + lane_id; i < end; i += 32) {
                int col = col_indices[i];
                
                // --- 核心优化：有序性判定 ---
                // 如果当前列索引已经超出了 Tile 范围，后面一定也超出了，直接跳出
                if (col >= tile_start + TILE_SIZE) {
                    break; 
                }
                
                // 只有在当前 Tile 范围内的才累加
                if (col >= tile_start) {
                    row_sum += values[i] * x_shared[col - tile_start];
                }
            }
        }
        __syncthreads();
    }

    // 3. 规约与写回
    if (row < num_rows) {
        for (int offset = 16; offset > 0; offset /= 2) 
            row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
        if (lane_id == 0) y[row] = row_sum;
    }
}

// --- 自动化测试逻辑 ---
void run_benchmark_pro(int M) {
    const int avg_nnz = 128;
    const int NNZ = M * avg_nnz;
    
    std::vector<int> h_row_ptr(M + 1);
    std::vector<int> h_col_indices(NNZ);
    std::vector<float> h_values(NNZ), h_x(M);
    
    // 初始化数据
    for (int i = 0; i < M; i++) {
        h_row_ptr[i] = i * avg_nnz;
        h_x[i] = 1.0f;
        
        // 先生成随机列
        std::vector<std::pair<int, float>> temp_row;
        for (int j = 0; j < avg_nnz; j++) {
            temp_row.push_back({rand() % M, 1.0f});
        }
        // --- 关键：在 Host 端对每一行进行排序 ---
        std::sort(temp_row.begin(), temp_row.end());
        
        for (int j = 0; j < avg_nnz; j++) {
            h_col_indices[i * avg_nnz + j] = temp_row[j].first;
            h_values[i * avg_nnz + j] = temp_row[j].second;
        }
    }
    h_row_ptr[M] = NNZ;

    int *d_row_ptr, *d_col_indices;
    float *d_values, *d_x, *d_y;
    cudaMalloc(&d_row_ptr, (M + 1) * sizeof(int));
    cudaMalloc(&d_col_indices, NNZ * sizeof(int));
    cudaMalloc(&d_values, NNZ * sizeof(float));
    cudaMalloc(&d_x, M * sizeof(float));
    cudaMalloc(&d_y, M * sizeof(float));

    cudaMemcpy(d_row_ptr, h_row_ptr.data(), (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, h_col_indices.data(), NNZ * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values.data(), NNZ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), M * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((M + (block.x / 32) - 1) / (block.x / 32));

    // Warmup
    for(int i = 0; i < 5; i++) 
        spmv_csr_tiled_sorted_kernel<<<grid, block>>>(M, M, d_row_ptr, d_col_indices, d_values, d_x, d_y);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i = 0; i < 20; i++) 
        spmv_csr_tiled_sorted_kernel<<<grid, block>>>(M, M, d_row_ptr, d_col_indices, d_values, d_x, d_y);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / 20.0f;
    float gflops = (2.0f * NNZ / 1e9) / (avg_ms / 1000.0f);

    std::cout << std::setw(10) << M << "," << std::setw(12) << avg_ms << "," << std::setw(10) << gflops << std::endl;

    cudaFree(d_row_ptr); cudaFree(d_col_indices); cudaFree(d_values); cudaFree(d_x); cudaFree(d_y);
}

int main() {
    std::cout << "V5-Pro Sensitivity Analysis (Sorted + Early Exit)" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "M_Size    , Avg_Time(ms), GFLOPS" << std::endl;

    for (int m = 10000; m <= 100000; m += 10000) {
        run_benchmark_pro(m);
    }
    return 0;
}