#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <string>
#include <cmath>

// --- Kernel 1: Scalar (1 Thread per Row) ---
__global__ void spmv_csr_scalar_kernel(int num_rows, int *row_ptr, int *col_indices, float *values, float *x, float *y) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < num_rows) {
        int start = row_ptr[row];
        int end = row_ptr[row + 1];
        float tmp = 0.0f;
        for (int i = start; i < end; ++i) {
            tmp += values[i] * x[col_indices[i]];
        }
        y[row] = tmp;
    }
}

// --- Kernel 2: Vector (1 Warp per Row) ---
__global__ void spmv_csr_vector_kernel(int num_rows, int *row_ptr, int *col_indices, float *values, float *x, float *y) {
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = thread_id / 32;
    int lane_id = thread_id % 32;
    int row = warp_id;
    if (row < num_rows) {
        int start = row_ptr[row];
        int end = row_ptr[row + 1];
        float sum = 0.0f;
        for (int i = start + lane_id; i < end; i += 32) {
            sum += values[i] * x[col_indices[i]];
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane_id == 0) y[row] = sum;
    }
}

struct Result {
    int M, L;
    float time_ms;
    double gflops;
    double bandwidth;
};

// 核心测试逻辑
Result run_test(int M, int L, bool use_vector) {
    int N = M;
    int NNZ = M * L;
    std::vector<int> h_row_ptr(M + 1);
    std::vector<int> h_col_indices(NNZ);
    std::vector<float> h_values(NNZ, 1.0f);
    std::vector<float> h_x(N, 1.0f);

    for (int i = 0; i < M; i++) {
        h_row_ptr[i] = i * L;
        for (int j = 0; j < L; j++) {
            h_col_indices[i * L + j] = (i + j) % N;
        }
    }
    h_row_ptr[M] = NNZ;

    int *d_row, *d_col; float *d_val, *d_x, *d_y;
    cudaMalloc(&d_row, (M+1)*sizeof(int));
    cudaMalloc(&d_col, NNZ*sizeof(int));
    cudaMalloc(&d_val, NNZ*sizeof(float));
    cudaMalloc(&d_x, N*sizeof(float));
    cudaMalloc(&d_y, M*sizeof(float));

    cudaMemcpy(d_row, h_row_ptr.data(), (M+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, h_col_indices.data(), NNZ*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, h_values.data(), NNZ*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), N*sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize;
    if (use_vector) {
        int rowsPerBlock = blockSize / 32;
        gridSize = (M + rowsPerBlock - 1) / rowsPerBlock;
    } else {
        gridSize = (M + blockSize - 1) / blockSize;
    }

    // Warm-up
    for(int i=0; i<10; i++){
        if(use_vector) spmv_csr_vector_kernel<<<gridSize, blockSize>>>(M, d_row, d_col, d_val, d_x, d_y);
        else spmv_csr_scalar_kernel<<<gridSize, blockSize>>>(M, d_row, d_col, d_val, d_x, d_y);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i=0; i<100; i++){
        if(use_vector) spmv_csr_vector_kernel<<<gridSize, blockSize>>>(M, d_row, d_col, d_val, d_x, d_y);
        else spmv_csr_scalar_kernel<<<gridSize, blockSize>>>(M, d_row, d_col, d_val, d_x, d_y);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / 100.0f;

    double gflops = (2.0 * NNZ) / (avg_ms * 1e6);
    double bytes = (double)((M + 1 + NNZ) * sizeof(int) + (NNZ + N + M) * sizeof(float));
    double bandwidth = bytes / (avg_ms * 1e6);

    cudaFree(d_row); cudaFree(d_col); cudaFree(d_val); cudaFree(d_x); cudaFree(d_y);
    return {M, L, avg_ms, gflops, bandwidth};
}

void print_table(const std::string& title, const std::vector<Result>& results) {
    std::cout << "\n### " << title << "\n";
    std::cout << "| M (Scale) | L (Width) | Time (ms) | GFLOPS | Bandwidth (GB/s) |\n";
    std::cout << "| :--- | :--- | :--- | :--- | :--- |\n";
    for (const auto& r : results) {
        std::cout << "| " << r.M << " | " << r.L << " | " << std::fixed << std::setprecision(4) << r.time_ms 
                  << " | " << std::setprecision(2) << r.gflops << " | " << r.bandwidth << " |\n";
    }
}

int main() {
    std::vector<int> M_list = {10000, 50000, 100000, 400000};
    std::vector<int> L_list = {8, 32, 64, 128, 256};

    std::vector<Result> scalar_results, vector_results;

    for (int M : M_list) {
        for (int L : L_list) {
            scalar_results.push_back(run_test(M, L, false));
            vector_results.push_back(run_test(M, L, true));
        }
    }

    print_table("Scalar Kernel (Thread-per-Row) Sensitivity", scalar_results);
    print_table("Vector Kernel (Warp-per-Row) Sensitivity", vector_results);

    return 0;
}