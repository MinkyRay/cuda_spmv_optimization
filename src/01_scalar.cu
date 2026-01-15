#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>


__global__ void spmv_csr_scalar_kernel(
    int num_rows, 
    int *row_ptr, 
    int *col_indices, 
    float *values, 
    float *x, 
    float *y
){
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < num_rows){
        int start = row_ptr[row];
        int end = row_ptr[row + 1];
        float tmp = 0.0f;
        for (int i = start; i < end; ++i){
            tmp += values[i] * x[col_indices[i]];
        }
        y[row] = tmp;
    }
}

#include <iostream>
#include <vector>
#include <iomanip>


void run_benchmark(int M, int nnz_per_row) {
    int N = M; 
    int NNZ = M * nnz_per_row;

    std::vector<int> h_row_ptr(M + 1);
    std::vector<int> h_col_indices(NNZ);
    std::vector<float> h_values(NNZ);
    std::vector<float> h_x(N, 1.0f);
    std::vector<float> h_y(M, 0.0f);

    for (int i = 0; i < M; i++) {
        h_row_ptr[i] = i * nnz_per_row;
        for (int j = 0; j < nnz_per_row; j++) {
            h_col_indices[i * nnz_per_row + j] = (i + j) % N; 
            h_values[i * nnz_per_row + j] = 1.0f;
        }
    }
    h_row_ptr[M] = NNZ;

    int *d_row_ptr, *d_col_indices;
    float *d_values, *d_x, *d_y;
    cudaMalloc(&d_row_ptr, (M + 1) * sizeof(int));
    cudaMalloc(&d_col_indices, NNZ * sizeof(int));
    cudaMalloc(&d_values, NNZ * sizeof(float));
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, M * sizeof(float));

    cudaMemcpy(d_row_ptr, h_row_ptr.data(), (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, h_col_indices.data(), NNZ * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values.data(), NNZ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice);


    int blockSize = 256;

    int gridSize = (M + blockSize - 1) / blockSize; 

    // --- 1. Warm-up---

    for (int i = 0; i < 5; i++) {
        spmv_csr_scalar_kernel<<<gridSize, blockSize>>>(M, d_row_ptr, d_col_indices, d_values, d_x, d_y);
    }
    cudaDeviceSynchronize();

    // --- 2. Multi-iteration ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iterations = 100; 
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        spmv_csr_scalar_kernel<<<gridSize, blockSize>>>(M, d_row_ptr, d_col_indices, d_values, d_x, d_y);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_total = 0;
    cudaEventElapsedTime(&ms_total, start, stop);
    float ms_avg = ms_total / iterations;


    double gflops = (2.0 * NNZ) / (ms_avg * 1e6);
    double bytes = (double)((M + 1 + NNZ) * sizeof(int) + (NNZ + N + M) * sizeof(float));
    double bandwidth = bytes / (ms_avg * 1e6);


    std::cout << std::setw(8) << M << ", " 
              << std::setw(10) << ms_avg << ", " 
              << std::setw(10) << gflops << ", " 
              << std::setw(10) << bandwidth << std::endl;


    cudaFree(d_row_ptr); cudaFree(d_col_indices);
    cudaFree(d_values); cudaFree(d_x); cudaFree(d_y);
}

int main() {
    std::cout << "M_Size,   Avg_Time(ms),  GFLOPS,   Bandwidth(GB/s)" << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;


    std::vector<int> test_sizes = {10000, 20000, 40000, 80000, 160000, 320000};
    int nnz_per_row = 32;

    for (int M : test_sizes) {
        run_benchmark(M, nnz_per_row);
    }

    return 0;
}
