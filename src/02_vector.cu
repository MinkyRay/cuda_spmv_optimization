#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>


__global__ void spmv_csr_vector_kernel(
    int num_rows, int *row_ptr, int *col_indices, 
    float *values, float *x, float *y
) {
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

        // Warp Shuffle Reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane_id == 0) y[row] = sum;
    }
}


void run_vector_benchmark(int M, int nnz_per_row) {
    int N = M;
    int NNZ = M * nnz_per_row;


    std::vector<int> h_row_ptr(M + 1);
    std::vector<int> h_col_indices(NNZ);
    std::vector<float> h_values(NNZ);
    std::vector<float> h_x(N, 1.0f);
    std::vector<float> h_y_gpu(M, 0.0f);

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
    int rowsPerBlock = blockSize / 32;
    int gridSize = (M + rowsPerBlock - 1) / rowsPerBlock;

    // Warm-up
    for (int i = 0; i < 10; i++) {
        spmv_csr_vector_kernel<<<gridSize, blockSize>>>(M, d_row_ptr, d_col_indices, d_values, d_x, d_y);
    }
    cudaDeviceSynchronize();


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    const int iterations = 100;
    for (int i = 0; i < iterations; i++) {
        spmv_csr_vector_kernel<<<gridSize, blockSize>>>(M, d_row_ptr, d_col_indices, d_values, d_x, d_y);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_total;
    cudaEventElapsedTime(&ms_total, start, stop);
    float ms_avg = ms_total / iterations;


    cudaMemcpy(h_y_gpu.data(), d_y, M * sizeof(float), cudaMemcpyDeviceToHost);
    bool pass = true;
    for(int i = 0; i < M; i++) {
        if (std::abs(h_y_gpu[i] - (float)nnz_per_row) > 1e-3) {
            pass = false;
            break;
        }
    }


    double gflops = (2.0 * NNZ) / (ms_avg * 1e6);
    double bytes = (double)((M + 1 + NNZ) * sizeof(int) + (NNZ + N + M) * sizeof(float));
    double bandwidth = bytes / (ms_avg * 1e6);

    std::cout << std::setw(8) << M << ", " 
              << std::setw(10) << ms_avg << ", " 
              << std::setw(10) << gflops << ", " 
              << std::setw(10) << bandwidth << (pass ? " [PASS]" : " [FAIL]") << std::endl;

    cudaFree(d_row_ptr); cudaFree(d_col_indices); cudaFree(d_values); cudaFree(d_x); cudaFree(d_y);
}

int main() {
    std::cout << "--- Vector CSR Sensitivity Analysis ---" << std::endl;
    std::cout << "M_Size,   Avg_Time(ms),  GFLOPS,   Bandwidth(GB/s)" << std::endl;
    
    std::vector<int> test_sizes = {10000, 20000, 40000, 80000, 160000, 320000};
    for (int M : test_sizes) {
        run_vector_benchmark(M, 32);
    }
    return 0;
}
