#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>

#define TILE_SIZE 1024


__global__ void spmv_csr_tiled_shared_kernel(
    int num_rows,
    int num_cols,
    int *row_ptr,
    int *col_indices,
    float *values,
    float *x,
    float *y
) {

    __shared__ float x_shared[TILE_SIZE];

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int row = warp_id;

    float row_sum = 0.0f;


    for (int tile_start = 0; tile_start < num_cols; tile_start += TILE_SIZE) {
        


        for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
            int global_x_idx = tile_start + i;
            if (global_x_idx < num_cols) {
                x_shared[i] = x[global_x_idx];
            } else {
                x_shared[i] = 0.0f;
            }
        }


        __syncthreads();


        if (row < num_rows) {
            int start = row_ptr[row];
            int end = row_ptr[row + 1];


            for (int i = start + lane_id; i < end; i += 32) {
                int col = col_indices[i];
                if (col >= tile_start && col < tile_start + TILE_SIZE) {
                    row_sum += values[i] * x_shared[col - tile_start];
                }
            }
        }


        __syncthreads();
    }

    // 3. Warp Shuffle Reduction
    if (row < num_rows) {
        for (int offset = 16; offset > 0; offset /= 2) {
            row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
        }
        if (lane_id == 0) {
            y[row] = row_sum;
        }
    }
}


void spmv_cpu(int M, const int* row_ptr, const int* col_indices, const float* values, const float* x, float* y) {
    for (int i = 0; i < M; ++i) {
        double sum = 0;
        for (int j = row_ptr[i]; j < row_ptr[i+1]; ++j) {
            sum += (double)values[j] * x[col_indices[j]];
        }
        y[i] = (float)sum;
    }
}

int main() {

    const int M = 30000; 
    const int avg_nnz = 128;
    const int NNZ = M * avg_nnz;

    std::vector<int> h_row_ptr(M + 1);
    std::vector<int> h_col_indices(NNZ);
    std::vector<float> h_values(NNZ);
    std::vector<float> h_x(M);
    std::vector<float> h_y_cpu(M, 0.0f);
    std::vector<float> h_y_gpu(M, 0.0f);


    srand(42);
    for (int i = 0; i < M; i++) {
        h_row_ptr[i] = i * avg_nnz;
        h_x[i] = (float)rand() / RAND_MAX;
        for (int j = 0; j < avg_nnz; j++) {
            h_col_indices[i * avg_nnz + j] = rand() % M;
            h_values[i * avg_nnz + j] = (float)rand() / RAND_MAX;
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


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 block(256); 
    dim3 grid((M + (block.x / 32) - 1) / (block.x / 32));

    cudaEventRecord(start);
    for(int i = 0; i < 50; i++) {
        spmv_csr_tiled_shared_kernel<<<grid, block>>>(M, M, d_row_ptr, d_col_indices, d_values, d_x, d_y);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / 50.0f;


    cudaMemcpy(h_y_gpu.data(), d_y, M * sizeof(float), cudaMemcpyDeviceToHost);
    spmv_cpu(M, h_row_ptr.data(), h_col_indices.data(), h_values.data(), h_x.data(), h_y_cpu.data());

    bool correct = true;
    for (int i = 0; i < M; i++) {
        if (fabs(h_y_gpu[i] - h_y_cpu[i]) > 1e-2) {
            correct = false;
            break;
        }
    }

    std::cout << "\n--- V5 Tiled Shared Memory SpMV Result ---" << std::endl;
    std::cout << "Matrix: " << M << " x " << M << ", NNZ: " << NNZ << std::endl;
    std::cout << "Status: " << (correct ? "PASS" : "FAIL") << std::endl;
    std::cout << "Avg Time: " << avg_ms << " ms" << std::endl;
    std::cout << "Throughput: " << (2.0 * NNZ / 1e9) / (avg_ms / 1000.0) << " GFLOPS" << std::endl;

    cudaFree(d_row_ptr); cudaFree(d_col_indices); cudaFree(d_values); cudaFree(d_x); cudaFree(d_y);
    return 0;
}
