#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <numeric>


template<int SWS>
__global__ void spmv_subwarp_kernel(
    int num_rows, int *row_ptr, int *col_indices, float *values, float *x, float *y
) {

    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int lane_id = thread_id % 32;
    

    int sub_warp_lane_id = lane_id % SWS;
    int row = thread_id / SWS;

    if (row < num_rows) {
        int start = row_ptr[row];
        int end = row_ptr[row + 1];
        float sum = 0.0f;


        for (int i = start + sub_warp_lane_id; i < end; i += SWS) {
            sum += values[i] * x[col_indices[i]];
        }


        for (int offset = SWS / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }


        if (sub_warp_lane_id == 0) {
            y[row] = sum;
        }
    }
}


template<int SWS>
void run_test(int M, int NNZ, int *d_ptr, int *d_col, float *d_val, float *d_x, float *d_y) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    int threadsPerBlock = 128;
    int totalThreadsNeeded = M * SWS; 
    int gridSize = (totalThreadsNeeded + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start);
    spmv_subwarp_kernel<SWS><<<gridSize, threadsPerBlock>>>(M, d_ptr, d_col, d_val, d_x, d_y);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    double gflops = (2.0 * NNZ) / (ms * 1e6);
    double bandwidth = ((M + 1 + NNZ) * sizeof(int) + (NNZ + M + M) * sizeof(float)) / (ms * 1e6);

    std::cout << std::left << std::setw(12) << SWS 
              << "| " << std::setw(12) << ms 
              << "| " << std::setw(12) << gflops 
              << "| " << bandwidth << " GB/s" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int M = 20000;
    int avg_nnz = 4; 
    int NNZ = M * avg_nnz;

    std::cout << "Starting Sensitivity Analysis: M=" << M << ", Avg NNZ/row=" << avg_nnz << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;
    std::cout << "SubWarpSize | Time (ms)   | GFLOPS      | Bandwidth (GB/s)" << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;


    std::vector<int> h_row_ptr(M + 1);
    std::vector<int> h_col_indices(NNZ);
    std::vector<float> h_values(NNZ, 1.0f);
    std::vector<float> h_x(M, 1.0f);
    
    for (int i = 0; i < M; i++) {
        h_row_ptr[i] = i * avg_nnz;
        for (int j = 0; j < avg_nnz; j++) {
            h_col_indices[i * avg_nnz + j] = rand() % M; 
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


    run_test<2>(M, NNZ, d_row_ptr, d_col_indices, d_values, d_x, d_y);
    run_test<4>(M, NNZ, d_row_ptr, d_col_indices, d_values, d_x, d_y);
    run_test<8>(M, NNZ, d_row_ptr, d_col_indices, d_values, d_x, d_y);
    run_test<16>(M, NNZ, d_row_ptr, d_col_indices, d_values, d_x, d_y);
    run_test<32>(M, NNZ, d_row_ptr, d_col_indices, d_values, d_x, d_y);


    cudaFree(d_row_ptr); cudaFree(d_col_indices);
    cudaFree(d_values); cudaFree(d_x); cudaFree(d_y);

    return 0;
}