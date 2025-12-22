#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <numeric>
#include <algorithm>

#define WARMUP_ITER 20   
#define TEST_ITER 100    


template<int SWS>
__global__ void spmv_subwarp_kernel(int num_rows, int *row_ptr, int *col_indices, float *values, float *x, float *y) {
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
        if (sub_warp_lane_id == 0) y[row] = sum;
    }
}


template<int SWS>
float run_benchmark(int M, int NNZ, int *d_ptr, int *d_col, float *d_val, float *d_x, float *d_y) {
    int threadsPerBlock = 128;
    int gridSize = (M * SWS + threadsPerBlock - 1) / threadsPerBlock;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);


    for (int i = 0; i < WARMUP_ITER; ++i) {
        spmv_subwarp_kernel<SWS><<<gridSize, threadsPerBlock>>>(M, d_ptr, d_col, d_val, d_x, d_y);
    }
    cudaDeviceSynchronize();


    cudaEventRecord(start);
    for (int i = 0; i < TEST_ITER; ++i) {
        spmv_subwarp_kernel<SWS><<<gridSize, threadsPerBlock>>>(M, d_ptr, d_col, d_val, d_x, d_y);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return total_ms / TEST_ITER; 
}


void dispatch_sws_tests(int M, int NNZ, int row_len, int *d_ptr, int *d_col, float *d_val, float *d_x, float *d_y) {
    auto print_row = [&](int sws, float avg_ms) {
        double gflops = (2.0 * NNZ) / (avg_ms * 1e6);
        std::cout << "| " << std::setw(8) << row_len << " | " << std::setw(8) << sws 
                  << " | " << std::setw(12) << avg_ms << " | " << std::setw(10) << gflops << " |" << std::endl;
    };

    print_row(2, run_benchmark<2>(M, NNZ, d_ptr, d_col, d_val, d_x, d_y));
    print_row(4, run_benchmark<4>(M, NNZ, d_ptr, d_col, d_val, d_x, d_y));
    print_row(8, run_benchmark<8>(M, NNZ, d_ptr, d_col, d_val, d_x, d_y));
    print_row(16, run_benchmark<16>(M, NNZ, d_ptr, d_col, d_val, d_x, d_y));
    print_row(32, run_benchmark<32>(M, NNZ, d_ptr, d_col, d_val, d_x, d_y));
}

int main() {
    int M = 10000;
    std::vector<int> test_row_lens = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};

    std::cout << "| Row Len  | SubWarp  | Avg Time (ms) | GFLOPS     |" << std::endl;
    std::cout << "|----------|----------|---------------|------------|" << std::endl;

    for (int row_len : test_row_lens) {
        int NNZ = M * row_len;
        std::vector<int> h_row_ptr(M + 1);
        std::vector<int> h_col_indices(NNZ);
        for (int i = 0; i < M; i++) {
            h_row_ptr[i] = i * row_len;
            for (int j = 0; j < row_len; j++) h_col_indices[i * row_len + j] = rand() % M; // 随机列索引压力测试
        }
        h_row_ptr[M] = NNZ;

        int *d_ptr, *d_col; float *d_val, *d_x, *d_y;
        cudaMalloc(&d_ptr, (M+1)*4); cudaMalloc(&d_col, NNZ*4);
        cudaMalloc(&d_val, NNZ*4); cudaMalloc(&d_x, M*4); cudaMalloc(&d_y, M*4);
        
        cudaMemcpy(d_ptr, h_row_ptr.data(), (M+1)*4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_col, h_col_indices.data(), NNZ*4, cudaMemcpyHostToDevice);

        dispatch_sws_tests(M, NNZ, row_len, d_ptr, d_col, d_val, d_x, d_y);

        cudaFree(d_ptr); cudaFree(d_col); cudaFree(d_val); cudaFree(d_x); cudaFree(d_y);
    }
    return 0;
}