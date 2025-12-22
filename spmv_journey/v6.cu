#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>

#define TILE_SIZE 1024


__global__ void spmv_v5_ultimate_kernel(
    int num_rows, int num_cols, int num_tiles,
    int *row_ptr, int *col_indices, float *values, 
    int *tile_ptr, // tile_ptr[row * num_tiles + tile_id]
    float *x, float *y
) {
    __shared__ float x_shared[TILE_SIZE];
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row = tid / 32;
    int lane_id = tid % 32;
    float row_sum = 0.0f;

    for (int t = 0; t < num_tiles; ++t) {

        int tile_start_col = t * TILE_SIZE;
        for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
            int global_x_idx = tile_start_col + i;
            x_shared[i] = (global_x_idx < num_cols) ? x[global_x_idx] : 0.0f;
        }
        __syncthreads();


        if (row < num_rows) {

            int my_start = tile_ptr[row * num_tiles + t];
            int my_end   = (t + 1 < num_tiles) ? tile_ptr[row * num_tiles + t + 1] : row_ptr[row + 1];


            for (int i = my_start + lane_id; i < my_end; i += 32) {
                row_sum += values[i] * x_shared[col_indices[i] - tile_start_col];
            }
        }
        __syncthreads();
    }

    if (row < num_rows) {
        for (int offset = 16; offset > 0; offset /= 2) 
            row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
        if (lane_id == 0) y[row] = row_sum;
    }
}

void run_ultimate_benchmark(int M) {
    const int avg_nnz = 128;
    const int NNZ = M * avg_nnz;
    int num_tiles = (M + TILE_SIZE - 1) / TILE_SIZE;
    

    std::vector<int> h_row_ptr(M + 1);
    std::vector<int> h_col_indices(NNZ);
    std::vector<float> h_values(NNZ), h_x(M, 1.0f);

    for (int i = 0; i < M; i++) {
        h_row_ptr[i] = i * avg_nnz;
        std::vector<std::pair<int, float>> temp;
        for (int j = 0; j < avg_nnz; j++) temp.push_back({rand() % M, 1.0f});
        std::sort(temp.begin(), temp.end());
        for (int j = 0; j < avg_nnz; j++) {
            h_col_indices[i * avg_nnz + j] = temp[j].first;
            h_values[i * avg_nnz + j] = temp[j].second;
        }
    }
    h_row_ptr[M] = NNZ;


    std::vector<int> h_tile_ptr(M * num_tiles, 0);
    for (int i = 0; i < M; i++) {
        int start = h_row_ptr[i];
        int end = h_row_ptr[i+1];
        int curr_t = 0;
        for (int j = start; j < end; j++) {
            int t_id = h_col_indices[j] / TILE_SIZE;
            while (curr_t <= t_id) {
                h_tile_ptr[i * num_tiles + curr_t] = j;
                curr_t++;
            }
        }
        while (curr_t < num_tiles) {
            h_tile_ptr[i * num_tiles + curr_t] = end;
            curr_t++;
        }
    }


    int *d_row, *d_col, *d_tile; float *d_val, *d_x, *d_y;
    cudaMalloc(&d_row, (M+1)*4); cudaMalloc(&d_col, NNZ*4);
    cudaMalloc(&d_tile, M*num_tiles*4);
    cudaMalloc(&d_val, NNZ*4); cudaMalloc(&d_x, M*4); cudaMalloc(&d_y, M*4);

    cudaMemcpy(d_row, h_row_ptr.data(), (M+1)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, h_col_indices.data(), NNZ*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tile, h_tile_ptr.data(), M*num_tiles*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, h_values.data(), NNZ*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), M*4, cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((M + 7) / 8);

    // Warmup & Timing
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for(int i=0; i<20; i++) 
        spmv_v5_ultimate_kernel<<<grid, block>>>(M, M, num_tiles, d_row, d_col, d_val, d_tile, d_x, d_y);
    cudaEventRecord(e); cudaEventSynchronize(e);

    float ms; cudaEventElapsedTime(&ms, s, e);
    float avg_ms = ms / 20.0f;
    std::cout << std::setw(10) << M << "," << std::setw(12) << avg_ms << "," 
              << std::setw(10) << (2.0*NNZ/1e9)/(avg_ms/1000.0) << std::endl;

    cudaFree(d_row); cudaFree(d_col); cudaFree(d_tile); cudaFree(d_val); cudaFree(d_x); cudaFree(d_y);
}

int main() {
    std::cout << "V5-Ultimate (Hierarchical Indexing) Sensitivity Analysis" << std::endl;
    std::cout << "M_Size    , Avg_Time(ms), GFLOPS" << std::endl;
    for (int m = 10000; m <= 100000; m += 10000) run_ultimate_benchmark(m);
    return 0;
}