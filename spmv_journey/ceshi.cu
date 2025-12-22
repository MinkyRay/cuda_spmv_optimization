#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdio>

__device__ void get_intersection(int k, const int* row_ptr, int M, int NNZ, int* out_i, int* out_j) {
    int i_low = max(0, k - NNZ);
    int i_high = min(k, M);
    while (i_low < i_high) {
        int mid = (i_low + i_high) >> 1;
        if (row_ptr[mid] < k - mid) i_low = mid + 1;
        else i_high = mid;
    }
    *out_i = i_low;
    *out_j = k - i_low;
}

__global__ void debug_merge_path_kernel(int M, int NNZ, const int* row_ptr, const int* col_indices, const float* values, const float* x, float* y, int items_per_thread) {
    int tid = threadIdx.x; // 假设只开 1 个 block
    int k_start = tid * items_per_thread;
    int k_end   = (tid + 1) * items_per_thread;

    int i, j;
    get_intersection(k_start, row_ptr, M, NNZ, &i, &j);

    printf("[Thread %d] Start at (%d, %d), Path Range: [%d, %d)\n", tid, i, j, k_start, k_end);

    float sum = 0.0f;
    for (int k = k_start; k < k_end; ++k) {
        if (i < M && j >= row_ptr[i + 1]) {
            printf("  -> Step %d: JUMP Row %d to %d (Partial Sum: %.1f)\n", k, i, i+1, sum);
            if (sum != 0.0f) atomicAdd(&y[i], sum);
            sum = 0.0f;
            i++; 
        } else if (j < NNZ) {
            printf("  -> Step %d: CALC Element %d at Row %d (Val: %.1f)\n", k, j, i, values[j]);
            sum += values[j] * x[col_indices[j]];
            j++;
        }
    }
    if (sum != 0.0f && i < M) {
        printf("  -> Final: FLUSH Row %d (Sum: %.1f)\n", i, sum);
        atomicAdd(&y[i], sum);
    }
}

int main() {
    // 构造一个包含空行的 4x4 矩阵
    // Row 0: [10] (1 nnz)
    // Row 1: [20, 30] (2 nnz)
    // Row 2: [] (Empty Row)
    // Row 3: [40] (1 nnz)
    int M = 4, NNZ = 4;
    int h_row_ptr[] = {0, 1, 3, 3, 4}; 
    int h_col_idx[] = {0, 1, 2, 3};
    float h_val[] = {10.0f, 20.0f, 30.0f, 40.0f};
    float h_x[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float h_y[4] = {0};

    int *d_row, *d_col; float *d_val, *d_x, *d_y;
    cudaMalloc(&d_row, 5*4); cudaMalloc(&d_col, 4*4);
    cudaMalloc(&d_val, 4*4); cudaMalloc(&d_x, 4*4); cudaMalloc(&d_y, 4*4);
    cudaMemcpy(d_row, h_row_ptr, 5*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, h_col_idx, 4*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, h_val, 4*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, 4*4, cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, 4*4);

    // 总路径长度 L = 4 + 4 = 8。我们开 4 个线程，每个线程走 2 步。
    debug_merge_path_kernel<<<1, 4>>>(M, NNZ, d_row, d_col, d_val, d_x, d_y, 2);
    cudaDeviceSynchronize();

    cudaMemcpy(h_y, d_y, 4*4, cudaMemcpyDeviceToHost);
    printf("\nResult: [%.1f, %.1f, %.1f, %.1f]\n", h_y[0], h_y[1], h_y[2], h_y[3]);

    return 0;
}