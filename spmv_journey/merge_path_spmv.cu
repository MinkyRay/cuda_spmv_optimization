#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

__device__ __forceinline__ void get_intersection(
    int k, const int* row_ptr, int M, int NNZ, int* out_i, int* out_j) {
    
    // Find row i such that: i + row_ptr[i] <= k < (i+1) + row_ptr[i+1]
    int i_low = 0;
    int i_high = M;
    
    while (i_low < i_high) {
        int mid = (i_low + i_high) >> 1;
        if (mid + row_ptr[mid] <= k) {
            i_low = mid + 1;
        } else {
            i_high = mid;
        }
    }
    
    // After binary search, i_low points to the first row where mid + row_ptr[mid] > k
    // So we want i = i_low - 1
    int i = (i_low > 0) ? i_low - 1 : 0;
    
    // Adjust if needed
    while (i < M && i + row_ptr[i + 1] <= k) {
        i++;
    }
    
    *out_i = i;
    *out_j = k - i;
}

__global__ void spmv_merge_path_kernel(
    int M, int NNZ, const int* row_ptr, const int* col_indices, 
    const float* values, const float* x, float* y) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    int total_path = M + NNZ;
    int items_per_thread = (total_path + total_threads - 1) / total_threads;

    int k_start = min(tid * items_per_thread, total_path);
    int k_end   = min((tid + 1) * items_per_thread, total_path);

    if (k_start >= k_end) return;

    int i, j;
    get_intersection(k_start, row_ptr, M, NNZ, &i, &j);

    float partial_sum = 0.0f;
    int current_row = i;
    
    // Walk the merge path from k_start to k_end
    while (k_start < k_end) {
        // Check if we've finished current row
        if (i < M && j >= row_ptr[i + 1]) {
            // Write accumulated result
            if (partial_sum != 0.0f) {
                atomicAdd(&y[current_row], partial_sum);
                partial_sum = 0.0f;
            }
            // Move to next row
            i++;
            current_row = i;
            k_start++;  // Increment path counter for row transition
        } 
        // Process element from current row
        else if (i < M && j < row_ptr[i + 1] && j < NNZ) {
            partial_sum += values[j] * x[col_indices[j]];
            j++;
            k_start++;  // Increment path counter for element
        }
        else {
            break;  // Safety exit
        }
    }

    // Flush remaining partial sum
    if (partial_sum != 0.0f && current_row < M) {
        atomicAdd(&y[current_row], partial_sum);
    }
}

// --- CPU 版本验证 ---
void spmv_cpu(int M, const int* row_ptr, const int* col_indices, const float* values, const float* x, float* y) {
    for (int i = 0; i < M; ++i) {
        float sum = 0.0f;
        for (int j = row_ptr[i]; j < row_ptr[i+1]; ++j) {
            sum += values[j] * x[col_indices[j]];
        }
        y[i] = sum;
    }
}

int main() {
    const int M = 20000;
    const int avg_nnz = 128;
    const int NNZ = M * avg_nnz;

    std::vector<int> h_row_ptr(M + 1);
    std::vector<int> h_col_indices(NNZ);
    std::vector<float> h_values(NNZ);
    std::vector<float> h_x(M, 1.0f);
    std::vector<float> h_y_cpu(M, 0.0f);
    std::vector<float> h_y_gpu(M, 0.0f);

    // Initialize with fixed seed for reproducibility
    srand(42);
    for (int i = 0; i < M; ++i) {
        h_row_ptr[i] = i * avg_nnz;
        for (int j = 0; j < avg_nnz; ++j) {
            h_col_indices[i * avg_nnz + j] = rand() % M;
            h_values[i * avg_nnz + j] = 1.0f;
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
    cudaMemset(d_y, 0, M * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 block(256);
    dim3 grid(128);

    // Warmup
    spmv_merge_path_kernel<<<grid, block>>>(M, NNZ, d_row_ptr, d_col_indices, d_values, d_x, d_y);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for(int i = 0; i < 100; i++) {
        cudaMemset(d_y, 0, M * sizeof(float));
        spmv_merge_path_kernel<<<grid, block>>>(M, NNZ, d_row_ptr, d_col_indices, d_values, d_x, d_y);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_ms = milliseconds / 100.0f;

    cudaMemcpy(h_y_gpu.data(), d_y, M * sizeof(float), cudaMemcpyDeviceToHost);
    spmv_cpu(M, h_row_ptr.data(), h_col_indices.data(), h_values.data(), h_x.data(), h_y_cpu.data());

    // Better validation
    bool correct = true;
    int error_count = 0;
    for (int i = 0; i < M; ++i) {
        float diff = fabs(h_y_gpu[i] - h_y_cpu[i]);
        float rel_error = (h_y_cpu[i] != 0.0f) ? diff / fabs(h_y_cpu[i]) : diff;
        if (rel_error > 1e-4) {
            if (error_count < 5) {  // Print first 5 errors
                std::cout << "Mismatch at i=" << i << ": GPU=" << h_y_gpu[i] 
                          << " CPU=" << h_y_cpu[i] << " diff=" << diff << std::endl;
            }
            error_count++;
            correct = false;
        }
    }
    if (error_count > 0) {
        std::cout << "Total errors: " << error_count << " / " << M << std::endl;
    }

    // Performance analysis
    long long flops = 2LL * NNZ;
    long long bytes = NNZ * sizeof(float) +      // values
                      NNZ * sizeof(int) +         // col_indices  
                      NNZ * sizeof(float) +       // x (random reads)
                      (M + 1) * sizeof(int) +     // row_ptr
                      M * sizeof(float);          // y (writes)
    
    float gflops = (flops / 1e9) / (avg_ms / 1000.0);
    float bandwidth_gbs = (bytes / 1e9) / (avg_ms / 1000.0);
    float arithmetic_intensity = (float)flops / bytes;

    std::cout << "\n========== Merge-Path SpMV Performance ==========\n";
    std::cout << "Matrix Size: " << M << " x " << M << " (NNZ: " << NNZ << ")\n";
    std::cout << "Status:      " << (correct ? "PASS" : "FAIL") << "\n";
    std::cout << "Avg Time:    " << avg_ms << " ms\n";
    std::cout << "\n--- Performance Metrics ---\n";
    std::cout << "GFLOPS:      " << gflops << "\n";
    std::cout << "Bandwidth:   " << bandwidth_gbs << " GB/s\n";
    std::cout << "Data Volume: " << bytes / 1e6 << " MB\n";
    std::cout << "Arithmetic Intensity: " << arithmetic_intensity << " FLOPs/Byte\n";
    std::cout << "\n--- Analysis ---\n";
    std::cout << "Peak Bandwidth (RTX 4060): ~272 GB/s\n";
    std::cout << "Bandwidth Utilization: " << (bandwidth_gbs / 272.0) * 100 << "%\n";
    
    if (arithmetic_intensity < 0.5) {
        std::cout << "\nSTRONGLY MEMORY BOUND\n";
        std::cout << "Optimization suggestions:\n";
        std::cout << "  - Too many atomicAdd conflicts (main issue of merge-path)\n";
        std::cout << "  - Consider warp-level reduce to reduce atomics\n";
        std::cout << "  - Or switch to CSR-Vector kernel (better for regular matrices)\n";
    }
    std::cout << "=================================================\n";

    cudaFree(d_row_ptr); 
    cudaFree(d_col_indices); 
    cudaFree(d_values); 
    cudaFree(d_x); 
    cudaFree(d_y);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}