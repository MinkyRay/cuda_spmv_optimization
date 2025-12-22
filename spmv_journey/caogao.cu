




__global__ void spmv_csr_scalar_kernel(
    int num_rows, 
    int *row_ptr, 
    int *col_indices, 
    float *values, 
    float *x, 
    float *y
){
    //index
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < num_rows){
        int start = row_ptr[row];
        int end = row_ptr[row+1];
        int sum = 0.0f;
        for (int i = start; i < end; ++i){
            sum += values[i] * x[col_indices[i]];
        }
        y[row] = sum;
    }
}

__global__ void spmv_csr_scalar_kernel(
    int num_rows, 
    int *row_ptr, 
    int *col_indices, 
    float *values, 
    float *x, 
    float *y
){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row = tid / 32;
    int lane_id = tid % 32;
    if (row < num_rows){
        int start = row_ptr[row];
        int end = row_ptr[row+1];
        int sum = 0.0f;
        for (int i = start + lane_id; i < end; i+=32){
            sum += values[i] * x[col_indices[i]];
        }
        for (int offset = 16; offset > 0; offset /= 2){
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane_id == 0){
            y[row] = sum;
        }
    }
}

__global__ void spmv_csr_scalar_kernel(
    int num_rows, 
    int *row_ptr, 
    int *col_indices, 
    float *values, 
    float *x, 
    float *y
){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row = tid / SWS;
    int lane_id = tid % SWS;
    if (row < num_rows){
        float sum = 0.0f;
        int start = row_ptr[row];
        int end = row_ptr[row+1];
        for (int i = start + lane_id; i < end; i += 8){
            sum += values[i] * x[col_indices[i]];
        }
        for ()
    }
}




























//对于K
left = 0; right = len(values);
while (left <= right){
    mid = (left + right) / 2;
    if (row_ptr[mid] > K - mid){
        right = mid - 1;
    }
    elif (row_ptr[mid] < K - mid){
        left = mid + 1;
    }
    else{
        point.append([i,K - mid]);
    }
}