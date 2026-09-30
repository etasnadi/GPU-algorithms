#include <iostream>

__global__ void block_reduce(float* numbers, float* out, int n){
    __shared__ float partial[128];
    partial[threadIdx.x] = numbers[threadIdx.x];

    __syncthreads();
    //printf("%d %.2f\n", threadIdx.x, partial[threadIdx.x]);

    // --- reduction tree ---

    if(threadIdx.x < 64){
        partial[threadIdx.x] += partial[threadIdx.x + 64];
    }
    __syncthreads();

    if(threadIdx.x < 32){
        partial[threadIdx.x] += partial[threadIdx.x + 32];
    }
    __syncthreads();

    if(threadIdx.x < 16){
        partial[threadIdx.x] += partial[threadIdx.x + 16];
    }
    __syncthreads();

    if(threadIdx.x < 8){
        partial[threadIdx.x] += partial[threadIdx.x + 8];
    }
    __syncthreads();

    if(threadIdx.x < 4){
        partial[threadIdx.x] += partial[threadIdx.x + 4];
    }
    __syncthreads();

    if(threadIdx.x < 2){
        partial[threadIdx.x] += partial[threadIdx.x + 2];
    }
    __syncthreads();

    if(threadIdx.x < 1){
        partial[threadIdx.x] += partial[threadIdx.x + 1];
    }
    __syncthreads();

    // ----
    if(threadIdx.x == 0){
        out[0] = partial[threadIdx.x]; 
    }
}

int main(){
    int n = 128;
    float* h_numbers;
    float* h_out;

    float* d_numbers;
    float* d_out;

    // Allocate on host
    h_numbers = new float[n];
    h_out = new float[1];

    // Initialize
    for(int i = 0; i < n; i++){
        h_numbers[i] = i;
    }

    // Compute exepcted result
    float reference = 0.0;
    for(int i = 0; i < n; i++){
        reference += h_numbers[i];
    }

    // Allocate on GPU
    cudaMalloc(&d_numbers, n*sizeof(float));
    cudaMalloc(&d_out, 1*sizeof(float));

    // Copy host buffer to device buffer
    cudaMemcpy(d_numbers, h_numbers, n*sizeof(float), cudaMemcpyHostToDevice);

    block_reduce<<<1, 128>>>(d_numbers, d_out, n);
    cudaDeviceSynchronize();

    // Copy the result from the GPU memory to the host
    cudaMemcpy(h_out, d_out, 1*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Reference result:" << reference << std::endl;
    std::cout << "Computed result:" << h_out[0] << std::endl;
}
