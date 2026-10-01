#include <iostream>
#include "common.cuh"

__global__ void trans(float* A, float* B, int M, int N){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    /*
    Linear thread index is computed as: 
    threadId = threadIdx.y*blockDim.x + threadIdx.x
    
    Assuming that warp size is 16 by 16, the warp/lane - (threadIdx.x, threadIdx.y) 
    assignment is the following:
    
    warp 0: 
        lane 0 (threadIdx.y = 0, threadIdx.x=0)
        lane 1 (threadIdx.y = 0, threadIdx.x=1)
        lane 2 (threadIdx.y = 0, threadIdx.x=2)
        lane 3 (threadIdx.y = 0, threadIdx.x=3)
        ...
        lane 15 (threadIdx.y = 0, threadIdx.x=15)
        lane 16 (threadIdx.y = 1, threadIdx.x=0)
        lane 17 (threadIdx.y = 1, threadIdx.x=1)
        lane 18 (threadIdx.y = 1, threadIdx.x=2)
        ...
        lane 31 (threadIdx.y = 2, threadIdx.x=15)
    warp 1:
        lane 0 (threadIdx.y = 3, threadIdx.x=0)
        lane 1 (threadIdx.y = 3, threadIdx.x=1)
        ...
    */

    // This implies that the transposed copy does strided access.

    // Transpose A into B
    B[i*N+j] = A[j*M+i];

    // Tiled copy of A into B, this is not strided, as j depends on j that is contigous.
    //B[i*N+j] = A[i*N+j];
}

int main(){
    float* h_A;
    float* h_B;
    
    float* d_A;
    float* d_B;
    
    int M = 32;
    int N = 32;
    int TILE_SIZE = 16;
    h_A = new float[M*N];
    h_B = new float[M*N];

	cudaEvent_t start;
	cudaEvent_t stop;    
	CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaMalloc(&d_A, M*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, M*N*sizeof(float)));

    // Fill host buffer
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            h_A[i*N+j] = i*N+j;            
        }
    }

    CUDA_CHECK(cudaMemcpy(d_A, h_A, M*N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(start));
    trans<<<dim3(M/TILE_SIZE, N/TILE_SIZE), dim3(TILE_SIZE, TILE_SIZE)>>>(d_A, d_B, M, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_B, d_B, M*N*sizeof(float), cudaMemcpyDeviceToHost));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    std::cout << "Elapsed time: " << elapsed_ms << std::endl;

    // Print matrix
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            std::cout << h_B[i*N+j] << " ";
        }
        std::cout << std::endl;
    }
    bool equals = true;
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            if(h_B[i*N+j] != h_A[j*N+i]){
                equals = false;
            }
        }
    }
    std::cout << "Success: " << equals << std::endl;
    delete[] h_B;
    delete[] h_A;
    cudaFree(d_B);
    cudaFree(d_A);
}