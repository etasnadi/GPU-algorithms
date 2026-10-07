#include <cassert>
#include <iostream>
#include "common.cuh"

// Matrix transpose

/*

A: MxN
B: NxM

A:
0,  1,  2,  3
4,  5,  6,  7
8,  9,  10, 11
12, 13, 14, 15

B = A.T
0,  4,  8,  12
1,  5,  9,  13
2,  6,  10, 14
3,  7,  11, 15

Exercise: compute B = A.T efficiently.

*/

#define ADD_ASSERTS
//#define PRINT_MATRIX

// Select one: naive, shmem, shmem_fix_bank_conflicts
#ifndef TRANSPOSE_KERNEL
#define TRANSPOSE_KERNEL shmem
#endif

// Naive kernel
template<int TILE_SIZE>
__global__ void naive(float* A, float* B, int M, int N){
    int tile_Ax = blockIdx.x;
    int tile_Ay = blockIdx.y;
    int pos_Ax = threadIdx.x;
    int pos_Ay = threadIdx.y;

    #ifdef ADD_ASSERTS
    assert(TILE_SIZE == blockDim.x);
    assert(TILE_SIZE == blockDim.y);
    #endif

    B[(tile_Ax*TILE_SIZE+pos_Ax)*N + tile_Ay*TILE_SIZE+pos_Ay] = A[(tile_Ay*TILE_SIZE+pos_Ay)*N + tile_Ax*TILE_SIZE+pos_Ax];

    // =====> Problem?
    //  pos_Ay = threadIdx.y and 
    //  pos_Ax = threadIdx.x
    // Linear thread index: threadIdx.y*blockDim.x+threadIdx.x
    // Which array is access is not coalesced but strided?
}

/*

More efficient kernel: transpose in shmem.

Example:
tile_size = 2x2
tile B(1, 0) = tile A(0, 1).T:

tmp = tile A(0, 1):
2,  3
6,  7

tile B(1, 0) = tmp.T:
2,  6
3,  7

*/
template<int TILE_SIZE>
__global__ void shmem(float* A, float* B, int M, int N){
    int tile_Ax = blockIdx.x;
    int tile_Ay = blockIdx.y;
    int pos_Ax = threadIdx.x;
    int pos_Ay = threadIdx.y;

    #ifdef ADD_ASSERTS
    assert(TILE_SIZE == blockDim.x);
    assert(TILE_SIZE == blockDim.y);
    #endif

    __shared__ float tmp[TILE_SIZE][TILE_SIZE];

    // tmp = tile A(Ay, Ax)
    tmp[pos_Ay][pos_Ax] = A[(tile_Ay*TILE_SIZE+pos_Ay)*N + tile_Ax*TILE_SIZE+pos_Ax];

    __syncthreads();

    // Now we need to write tile B(Ax, Ay) = tmp.T
    
    // Variant #0 - simple copy: B(Ay, Ax) = tmp
    // B[(tile_Ay*TILE_SIZE+pos_Ay)*N + tile_Ax*TILE_SIZE+pos_Ax] = tmp[pos_Ay][pos_Ax];
    
    // Variant #1 - only transpose tiles: B(Ax, Ay) = tmp
    // B[(tile_Ax*TILE_SIZE+pos_Ay)*N + tile_Ay*TILE_SIZE+pos_Ax] = tmp[pos_Ay][pos_Ax];

    // Variant #2 - complete shmem transpose: B(Ax, Ay) = tmp.T
    // =====> Problem?
    // B[(tile_Ax*TILE_SIZE+pos_Ax)*N + tile_Ay*TILE_SIZE+pos_Ay] = tmp[pos_Ay][pos_Ax];

    // Variant #3 - efficient solution
    B[(tile_Ax*TILE_SIZE+pos_Ay)*N + tile_Ay*TILE_SIZE+pos_Ax] = tmp[pos_Ax][pos_Ay];

    // =====> Problem?
}

template<int TILE_SIZE>
__global__ void shmem_fix_bank_conflicts(float* A, float* B, int M, int N){
    int tile_Ax = blockIdx.x;
    int tile_Ay = blockIdx.y;
    int pos_Ax = threadIdx.x;
    int pos_Ay = threadIdx.y;

    #ifdef ADD_ASSERTS
    assert(TILE_SIZE == blockDim.x);
    assert(TILE_SIZE == blockDim.y);
    #endif

    __shared__ float tmp[TILE_SIZE][TILE_SIZE+1];

    // tmp = tile A(Ay, Ax)
    tmp[pos_Ay][pos_Ax] = A[(tile_Ay*TILE_SIZE+pos_Ay)*N + tile_Ax*TILE_SIZE+pos_Ax];

    __syncthreads();

    // Now we need to write tile B(Ax, Ay) = tmp.T
    
    // Variant #0 - simple copy: B(Ay, Ax) = tmp
    // B[(tile_Ay*TILE_SIZE+pos_Ay)*N + tile_Ax*TILE_SIZE+pos_Ax] = tmp[pos_Ay][pos_Ax];
    
    // Variant #1 - only transpose tiles: B(Ax, Ay) = tmp
    // B[(tile_Ax*TILE_SIZE+pos_Ay)*N + tile_Ay*TILE_SIZE+pos_Ax] = tmp[pos_Ay][pos_Ax];

    // Variant #2 - complete shmem transpose: B(Ax, Ay) = tmp.T
    // Problem?
    // B[(tile_Ax*TILE_SIZE+pos_Ax)*N + tile_Ay*TILE_SIZE+pos_Ay] = tmp[pos_Ay][pos_Ax];

    // Variant #3 - efficient solution
    B[(tile_Ax*TILE_SIZE+pos_Ay)*N + tile_Ay*TILE_SIZE+pos_Ax] = tmp[pos_Ax][pos_Ay];
}

int main(){
    float* h_A;
    float* h_B;
    
    float* d_A;
    float* d_B;
    
    int M = 1024;
    int N = 1024;
    const int TILE_SIZE = 16;
    #ifdef ADD_ASSERTS
        assert(M % TILE_SIZE == 0);
        assert(N % TILE_SIZE == 0);
    #endif
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
    TRANSPOSE_KERNEL<TILE_SIZE><<<dim3(M/TILE_SIZE, N/TILE_SIZE), dim3(TILE_SIZE, TILE_SIZE)>>>(d_A, d_B, M, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_B, d_B, M*N*sizeof(float), cudaMemcpyDeviceToHost));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    std::cout << "Elapsed time: " << elapsed_ms << std::endl;

    // Print matrix
    #ifdef PRINT_MATRIX
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            std::cout << h_B[i*N+j] << " ";
        }
        std::cout << std::endl;
    }
    #endif
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
