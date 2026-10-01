#include <cuda_runtime.h>
#include <iostream>
#include "common.cuh"

__global__ void kernel(float* input, float* output, int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n){
		output[i] = input[i];
	}
}

int main(){
	int n = 13;
	// Host buffers
	float* h_input = new float[n];
	float* h_output = new float[n];

	// CUDA buferrs
	float* d_input;
	float* d_output;
	CUDA_CHECK(cudaMalloc(&d_input, n*sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_output, n*sizeof(float)));
	
	// Fill with data
	for(int i = 0; i < n; i++){
		h_input[i] = i*2;
		h_output[i] = 0;
	}
	
	// Copy host to device
	CUDA_CHECK(cudaMemcpy(d_input, h_input, n*sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_output, h_output, n*sizeof(float), cudaMemcpyHostToDevice));	
	
	// Kernel launch
	int block_dim = 3;
	int grid_dim = (n + block_dim - 1) / block_dim;
	
	std::cout << "Block dim: " << block_dim << std::endl;
	std::cout << "Grid dim: " << grid_dim << std::endl;	
	
	kernel<<<grid_dim, block_dim>>>(d_input, d_output, n);
	CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
	
	// Copy device to host
	CUDA_CHECK(cudaMemcpy(h_output, d_output, n*sizeof(float), cudaMemcpyDeviceToHost));

	bool success = true;

	// Validation
	for(int i = 0; i < n; i++){
		if(h_output[i] != h_input[i]){
			success = false;
			break;
		}
	}
	std::cout << "Success: " << success << std::endl;
}
