#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <iomanip>
#include <iostream>
#include "common.cuh"

__global__ void kernel(float* input, float* output, int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n){
		output[i] = input[i];
	}
}

int main(){
	int n = 1 << 24;
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
	int block_dim = 256;
	int grid_dim = (n + block_dim - 1) / block_dim;
	
	std::cout << "Block dim: " << block_dim << std::endl;
	std::cout << "Grid dim: " << grid_dim << std::endl;	
	
	// Warm up the GPU before measuring.
	kernel<<<grid_dim, block_dim>>>(d_input, d_output, n);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	constexpr int iterations = 100;
	cudaEvent_t start;
	cudaEvent_t stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	CUDA_CHECK(cudaEventRecord(start));
	for(int iteration = 0; iteration < iterations; iteration++){
		if(iteration == 22){
			nvtxRangePushA("copy-kernel iteration 23");
		}
		kernel<<<grid_dim, block_dim>>>(d_input, d_output, n);
		if(iteration == 22){
			nvtxRangePop();
		}
	}
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));

	float elapsed_ms = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
	const double average_ms = elapsed_ms / iterations;
	// Every element is read once and written once by the copy kernel.
	const double bytes_per_iteration = 2.0 * n * sizeof(float);
	const double bandwidth_gb_s = (bytes_per_iteration / (1'000'000'000.0)) / (average_ms / 1000.0);

	std::cout << std::fixed << std::setprecision(3);
	std::cout << "Buffer size: " 			<< bytes_per_iteration / 1'000'000 << " MB" << std::endl;
	std::cout << "Average kernel time: " 	<< average_ms << " ms" << std::endl;
	std::cout << "Effective bandwidth: " 	<< bandwidth_gb_s << " GB/s" << std::endl;
	
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

	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));
	CUDA_CHECK(cudaFree(d_input));
	CUDA_CHECK(cudaFree(d_output));
	delete[] h_input;
	delete[] h_output;
}
