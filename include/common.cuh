#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

// https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#error-checking-in-cuda
#define CUDA_CHECK(expr_to_check) do {            \
    cudaError_t result  = expr_to_check;          \
    if(result != cudaSuccess)                     \
    {                                             \
        fprintf(stderr,                           \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__,                         \
                __LINE__,                         \
                result,\
                cudaGetErrorString(result));      \
    }                                             \
} while(0)

__global__ void flush_l2_cache(
    std::uint32_t* buffer,
    std::size_t element_count,
    std::uint32_t value)
{
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < element_count;
         i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        buffer[i] += value;
    }
}

template<int BLOCK_SIZE>
void launch_flush_l2_cache(
    std::uint32_t* buffer,
    std::size_t element_count,
    std::uint32_t value)
{
    const int block_count = static_cast<int>(
        (element_count + BLOCK_SIZE - 1) / BLOCK_SIZE);

    flush_l2_cache<<<block_count, BLOCK_SIZE>>>(
        buffer, element_count, value);
}

inline void print_performance(
    const char* name,
    const std::vector<float>& elapsed_times,
    int MATRIX_M,
    int MATRIX_N,
    int MATRIX_K)
{
    double total_ms = 0.0;
    float minimum_ms = std::numeric_limits<float>::max();
    float maximum_ms = 0.0f;

    for (float elapsed_ms : elapsed_times) {
        total_ms += elapsed_ms;
        minimum_ms = std::min(minimum_ms, elapsed_ms);
        maximum_ms = std::max(maximum_ms, elapsed_ms);
    }

    const double average_ms = total_ms / elapsed_times.size();
    const double operation_count =
        2.0 * static_cast<double>(MATRIX_M) * MATRIX_N * MATRIX_K;
    const double gflops = operation_count / (average_ms * 1.0e6);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n" << name << std::endl;
    std::cout << "Average time: " << average_ms << " ms" << std::endl;
    std::cout << "Minimum time: " << minimum_ms << " ms" << std::endl;
    std::cout << "Maximum time: " << maximum_ms << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOP/s" << std::endl;
}
