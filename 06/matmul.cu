#include <cuda_runtime.h>
#include <curand.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "common.cuh"

void check_curand(curandStatus_t status, const char* operation)
{
    if (status != CURAND_STATUS_SUCCESS) {
        std::cerr << "cuRAND error in " << operation
                  << ": status " << status << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

__global__ void matmul_naive(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || column >= N) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + column];
    }
    C[row * N + column] = sum;
}

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

bool check_result(
    const float* d_A,
    const float* d_B,
    const float* d_C,
    int M,
    int N,
    int K)
{
    constexpr float absolute_tolerance = 1.0e-3f;
    constexpr float relative_tolerance = 1.0e-4f;

    const std::size_t a_elements = static_cast<std::size_t>(M) * K;
    const std::size_t b_elements = static_cast<std::size_t>(K) * N;
    const std::size_t c_elements = static_cast<std::size_t>(M) * N;

    std::vector<float> h_A(a_elements);
    std::vector<float> h_B(b_elements);
    std::vector<float> h_C(c_elements);
    CUDA_CHECK(cudaMemcpy(
        h_A.data(), d_A, a_elements * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        h_B.data(), d_B, b_elements * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        h_C.data(), d_C, c_elements * sizeof(float), cudaMemcpyDeviceToHost));

    std::size_t error_count = 0;
    float maximum_absolute_error = 0.0f;
    float maximum_relative_error = 0.0f;

    for (int row = 0; row < M; ++row) {
        for (int column = 0; column < N; ++column) {
            float reference = 0.0f;
            for (int k = 0; k < K; ++k) {
                reference += h_A[row * K + k] * h_B[k * N + column];
            }

            const float actual = h_C[row * N + column];
            const float absolute_error = std::abs(actual - reference);
            const float relative_error =
                absolute_error / std::max(std::abs(reference), 1.0e-6f);
            maximum_absolute_error =
                std::max(maximum_absolute_error, absolute_error);
            maximum_relative_error =
                std::max(maximum_relative_error, relative_error);

            const float tolerance =
                absolute_tolerance + relative_tolerance * std::abs(reference);
            if (absolute_error > tolerance) {
                if (error_count < 5) {
                    std::cerr << "Mismatch at (" << row << ", " << column
                              << "): expected " << reference
                              << ", got " << actual << std::endl;
                }
                ++error_count;
            }
        }
    }

    std::cout << "Maximum absolute error: " << maximum_absolute_error << std::endl;
    std::cout << "Maximum relative error: " << maximum_relative_error << std::endl;
    std::cout << "Success: " << (error_count == 0) << std::endl;
    return error_count == 0;
}

int main()
{
    constexpr int M = 512;
    constexpr int N = 512;
    constexpr int K = 512;
    constexpr int BLOCK_SIZE = 16;
    constexpr int WARMUP_ITERATIONS = 10;
    constexpr int MEASURED_ITERATIONS = 100;
    constexpr int PROFILED_ITERATION = 22;
    constexpr int FLUSH_BLOCK_SIZE = 256;
    constexpr unsigned long long RANDOM_SEED = 1;

    const std::size_t a_elements = static_cast<std::size_t>(M) * K;
    const std::size_t b_elements = static_cast<std::size_t>(K) * N;
    const std::size_t c_elements = static_cast<std::size_t>(M) * N;
    const std::size_t a_bytes = a_elements * sizeof(float);
    const std::size_t b_bytes = b_elements * sizeof(float);
    const std::size_t c_bytes = c_elements * sizeof(float);

    int device = 0;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    const std::size_t l2_bytes = properties.l2CacheSize;
    const std::size_t flush_bytes = 2 * l2_bytes;
    const std::size_t flush_elements =
        (flush_bytes + sizeof(std::uint32_t) - 1) / sizeof(std::uint32_t);
    const int flush_blocks = static_cast<int>(
        (flush_elements + FLUSH_BLOCK_SIZE - 1) / FLUSH_BLOCK_SIZE);

    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    std::uint32_t* d_l2_flush = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, c_bytes));
    CUDA_CHECK(cudaMalloc(&d_l2_flush, flush_elements * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMemset(d_l2_flush, 0, flush_elements * sizeof(std::uint32_t)));

    curandGenerator_t random_generator;
    check_curand(
        curandCreateGenerator(&random_generator, CURAND_RNG_PSEUDO_DEFAULT),
        "curandCreateGenerator");
    check_curand(
        curandSetPseudoRandomGeneratorSeed(random_generator, RANDOM_SEED),
        "curandSetPseudoRandomGeneratorSeed");

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::cout << "Matrix multiplication: " << M << " x " << K
              << " times " << K << " x " << N << std::endl;
    std::cout << "Block: " << BLOCK_SIZE << " x " << BLOCK_SIZE << std::endl;
    std::cout << "L2 cache: " << l2_bytes / 1024.0 / 1024.0 << " MiB" << std::endl;
    std::cout << "L2 flush buffer: " << flush_bytes / 1024.0 / 1024.0
              << " MiB" << std::endl;
    std::cout << "Warmup iterations: " << WARMUP_ITERATIONS << std::endl;
    std::cout << "Measured iterations: " << MEASURED_ITERATIONS << std::endl;

    // Each iteration receives fresh random input. The L2 flush is deliberately
    // launched after input generation and before the matrix multiplication.
    for (int iteration = 0; iteration < WARMUP_ITERATIONS; ++iteration) {
        check_curand(
            curandGenerateUniform(random_generator, d_A, a_elements),
            "curandGenerateUniform(A)");
        check_curand(
            curandGenerateUniform(random_generator, d_B, b_elements),
            "curandGenerateUniform(B)");
        flush_l2_cache<<<flush_blocks, FLUSH_BLOCK_SIZE>>>(
            d_l2_flush, flush_elements, iteration + 1);
        matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> elapsed_times;
    elapsed_times.reserve(MEASURED_ITERATIONS);

    for (int iteration = 0; iteration < MEASURED_ITERATIONS; ++iteration) {
        check_curand(
            curandGenerateUniform(random_generator, d_A, a_elements),
            "curandGenerateUniform(A)");
        check_curand(
            curandGenerateUniform(random_generator, d_B, b_elements),
            "curandGenerateUniform(B)");
        flush_l2_cache<<<flush_blocks, FLUSH_BLOCK_SIZE>>>(
            d_l2_flush, flush_elements, iteration + 1);

        if (iteration == PROFILED_ITERATION) {
            nvtxRangePushA("matmul_naive iteration 23");
        }
        CUDA_CHECK(cudaEventRecord(start));
        matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaEventRecord(stop));
        if (iteration == PROFILED_ITERATION) {
            nvtxRangePop();
        }

        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        elapsed_times.push_back(elapsed_ms);
    }
    CUDA_CHECK(cudaGetLastError());

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
        2.0 * static_cast<double>(M) * N * K;
    const double gflops = operation_count / (average_ms * 1.0e6);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\nmatmul_naive" << std::endl;
    std::cout << "Average time: " << average_ms << " ms" << std::endl;
    std::cout << "Minimum time: " << minimum_ms << " ms" << std::endl;
    std::cout << "Maximum time: " << maximum_ms << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOP/s" << std::endl;

    const bool success = check_result(d_A, d_B, d_C, M, N, K);

    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    check_curand(curandDestroyGenerator(random_generator), "curandDestroyGenerator");
    CUDA_CHECK(cudaFree(d_l2_flush));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_A));

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
