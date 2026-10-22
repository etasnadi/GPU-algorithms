#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
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

void check_cublas(cublasStatus_t status, const char* operation)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS error in " << operation
                  << ": status " << status << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

enum class CheckMode {
    Cublas,
    Cpu
};

CheckMode parse_check_mode(int argc, char** argv)
{
    if (argc == 1) {
        return CheckMode::Cublas;
    }
    if (argc == 3 && std::string(argv[1]) == "--check") {
        const std::string mode = argv[2];
        if (mode == "cublas") {
            return CheckMode::Cublas;
        }
        if (mode == "cpu") {
            return CheckMode::Cpu;
        }
    }

    std::cerr << "Usage: " << argv[0] << " [--check cublas|cpu]" << std::endl;
    std::exit(EXIT_FAILURE);
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

// TILE_SIZE is equal to blockDim.x and blockDim.y
// Why this kernel needs this at compile time?
template<int TILE_SIZE>
__global__ void matmul_shmem(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

    // Indexing C: (colum, row)
    const int column = blockIdx.x * TILE_SIZE + threadIdx.x;
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Indexing A: (c_row, a_column) -> a_column depends on the inner product dim
    // Indexing B: (b_row, c_column) -> b_row depends on the inner product dim

    float sum = 0.0f;
    const int tile_count = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Main loop: inner product dimension
    for (int tile = 0; tile < tile_count; ++tile) {
        const int a_column = tile * TILE_SIZE + threadIdx.x;
        const int b_row = tile * TILE_SIZE + threadIdx.y;

        if(row < M && a_column < K){
            A_tile[threadIdx.y][threadIdx.x] = A[row * K + a_column];
        }else{
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if(b_row < K && column < N){
            B_tile[threadIdx.y][threadIdx.x] = B[b_row * N + column];
        }else{
            B_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && column < N) {
        C[row * N + column] = sum;
    }
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

bool compare_results(
    const std::vector<float>& reference,
    const std::vector<float>& actual,
    int M,
    int N)
{
    constexpr float absolute_tolerance = 1.0e-3f;
    constexpr float relative_tolerance = 1.0e-4f;

    std::size_t error_count = 0;
    float maximum_absolute_error = 0.0f;
    float maximum_relative_error = 0.0f;

    for (int row = 0; row < M; ++row) {
        for (int column = 0; column < N; ++column) {
            const std::size_t index =
                static_cast<std::size_t>(row) * N + column;
            const float expected = reference[index];
            const float received = actual[index];
            const float absolute_error = std::abs(received - expected);
            const float relative_error =
                absolute_error / std::max(std::abs(expected), 1.0e-6f);
            maximum_absolute_error =
                std::max(maximum_absolute_error, absolute_error);
            maximum_relative_error =
                std::max(maximum_relative_error, relative_error);

            const float tolerance =
                absolute_tolerance + relative_tolerance * std::abs(expected);
            if (absolute_error > tolerance) {
                if (error_count < 5) {
                    std::cerr << "Mismatch at (" << row << ", " << column
                              << "): expected " << expected
                              << ", got " << received << std::endl;
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

bool check_result_cpu(
    const float* d_A,
    const float* d_B,
    const float* d_C,
    int M,
    int N,
    int K)
{
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

    std::vector<float> reference(c_elements);

    for (int row = 0; row < M; ++row) {
        for (int column = 0; column < N; ++column) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += h_A[row * K + k] * h_B[k * N + column];
            }
            reference[static_cast<std::size_t>(row) * N + column] = sum;
        }
    }

    return compare_results(reference, h_C, M, N);
}

bool check_result_cublas(
    cublasHandle_t handle,
    const float* d_A,
    const float* d_B,
    const float* d_C,
    float* d_reference,
    int M,
    int N,
    int K)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS uses column-major matrices. Swapping A and B computes
    // C^T = B^T * A^T while preserving the row-major memory layout.

    // Explanation of how we can enforce a specific computation
    // https://leimao.github.io/blog/cuBLAS-Transpose-Column-Major-Relationship/
    check_cublas(
        cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N,
            M,
            K,
            &alpha,
            d_B,
            N,
            d_A,
            K,
            &beta,
            d_reference,
            N),
        "cublasSgemm");

    const std::size_t c_elements = static_cast<std::size_t>(M) * N;
    const std::size_t c_bytes = c_elements * sizeof(float);
    std::vector<float> reference(c_elements);
    std::vector<float> actual(c_elements);
    CUDA_CHECK(cudaMemcpy(
        reference.data(), d_reference, c_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        actual.data(), d_C, c_bytes, cudaMemcpyDeviceToHost));

    return compare_results(reference, actual, M, N);
}

void print_performance(
    const char* name,
    const std::vector<float>& elapsed_times,
    int M,
    int N,
    int K)
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
        2.0 * static_cast<double>(M) * N * K;
    const double gflops = operation_count / (average_ms * 1.0e6);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n" << name << std::endl;
    std::cout << "Average time: " << average_ms << " ms" << std::endl;
    std::cout << "Minimum time: " << minimum_ms << " ms" << std::endl;
    std::cout << "Maximum time: " << maximum_ms << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOP/s" << std::endl;
}

int main(int argc, char** argv)
{
    const CheckMode check_mode = parse_check_mode(argc, argv);

    constexpr int M = 2048;
    constexpr int N = 2048;
    constexpr int K = 2048;
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
    float* d_reference = nullptr;
    std::uint32_t* d_l2_flush = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, c_bytes));
    CUDA_CHECK(cudaMalloc(&d_l2_flush, flush_elements * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMemset(d_l2_flush, 0, flush_elements * sizeof(std::uint32_t)));

    cublasHandle_t cublas_handle = nullptr;
    if (check_mode == CheckMode::Cublas) {
        CUDA_CHECK(cudaMalloc(&d_reference, c_bytes));
        check_cublas(cublasCreate(&cublas_handle), "cublasCreate");
        check_cublas(
            cublasSetMathMode(cublas_handle, CUBLAS_PEDANTIC_MATH),
            "cublasSetMathMode");
    }

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

    std::cout << "Matrix multiplication: " << M << " x " << K << " times " << K << " x " << N << std::endl;
    std::cout << "Block: " << BLOCK_SIZE << " x " << BLOCK_SIZE << std::endl;
    std::cout << "L2 cache: " << l2_bytes / 1024.0 / 1024.0 << " MiB" << std::endl;
    std::cout << "L2 flush buffer: " << flush_bytes / 1024.0 / 1024.0 << " MiB" << std::endl;
    std::cout << "Warmup iterations: " << WARMUP_ITERATIONS << std::endl;
    std::cout << "Measured iterations: " << MEASURED_ITERATIONS << std::endl;
    std::cout << "Result check: "
              << (check_mode == CheckMode::Cublas ? "cuBLAS" : "CPU")
              << std::endl;

    // Each iteration receives fresh random input. The L2 flush is launched 
    // after input generation and before the matrix multiplication.
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

    print_performance("matmul_naive", elapsed_times, M, N, K);
    bool success = check_mode == CheckMode::Cublas
        ? check_result_cublas(
              cublas_handle, d_A, d_B, d_C, d_reference, M, N, K)
        : check_result_cpu(d_A, d_B, d_C, M, N, K);

    // Shared-memory tiled kernel. Each thread still computes exactly one
    // element of C; shared memory only reduces redundant global-memory loads.
    elapsed_times.clear();
    check_curand(
        curandSetPseudoRandomGeneratorSeed(random_generator, RANDOM_SEED),
        "curandSetPseudoRandomGeneratorSeed");

    for (int iteration = 0; iteration < WARMUP_ITERATIONS; ++iteration) {
        check_curand(
            curandGenerateUniform(random_generator, d_A, a_elements),
            "curandGenerateUniform(A)");
        check_curand(
            curandGenerateUniform(random_generator, d_B, b_elements),
            "curandGenerateUniform(B)");
        flush_l2_cache<<<flush_blocks, FLUSH_BLOCK_SIZE>>>(
            d_l2_flush, flush_elements, iteration + 1);
        matmul_shmem<BLOCK_SIZE><<<grid, block>>>(
            d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

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
            nvtxRangePushA("matmul_shmem iteration 23");
        }
        CUDA_CHECK(cudaEventRecord(start));
        matmul_shmem<BLOCK_SIZE><<<grid, block>>>(
            d_A, d_B, d_C, M, N, K);
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

    print_performance("matmul_shmem", elapsed_times, M, N, K);
    success &= check_mode == CheckMode::Cublas
        ? check_result_cublas(
              cublas_handle, d_A, d_B, d_C, d_reference, M, N, K)
        : check_result_cpu(d_A, d_B, d_C, M, N, K);

    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    check_curand(curandDestroyGenerator(random_generator), "curandDestroyGenerator");
    if (check_mode == CheckMode::Cublas) {
        check_cublas(cublasDestroy(cublas_handle), "cublasDestroy");
        CUDA_CHECK(cudaFree(d_reference));
    }
    CUDA_CHECK(cudaFree(d_l2_flush));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_A));

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
