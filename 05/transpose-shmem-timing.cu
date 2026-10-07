#include <cuda_runtime.h>
#include <curand.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
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

template<int TILE_SIZE>
__global__ void naive(float* A, float* B, int M, int N)
{
    int tile_Ax = blockIdx.x;
    int tile_Ay = blockIdx.y;
    int pos_Ax = threadIdx.x;
    int pos_Ay = threadIdx.y;

    B[(tile_Ax*TILE_SIZE+pos_Ax)*N + tile_Ay*TILE_SIZE+pos_Ay] =
        A[(tile_Ay*TILE_SIZE+pos_Ay)*N + tile_Ax*TILE_SIZE+pos_Ax];
}

template<int TILE_SIZE>
__global__ void shmem(float* A, float* B, int M, int N)
{
    int tile_Ax = blockIdx.x;
    int tile_Ay = blockIdx.y;
    int pos_Ax = threadIdx.x;
    int pos_Ay = threadIdx.y;

    __shared__ float tmp[TILE_SIZE][TILE_SIZE];

    tmp[pos_Ay][pos_Ax] =
        A[(tile_Ay*TILE_SIZE+pos_Ay)*N + tile_Ax*TILE_SIZE+pos_Ax];

    __syncthreads();

    B[(tile_Ax*TILE_SIZE+pos_Ay)*N + tile_Ay*TILE_SIZE+pos_Ax] =
        tmp[pos_Ax][pos_Ay];
}

template<int TILE_SIZE>
__global__ void shmem_fix_bank_conflicts(float* A, float* B, int M, int N)
{
    int tile_Ax = blockIdx.x;
    int tile_Ay = blockIdx.y;
    int pos_Ax = threadIdx.x;
    int pos_Ay = threadIdx.y;

    __shared__ float tmp[TILE_SIZE][TILE_SIZE+1];

    tmp[pos_Ay][pos_Ax] =
        A[(tile_Ay*TILE_SIZE+pos_Ay)*N + tile_Ax*TILE_SIZE+pos_Ax];

    __syncthreads();

    B[(tile_Ax*TILE_SIZE+pos_Ay)*N + tile_Ay*TILE_SIZE+pos_Ax] =
        tmp[pos_Ax][pos_Ay];
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

bool print_results(
    const char* name,
    const std::vector<float>& elapsed_times,
    float* d_A,
    float* d_B,
    int M,
    int N)
{
    double total_ms = 0.0;
    float minimum_ms = std::numeric_limits<float>::max();
    float maximum_ms = 0.0f;

    for (float elapsed_ms : elapsed_times) {
        total_ms += elapsed_ms;
        minimum_ms = std::min(minimum_ms, elapsed_ms);
        maximum_ms = std::max(maximum_ms, elapsed_ms);
    }

    const std::size_t element_count = static_cast<std::size_t>(M) * N;
    const std::size_t matrix_bytes = element_count * sizeof(float);
    const double average_ms = total_ms / elapsed_times.size();
    const double bandwidth_gb_s =
        2.0 * matrix_bytes / (average_ms * 1.0e6);

    std::vector<float> h_A(element_count);
    std::vector<float> h_B(element_count);
    CUDA_CHECK(cudaMemcpy(
        h_A.data(), d_A, matrix_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        h_B.data(), d_B, matrix_bytes, cudaMemcpyDeviceToHost));

    bool success = true;
    for (int i = 0; i < M && success; ++i) {
        for (int j = 0; j < N; ++j) {
            if (h_B[i*N+j] != h_A[j*N+i]) {
                success = false;
                break;
            }
        }
    }

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n" << name << std::endl;
    std::cout << "Average time: " << average_ms << " ms" << std::endl;
    std::cout << "Minimum time: " << minimum_ms << " ms" << std::endl;
    std::cout << "Maximum time: " << maximum_ms << " ms" << std::endl;
    std::cout << "Effective bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;
    std::cout << "Success: " << success << std::endl;

    return success;
}

int main()
{
    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int TILE_SIZE = 16;
    constexpr int WARMUP_ITERATIONS = 10;
    constexpr int MEASURED_ITERATIONS = 100;
    constexpr int PROFILED_ITERATION = 22;
    constexpr int FLUSH_BLOCK_SIZE = 256;
    constexpr unsigned long long RANDOM_SEED = 1;

    static_assert(M % TILE_SIZE == 0);
    static_assert(N % TILE_SIZE == 0);

    const std::size_t element_count = static_cast<std::size_t>(M) * N;
    const std::size_t matrix_bytes = element_count * sizeof(float);

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

    dim3 grid(N/TILE_SIZE, M/TILE_SIZE);
    dim3 block(TILE_SIZE, TILE_SIZE);

    float* d_A = nullptr;
    float* d_B = nullptr;
    std::uint32_t* d_l2_flush = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_l2_flush, flush_elements * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMemset(d_l2_flush, 0, flush_elements * sizeof(std::uint32_t)));

    curandGenerator_t random_generator;
    check_curand(
        curandCreateGenerator(&random_generator, CURAND_RNG_PSEUDO_DEFAULT),
        "curandCreateGenerator");

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::cout << "Matrix: " << M << " x " << N << std::endl;
    std::cout << "Tile: " << TILE_SIZE << " x " << TILE_SIZE << std::endl;
    std::cout << "L2 cache: " << l2_bytes / 1024.0 / 1024.0 << " MiB" << std::endl;
    std::cout << "L2 flush buffer: " << flush_bytes / 1024.0 / 1024.0
              << " MiB" << std::endl;
    std::cout << "Warmup iterations: " << WARMUP_ITERATIONS << std::endl;
    std::cout << "Measured iterations: " << MEASURED_ITERATIONS << std::endl;

    bool success = true;
    std::vector<float> elapsed_times;
    elapsed_times.reserve(MEASURED_ITERATIONS);

    // Naive kernel
    check_curand(
        curandSetPseudoRandomGeneratorSeed(random_generator, RANDOM_SEED),
        "curandSetPseudoRandomGeneratorSeed");
    for (int iteration = 0; iteration < WARMUP_ITERATIONS; ++iteration) {
        check_curand(
            curandGenerateUniform(random_generator, d_A, element_count),
            "curandGenerateUniform");
        flush_l2_cache<<<flush_blocks, FLUSH_BLOCK_SIZE>>>(
            d_l2_flush, flush_elements, iteration + 1);
        naive<TILE_SIZE><<<grid, block>>>(d_A, d_B, M, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int iteration = 0; iteration < MEASURED_ITERATIONS; ++iteration) {
        check_curand(
            curandGenerateUniform(random_generator, d_A, element_count),
            "curandGenerateUniform");
        flush_l2_cache<<<flush_blocks, FLUSH_BLOCK_SIZE>>>(
            d_l2_flush, flush_elements, iteration + 1);
        if (iteration == PROFILED_ITERATION) {
            nvtxRangePushA("naive iteration 23");
        }
        CUDA_CHECK(cudaEventRecord(start));
        naive<TILE_SIZE><<<grid, block>>>(d_A, d_B, M, N);
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
    success &= print_results("naive", elapsed_times, d_A, d_B, M, N);

    // Shared-memory kernel
    elapsed_times.clear();
    check_curand(
        curandSetPseudoRandomGeneratorSeed(random_generator, RANDOM_SEED),
        "curandSetPseudoRandomGeneratorSeed");
    for (int iteration = 0; iteration < WARMUP_ITERATIONS; ++iteration) {
        check_curand(
            curandGenerateUniform(random_generator, d_A, element_count),
            "curandGenerateUniform");
        flush_l2_cache<<<flush_blocks, FLUSH_BLOCK_SIZE>>>(
            d_l2_flush, flush_elements, iteration + 1);
        shmem<TILE_SIZE><<<grid, block>>>(d_A, d_B, M, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int iteration = 0; iteration < MEASURED_ITERATIONS; ++iteration) {
        check_curand(
            curandGenerateUniform(random_generator, d_A, element_count),
            "curandGenerateUniform");
        flush_l2_cache<<<flush_blocks, FLUSH_BLOCK_SIZE>>>(
            d_l2_flush, flush_elements, iteration + 1);
        if (iteration == PROFILED_ITERATION) {
            nvtxRangePushA("shmem iteration 23");
        }
        CUDA_CHECK(cudaEventRecord(start));
        shmem<TILE_SIZE><<<grid, block>>>(d_A, d_B, M, N);
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
    success &= print_results("shmem", elapsed_times, d_A, d_B, M, N);

    // Shared-memory kernel with padding
    elapsed_times.clear();
    check_curand(
        curandSetPseudoRandomGeneratorSeed(random_generator, RANDOM_SEED),
        "curandSetPseudoRandomGeneratorSeed");
    for (int iteration = 0; iteration < WARMUP_ITERATIONS; ++iteration) {
        check_curand(
            curandGenerateUniform(random_generator, d_A, element_count),
            "curandGenerateUniform");
        flush_l2_cache<<<flush_blocks, FLUSH_BLOCK_SIZE>>>(
            d_l2_flush, flush_elements, iteration + 1);
        shmem_fix_bank_conflicts<TILE_SIZE><<<grid, block>>>(d_A, d_B, M, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int iteration = 0; iteration < MEASURED_ITERATIONS; ++iteration) {
        check_curand(
            curandGenerateUniform(random_generator, d_A, element_count),
            "curandGenerateUniform");
        flush_l2_cache<<<flush_blocks, FLUSH_BLOCK_SIZE>>>(
            d_l2_flush, flush_elements, iteration + 1);
        if (iteration == PROFILED_ITERATION) {
            nvtxRangePushA("shmem_fix_bank_conflicts iteration 23");
        }
        CUDA_CHECK(cudaEventRecord(start));
        shmem_fix_bank_conflicts<TILE_SIZE><<<grid, block>>>(d_A, d_B, M, N);
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
    success &= print_results(
        "shmem_fix_bank_conflicts", elapsed_times, d_A, d_B, M, N);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    check_curand(curandDestroyGenerator(random_generator), "curandDestroyGenerator");
    CUDA_CHECK(cudaFree(d_l2_flush));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_A));

    return success ? 0 : 1;
}
