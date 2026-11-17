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

#define CEIL_DIV(numerator, denominator) \
    (((numerator) + (denominator) - 1) / (denominator))

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
    int MATRIX_M,
    int MATRIX_N,
    int MATRIX_K)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= MATRIX_M || column >= MATRIX_N) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < MATRIX_K; ++k) {
        sum += A[row * MATRIX_K + k] * B[k * MATRIX_N + column];
    }
    C[row * MATRIX_N + column] = sum;
}

template<int BLOCK_SIZE>
void launch_matmul_naive(
    const float* A,
    const float* B,
    float* C,
    int MATRIX_M,
    int MATRIX_N,
    int MATRIX_K)
{
    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid(
        CEIL_DIV(MATRIX_N, BLOCK_SIZE),
        CEIL_DIV(MATRIX_M, BLOCK_SIZE));

    matmul_naive<<<grid, block>>>(
        A, B, C, MATRIX_M, MATRIX_N, MATRIX_K);
}

// TILE_SIZE is equal to blockDim.x and blockDim.y
// Why this kernel needs this at compile time?
template<int TILE_SIZE>
__global__ void matmul_shmem(
    const float* A,
    const float* B,
    float* C,
    int MATRIX_M,
    int MATRIX_N,
    int MATRIX_K)
{
    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

    // Indexing C: (colum, row)
    const int column = blockIdx.x * TILE_SIZE + threadIdx.x;
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Indexing A: (c_row, a_column) -> a_column depends on the inner product dim
    // Indexing B: (b_row, c_column) -> b_row depends on the inner product dim

    float sum = 0.0f;
    const int tile_count = CEIL_DIV(MATRIX_K, TILE_SIZE);

    // Main loop: inner product dimension
    for (int tile = 0; tile < tile_count; ++tile) {
        const int a_column = tile * TILE_SIZE + threadIdx.x;
        const int b_row = tile * TILE_SIZE + threadIdx.y;

        if(row < MATRIX_M && a_column < MATRIX_K){
            A_tile[threadIdx.y][threadIdx.x] = A[row * MATRIX_K + a_column];
        }else{
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if(b_row < MATRIX_K && column < MATRIX_N){
            B_tile[threadIdx.y][threadIdx.x] = B[b_row * MATRIX_N + column];
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

    if (row < MATRIX_M && column < MATRIX_N) {
        C[row * MATRIX_N + column] = sum;
    }
}

template<int TILE_SIZE>
void launch_matmul_shmem(
    const float* A,
    const float* B,
    float* C,
    int MATRIX_M,
    int MATRIX_N,
    int MATRIX_K)
{
    const dim3 block(TILE_SIZE, TILE_SIZE);
    const dim3 grid(
        CEIL_DIV(MATRIX_N, TILE_SIZE),
        CEIL_DIV(MATRIX_M, TILE_SIZE));

    matmul_shmem<TILE_SIZE><<<grid, block>>>(
        A, B, C, MATRIX_M, MATRIX_N, MATRIX_K);
}

template<int TILE_SIZE, int THREAD_TILE_SIZE>
__global__ void matmul_thread_tiling(
    const float* A,
    const float* B,
    float* C,
    int MATRIX_M,
    int MATRIX_N,
    int MATRIX_K)
{
    const int base_column =
        blockIdx.x * TILE_SIZE * THREAD_TILE_SIZE + threadIdx.x;
    const int base_row =
        blockIdx.y * TILE_SIZE * THREAD_TILE_SIZE + threadIdx.y;

    float sums[THREAD_TILE_SIZE][THREAD_TILE_SIZE] = {};

    for (int k = 0; k < MATRIX_K; ++k) {
        float A_values[THREAD_TILE_SIZE];
        float B_values[THREAD_TILE_SIZE];

#pragma unroll
        for (int output_row = 0;
             output_row < THREAD_TILE_SIZE;
             ++output_row) {
            const int row = base_row + output_row * TILE_SIZE;
            if (row < MATRIX_M) {
                A_values[output_row] = A[row * MATRIX_K + k];
            } else {
                A_values[output_row] = 0.0f;
            }
        }

#pragma unroll
        for (int output_column = 0;
             output_column < THREAD_TILE_SIZE;
             ++output_column) {
            const int column = base_column + output_column * TILE_SIZE;
            if (column < MATRIX_N) {
                B_values[output_column] = B[k * MATRIX_N + column];
            } else {
                B_values[output_column] = 0.0f;
            }
        }

#pragma unroll
        for (int output_row = 0;
             output_row < THREAD_TILE_SIZE;
             ++output_row) {
#pragma unroll
            for (int output_column = 0;
                 output_column < THREAD_TILE_SIZE;
                 ++output_column) {
                sums[output_row][output_column] +=
                    A_values[output_row] * B_values[output_column];
            }
        }
    }

#pragma unroll
    for (int output_row = 0;
         output_row < THREAD_TILE_SIZE;
         ++output_row) {
        const int row = base_row + output_row * TILE_SIZE;

#pragma unroll
        for (int output_column = 0;
             output_column < THREAD_TILE_SIZE;
             ++output_column) {
            const int column = base_column + output_column * TILE_SIZE;
            if (row < MATRIX_M && column < MATRIX_N) {
                C[row * MATRIX_N + column] = sums[output_row][output_column];
            }
        }
    }
}

template<int TILE_SIZE, int THREAD_TILE_SIZE>
void launch_matmul_thread_tiling(
    const float* A,
    const float* B,
    float* C,
    int MATRIX_M,
    int MATRIX_N,
    int MATRIX_K)
{
    constexpr int THREAD_TILING_BLOCK_TILE_SIZE =
        TILE_SIZE * THREAD_TILE_SIZE;
    const dim3 block(TILE_SIZE, TILE_SIZE);
    const dim3 grid(
        CEIL_DIV(MATRIX_N, THREAD_TILING_BLOCK_TILE_SIZE),
        CEIL_DIV(MATRIX_M, THREAD_TILING_BLOCK_TILE_SIZE));

    matmul_thread_tiling<TILE_SIZE, THREAD_TILE_SIZE><<<grid, block>>>(
        A, B, C, MATRIX_M, MATRIX_N, MATRIX_K);
}

template<
    int BLOCK_TILE_M,
    int BLOCK_TILE_N,
    int THREAD_TILE_M,
    int THREAD_TILE_N>
__global__ void matmul_thread_tiling_tune(
    const float* A,
    const float* B,
    float* C,
    int MATRIX_M,
    int MATRIX_N,
    int MATRIX_K)
{
    static_assert(
        BLOCK_TILE_M % THREAD_TILE_M == 0,
        "BLOCK_TILE_M must be divisible by THREAD_TILE_M");
    static_assert(
        BLOCK_TILE_N % THREAD_TILE_N == 0,
        "BLOCK_TILE_N must be divisible by THREAD_TILE_N");

    constexpr int THREAD_TILING_TUNE_BLOCK_SIZE_Y =
        BLOCK_TILE_M / THREAD_TILE_M;
    constexpr int THREAD_TILING_TUNE_BLOCK_SIZE_X =
        BLOCK_TILE_N / THREAD_TILE_N;

    const int block_row = blockIdx.y * BLOCK_TILE_M;
    const int block_column = blockIdx.x * BLOCK_TILE_N;

    float sums[THREAD_TILE_M][THREAD_TILE_N] = {};

    for (int k = 0; k < MATRIX_K; ++k) {
        float A_values[THREAD_TILE_M];
        float B_values[THREAD_TILE_N];

#pragma unroll
        for (int output_row = 0;
             output_row < THREAD_TILE_M;
             ++output_row) {
            const int row =
                block_row + threadIdx.y +
                output_row * THREAD_TILING_TUNE_BLOCK_SIZE_Y;
            if (row < MATRIX_M) {
                A_values[output_row] = A[row * MATRIX_K + k];
            } else {
                A_values[output_row] = 0.0f;
            }
        }

#pragma unroll
        for (int output_column = 0;
             output_column < THREAD_TILE_N;
             ++output_column) {
            const int column =
                block_column + threadIdx.x +
                output_column * THREAD_TILING_TUNE_BLOCK_SIZE_X;
            if (column < MATRIX_N) {
                B_values[output_column] = B[k * MATRIX_N + column];
            } else {
                B_values[output_column] = 0.0f;
            }
        }

#pragma unroll
        for (int output_row = 0;
             output_row < THREAD_TILE_M;
             ++output_row) {
#pragma unroll
            for (int output_column = 0;
                 output_column < THREAD_TILE_N;
                 ++output_column) {
                sums[output_row][output_column] +=
                    A_values[output_row] * B_values[output_column];
            }
        }
    }

#pragma unroll
    for (int output_row = 0;
         output_row < THREAD_TILE_M;
         ++output_row) {
        const int row =
            block_row + threadIdx.y +
            output_row * THREAD_TILING_TUNE_BLOCK_SIZE_Y;

#pragma unroll
        for (int output_column = 0;
             output_column < THREAD_TILE_N;
             ++output_column) {
            const int column =
                block_column + threadIdx.x +
                output_column * THREAD_TILING_TUNE_BLOCK_SIZE_X;
            if (row < MATRIX_M && column < MATRIX_N) {
                C[row * MATRIX_N + column] = sums[output_row][output_column];
            }
        }
    }
}

template<
    int BLOCK_TILE_M,
    int BLOCK_TILE_N,
    int THREAD_TILE_M,
    int THREAD_TILE_N>
void launch_matmul_thread_tiling_tune(
    const float* A,
    const float* B,
    float* C,
    int MATRIX_M,
    int MATRIX_N,
    int MATRIX_K)
{
    static_assert(
        BLOCK_TILE_M % THREAD_TILE_M == 0,
        "BLOCK_TILE_M must be divisible by THREAD_TILE_M");
    static_assert(
        BLOCK_TILE_N % THREAD_TILE_N == 0,
        "BLOCK_TILE_N must be divisible by THREAD_TILE_N");

    constexpr int THREAD_TILING_TUNE_BLOCK_SIZE_X =
        BLOCK_TILE_N / THREAD_TILE_N;
    constexpr int THREAD_TILING_TUNE_BLOCK_SIZE_Y =
        BLOCK_TILE_M / THREAD_TILE_M;
    static_assert(
        THREAD_TILING_TUNE_BLOCK_SIZE_X *
                THREAD_TILING_TUNE_BLOCK_SIZE_Y <=
            1024,
        "The CUDA block cannot contain more than 1024 threads");

    const dim3 block(
        THREAD_TILING_TUNE_BLOCK_SIZE_X,
        THREAD_TILING_TUNE_BLOCK_SIZE_Y);
    const dim3 grid(
        CEIL_DIV(MATRIX_N, BLOCK_TILE_N),
        CEIL_DIV(MATRIX_M, BLOCK_TILE_M));

    matmul_thread_tiling_tune<
        BLOCK_TILE_M,
        BLOCK_TILE_N,
        THREAD_TILE_M,
        THREAD_TILE_N>
        <<<grid, block>>>(A, B, C, MATRIX_M, MATRIX_N, MATRIX_K);
}

template<int TILE_SIZE, int THREAD_TILE_SIZE>
__global__ void matmul_shmem_thread_tiling(
    const float* A,
    const float* B,
    float* C,
    int MATRIX_M,
    int MATRIX_N,
    int MATRIX_K)
{
    __shared__ float A_tile[THREAD_TILE_SIZE * TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][THREAD_TILE_SIZE * TILE_SIZE];

    const int base_column =
        blockIdx.x * TILE_SIZE * THREAD_TILE_SIZE + threadIdx.x;
    const int base_row =
        blockIdx.y * TILE_SIZE * THREAD_TILE_SIZE + threadIdx.y;

    float sums[THREAD_TILE_SIZE][THREAD_TILE_SIZE] = {};
    const int tile_count = CEIL_DIV(MATRIX_K, TILE_SIZE);

    for (int tile = 0; tile < tile_count; ++tile) {
        const int a_column = tile * TILE_SIZE + threadIdx.x;
        const int b_row = tile * TILE_SIZE + threadIdx.y;

#pragma unroll
        for (int output_row = 0;
             output_row < THREAD_TILE_SIZE;
             ++output_row) {
            const int row = base_row + output_row * TILE_SIZE;
            const int shared_row = threadIdx.y + output_row * TILE_SIZE;
            if (row < MATRIX_M && a_column < MATRIX_K) {
                A_tile[shared_row][threadIdx.x] = A[row * MATRIX_K + a_column];
            } else {
                A_tile[shared_row][threadIdx.x] = 0.0f;
            }
        }

#pragma unroll
        for (int output_column = 0;
             output_column < THREAD_TILE_SIZE;
             ++output_column) {
            const int column = base_column + output_column * TILE_SIZE;
            const int shared_column =
                threadIdx.x + output_column * TILE_SIZE;
            if (b_row < MATRIX_K && column < MATRIX_N) {
                B_tile[threadIdx.y][shared_column] = B[b_row * MATRIX_N + column];
            } else {
                B_tile[threadIdx.y][shared_column] = 0.0f;
            }
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            float A_values[THREAD_TILE_SIZE];
            float B_values[THREAD_TILE_SIZE];

#pragma unroll
            for (int output_row = 0;
                 output_row < THREAD_TILE_SIZE;
                 ++output_row) {
                const int shared_row =
                    threadIdx.y + output_row * TILE_SIZE;
                A_values[output_row] = A_tile[shared_row][k];
            }

#pragma unroll
            for (int output_column = 0;
                 output_column < THREAD_TILE_SIZE;
                 ++output_column) {
                const int shared_column =
                    threadIdx.x + output_column * TILE_SIZE;
                B_values[output_column] = B_tile[k][shared_column];
            }

#pragma unroll
            for (int output_row = 0;
                 output_row < THREAD_TILE_SIZE;
                 ++output_row) {
#pragma unroll
                for (int output_column = 0;
                     output_column < THREAD_TILE_SIZE;
                     ++output_column) {
                    sums[output_row][output_column] +=
                        A_values[output_row] * B_values[output_column];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int output_row = 0;
         output_row < THREAD_TILE_SIZE;
         ++output_row) {
        const int row = base_row + output_row * TILE_SIZE;

#pragma unroll
        for (int output_column = 0;
             output_column < THREAD_TILE_SIZE;
             ++output_column) {
            const int column = base_column + output_column * TILE_SIZE;
            if (row < MATRIX_M && column < MATRIX_N) {
                C[row * MATRIX_N + column] = sums[output_row][output_column];
            }
        }
    }
}

template<int TILE_SIZE, int THREAD_TILE_SIZE>
void launch_matmul_shmem_thread_tiling(
    const float* A,
    const float* B,
    float* C,
    int MATRIX_M,
    int MATRIX_N,
    int MATRIX_K)
{
    constexpr int SHMEM_THREAD_TILING_BLOCK_TILE_SIZE =
        TILE_SIZE * THREAD_TILE_SIZE;
    const dim3 block(TILE_SIZE, TILE_SIZE);
    const dim3 grid(
        CEIL_DIV(MATRIX_N, SHMEM_THREAD_TILING_BLOCK_TILE_SIZE),
        CEIL_DIV(MATRIX_M, SHMEM_THREAD_TILING_BLOCK_TILE_SIZE));

    matmul_shmem_thread_tiling<TILE_SIZE, THREAD_TILE_SIZE>
        <<<grid, block>>>(A, B, C, MATRIX_M, MATRIX_N, MATRIX_K);
}

template<
    int BLOCK_TILE_M,
    int BLOCK_TILE_N,
    int BLOCK_TILE_K,
    int THREAD_TILE_M,
    int THREAD_TILE_N>
__global__ void matmul_shmem_thread_tiling_tune(
    const float* A,
    const float* B,
    float* C,
    int MATRIX_M,
    int MATRIX_N,
    int MATRIX_K)
{
    static_assert(
        BLOCK_TILE_M % THREAD_TILE_M == 0,
        "BLOCK_TILE_M must be divisible by THREAD_TILE_M");
    static_assert(
        BLOCK_TILE_N % THREAD_TILE_N == 0,
        "BLOCK_TILE_N must be divisible by THREAD_TILE_N");

    constexpr int SHMEM_THREAD_TILING_TUNE_BLOCK_SIZE_Y =
        BLOCK_TILE_M / THREAD_TILE_M;
    constexpr int SHMEM_THREAD_TILING_TUNE_BLOCK_SIZE_X =
        BLOCK_TILE_N / THREAD_TILE_N;
    constexpr int SHMEM_THREAD_TILING_TUNE_THREAD_COUNT =
        SHMEM_THREAD_TILING_TUNE_BLOCK_SIZE_X *
        SHMEM_THREAD_TILING_TUNE_BLOCK_SIZE_Y;

    __shared__ float A_tile[BLOCK_TILE_M][BLOCK_TILE_K];
    __shared__ float B_tile[BLOCK_TILE_K][BLOCK_TILE_N];

    const int linear_thread =
        threadIdx.y * SHMEM_THREAD_TILING_TUNE_BLOCK_SIZE_X +
        threadIdx.x;
    const int block_row = blockIdx.y * BLOCK_TILE_M;
    const int block_column = blockIdx.x * BLOCK_TILE_N;

    float sums[THREAD_TILE_M][THREAD_TILE_N] = {};
    const int tile_count = CEIL_DIV(MATRIX_K, BLOCK_TILE_K);

    for (int tile = 0; tile < tile_count; ++tile) {
        const int block_k = tile * BLOCK_TILE_K;

        for (int index = linear_thread;
             index < BLOCK_TILE_M * BLOCK_TILE_K;
             index += SHMEM_THREAD_TILING_TUNE_THREAD_COUNT) {
            const int shared_row = index / BLOCK_TILE_K;
            const int shared_k = index % BLOCK_TILE_K;
            const int row = block_row + shared_row;
            const int k = block_k + shared_k;

            if (row < MATRIX_M && k < MATRIX_K) {
                A_tile[shared_row][shared_k] =
                    A[row * MATRIX_K + k];
            } else {
                A_tile[shared_row][shared_k] = 0.0f;
            }
        }

        for (int index = linear_thread;
             index < BLOCK_TILE_K * BLOCK_TILE_N;
             index += SHMEM_THREAD_TILING_TUNE_THREAD_COUNT) {
            const int shared_k = index / BLOCK_TILE_N;
            const int shared_column = index % BLOCK_TILE_N;
            const int k = block_k + shared_k;
            const int column = block_column + shared_column;

            if (k < MATRIX_K && column < MATRIX_N) {
                B_tile[shared_k][shared_column] =
                    B[k * MATRIX_N + column];
            } else {
                B_tile[shared_k][shared_column] = 0.0f;
            }
        }

        __syncthreads();

#pragma unroll
        for (int tile_k = 0; tile_k < BLOCK_TILE_K; ++tile_k) {
            float A_values[THREAD_TILE_M];
            float B_values[THREAD_TILE_N];

#pragma unroll
            for (int output_row = 0;
                 output_row < THREAD_TILE_M;
                 ++output_row) {
                const int shared_row =
                    threadIdx.y + output_row *
                                      SHMEM_THREAD_TILING_TUNE_BLOCK_SIZE_Y;
                A_values[output_row] = A_tile[shared_row][tile_k];
            }

#pragma unroll
            for (int output_column = 0;
                 output_column < THREAD_TILE_N;
                 ++output_column) {
                const int shared_column =
                    threadIdx.x + output_column *
                                      SHMEM_THREAD_TILING_TUNE_BLOCK_SIZE_X;
                B_values[output_column] =
                    B_tile[tile_k][shared_column];
            }

#pragma unroll
            for (int output_row = 0;
                 output_row < THREAD_TILE_M;
                 ++output_row) {
#pragma unroll
                for (int output_column = 0;
                     output_column < THREAD_TILE_N;
                     ++output_column) {
                    sums[output_row][output_column] +=
                        A_values[output_row] * B_values[output_column];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int output_row = 0;
         output_row < THREAD_TILE_M;
         ++output_row) {
        const int row =
            block_row + threadIdx.y + output_row *
                                          SHMEM_THREAD_TILING_TUNE_BLOCK_SIZE_Y;

#pragma unroll
        for (int output_column = 0;
             output_column < THREAD_TILE_N;
             ++output_column) {
            const int column =
                block_column + threadIdx.x + output_column *
                                                  SHMEM_THREAD_TILING_TUNE_BLOCK_SIZE_X;

            if (row < MATRIX_M && column < MATRIX_N) {
                C[row * MATRIX_N + column] =
                    sums[output_row][output_column];
            }
        }
    }
}

template<
    int BLOCK_TILE_M,
    int BLOCK_TILE_N,
    int BLOCK_TILE_K,
    int THREAD_TILE_M,
    int THREAD_TILE_N>
void launch_matmul_shmem_thread_tiling_tune(
    const float* A,
    const float* B,
    float* C,
    int MATRIX_M,
    int MATRIX_N,
    int MATRIX_K)
{
    static_assert(
        BLOCK_TILE_M % THREAD_TILE_M == 0,
        "BLOCK_TILE_M must be divisible by THREAD_TILE_M");
    static_assert(
        BLOCK_TILE_N % THREAD_TILE_N == 0,
        "BLOCK_TILE_N must be divisible by THREAD_TILE_N");

    constexpr int SHMEM_THREAD_TILING_TUNE_BLOCK_SIZE_X =
        BLOCK_TILE_N / THREAD_TILE_N;
    constexpr int SHMEM_THREAD_TILING_TUNE_BLOCK_SIZE_Y =
        BLOCK_TILE_M / THREAD_TILE_M;
    static_assert(
        SHMEM_THREAD_TILING_TUNE_BLOCK_SIZE_X *
                SHMEM_THREAD_TILING_TUNE_BLOCK_SIZE_Y <=
            1024,
        "The CUDA block cannot contain more than 1024 threads");

    const dim3 block(
        SHMEM_THREAD_TILING_TUNE_BLOCK_SIZE_X,
        SHMEM_THREAD_TILING_TUNE_BLOCK_SIZE_Y);
    const dim3 grid(
        CEIL_DIV(MATRIX_N, BLOCK_TILE_N),
        CEIL_DIV(MATRIX_M, BLOCK_TILE_M));

    matmul_shmem_thread_tiling_tune<
        BLOCK_TILE_M,
        BLOCK_TILE_N,
        BLOCK_TILE_K,
        THREAD_TILE_M,
        THREAD_TILE_N>
        <<<grid, block>>>(A, B, C, MATRIX_M, MATRIX_N, MATRIX_K);
}

bool compare_results(
    const std::vector<float>& reference,
    const std::vector<float>& actual,
    int MATRIX_M,
    int MATRIX_N)
{
    constexpr float CHECK_ABSOLUTE_TOLERANCE = 1.0e-3f;
    constexpr float CHECK_RELATIVE_TOLERANCE = 1.0e-4f;

    std::size_t error_count = 0;
    float maximum_absolute_error = 0.0f;
    float maximum_relative_error = 0.0f;

    for (int row = 0; row < MATRIX_M; ++row) {
        for (int column = 0; column < MATRIX_N; ++column) {
            const std::size_t index =
                static_cast<std::size_t>(row) * MATRIX_N + column;
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
                CHECK_ABSOLUTE_TOLERANCE +
                CHECK_RELATIVE_TOLERANCE * std::abs(expected);
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
    int MATRIX_M,
    int MATRIX_N,
    int MATRIX_K)
{
    const std::size_t a_elements = static_cast<std::size_t>(MATRIX_M) * MATRIX_K;
    const std::size_t b_elements = static_cast<std::size_t>(MATRIX_K) * MATRIX_N;
    const std::size_t c_elements = static_cast<std::size_t>(MATRIX_M) * MATRIX_N;

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

    for (int row = 0; row < MATRIX_M; ++row) {
        for (int column = 0; column < MATRIX_N; ++column) {
            float sum = 0.0f;
            for (int k = 0; k < MATRIX_K; ++k) {
                sum += h_A[row * MATRIX_K + k] * h_B[k * MATRIX_N + column];
            }
            reference[static_cast<std::size_t>(row) * MATRIX_N + column] = sum;
        }
    }

    return compare_results(reference, h_C, MATRIX_M, MATRIX_N);
}

void launch_cublas_matmul(
    cublasHandle_t handle,
    const float* A,
    const float* B,
    float* C,
    int MATRIX_M,
    int MATRIX_N,
    int MATRIX_K)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS uses column-major matrices. Swapping A and B computes
    // C^T = B^T * A^T while preserving the row-major memory layout.
    check_cublas(
        cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            MATRIX_N,
            MATRIX_M,
            MATRIX_K,
            &alpha,
            B,
            MATRIX_N,
            A,
            MATRIX_K,
            &beta,
            C,
            MATRIX_N),
        "cublasSgemm");
}

bool check_result_cublas(
    cublasHandle_t handle,
    const float* d_A,
    const float* d_B,
    const float* d_C,
    float* d_reference,
    int MATRIX_M,
    int MATRIX_N,
    int MATRIX_K)
{
    // Explanation of how we can enforce a specific computation
    // https://leimao.github.io/blog/cuBLAS-Transpose-Column-Major-Relationship/
    launch_cublas_matmul(
        handle,
        d_A,
        d_B,
        d_reference,
        MATRIX_M,
        MATRIX_N,
        MATRIX_K);

    const std::size_t c_elements = static_cast<std::size_t>(MATRIX_M) * MATRIX_N;
    const std::size_t c_bytes = c_elements * sizeof(float);
    std::vector<float> reference(c_elements);
    std::vector<float> actual(c_elements);
    CUDA_CHECK(cudaMemcpy(
        reference.data(), d_reference, c_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        actual.data(), d_C, c_bytes, cudaMemcpyDeviceToHost));

    return compare_results(reference, actual, MATRIX_M, MATRIX_N);
}

bool check_result(
    CheckMode check_mode,
    cublasHandle_t cublas_handle,
    const float* d_A,
    const float* d_B,
    const float* d_C,
    float* d_reference,
    int MATRIX_M,
    int MATRIX_N,
    int MATRIX_K)
{
    if (check_mode == CheckMode::Cublas) {
        return check_result_cublas(
            cublas_handle, d_A, d_B, d_C, d_reference, MATRIX_M, MATRIX_N, MATRIX_K);
    } else {
        return check_result_cpu(d_A, d_B, d_C, MATRIX_M, MATRIX_N, MATRIX_K);
    }
}

int main(int argc, char** argv)
{
    const CheckMode check_mode = parse_check_mode(argc, argv);

    constexpr int MATRIX_M = 2048;
    constexpr int MATRIX_N = 2048;
    constexpr int MATRIX_K = 2048;

    constexpr int BENCHMARK_WARMUP_ITERATIONS = 10;
    constexpr int BENCHMARK_MEASURED_ITERATIONS = 100;
    constexpr int BENCHMARK_PROFILED_ITERATION = 22;
    constexpr int L2_FLUSH_BLOCK_SIZE = 256;
    constexpr unsigned long long INPUT_RANDOM_SEED = 1;

    const std::size_t a_elements =
        static_cast<std::size_t>(MATRIX_M) * MATRIX_K;
    const std::size_t b_elements =
        static_cast<std::size_t>(MATRIX_K) * MATRIX_N;
    const std::size_t c_elements =
        static_cast<std::size_t>(MATRIX_M) * MATRIX_N;
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
        CEIL_DIV(flush_bytes, sizeof(std::uint32_t));

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

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::cout << "Matrix multiplication: "
              << MATRIX_M << " x " << MATRIX_K << " times "
              << MATRIX_K << " x " << MATRIX_N << std::endl;
    std::cout << "L2 cache: " << l2_bytes / 1024.0 / 1024.0 << " MiB" << std::endl;
    std::cout << "L2 flush buffer: " << flush_bytes / 1024.0 / 1024.0 << " MiB" << std::endl;
    std::cout << "Warmup iterations: " << BENCHMARK_WARMUP_ITERATIONS << std::endl;
    std::cout << "Measured iterations: " << BENCHMARK_MEASURED_ITERATIONS << std::endl;
    if (check_mode == CheckMode::Cublas) {
        std::cout << "Result check: cuBLAS" << std::endl;
    } else {
        std::cout << "Result check: CPU" << std::endl;
    }

    // Each iteration receives fresh random input. The L2 flush is launched 
    // after input generation and before the matrix multiplication.
    bool success = true;

    // ------------------------ matmul_naive ------------------------

    // Naive matrix multiplication kernel: BLOCK_SIZE is the block tile size.
    // Each block computes a 
    // - BLOCK_SIZE*BLOCK_SIZE output with 
    // - BLOCK_SIZE**2 threads, thus, the block size is actually BLOCK_SIZE**2.
    {
        constexpr int NAIVE_BLOCK_SIZE = 16;

        std::vector<float> elapsed_times;
        elapsed_times.reserve(BENCHMARK_MEASURED_ITERATIONS);
        check_curand(
            curandSetPseudoRandomGeneratorSeed(
                random_generator,
                INPUT_RANDOM_SEED),
            "curandSetPseudoRandomGeneratorSeed");

        std::cout << "\nRunning matmul_naive" << std::endl;
        std::cout << "Template parameters: none" << std::endl;

        for (int iteration = 0; iteration < BENCHMARK_WARMUP_ITERATIONS; ++iteration) {
            check_curand(
                curandGenerateUniform(random_generator, d_A, a_elements),
                "curandGenerateUniform(A)");
            check_curand(
                curandGenerateUniform(random_generator, d_B, b_elements),
                "curandGenerateUniform(B)");
            launch_flush_l2_cache<L2_FLUSH_BLOCK_SIZE>(
                d_l2_flush, flush_elements, iteration + 1);
            launch_matmul_naive<NAIVE_BLOCK_SIZE>(
                d_A, d_B, d_C, MATRIX_M, MATRIX_N, MATRIX_K);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int iteration = 0; iteration < BENCHMARK_MEASURED_ITERATIONS; ++iteration) {
            check_curand(
                curandGenerateUniform(random_generator, d_A, a_elements),
                "curandGenerateUniform(A)");
            check_curand(
                curandGenerateUniform(random_generator, d_B, b_elements),
                "curandGenerateUniform(B)");
            launch_flush_l2_cache<L2_FLUSH_BLOCK_SIZE>(
                d_l2_flush, flush_elements, iteration + 1);

            if (iteration == BENCHMARK_PROFILED_ITERATION) {
                nvtxRangePushA("matmul_naive iteration 23");
            }
            CUDA_CHECK(cudaEventRecord(start));
            launch_matmul_naive<NAIVE_BLOCK_SIZE>(
                d_A, d_B, d_C, MATRIX_M, MATRIX_N, MATRIX_K);
            CUDA_CHECK(cudaEventRecord(stop));
            if (iteration == BENCHMARK_PROFILED_ITERATION) {
                nvtxRangePop();
            }

            CUDA_CHECK(cudaEventSynchronize(stop));
            float elapsed_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
            elapsed_times.push_back(elapsed_ms);
        }
        CUDA_CHECK(cudaGetLastError());

        print_performance("matmul_naive", elapsed_times, MATRIX_M, MATRIX_N, MATRIX_K);
        success &= check_result(
            check_mode,
            cublas_handle,
            d_A,
            d_B,
            d_C,
            d_reference,
            MATRIX_M,
            MATRIX_N,
            MATRIX_K);
    }

    // ------------------------ matmul_shmem ------------------------

    // Shared-memory tiled kernel. Each thread still computes exactly one
    // element of C; shared memory only reduces redundant global-memory loads (if any).
    {
        constexpr int SHMEM_BLOCK_SIZE = 16;

        std::vector<float> elapsed_times;
        elapsed_times.reserve(BENCHMARK_MEASURED_ITERATIONS);
        check_curand(
            curandSetPseudoRandomGeneratorSeed(
                random_generator,
                INPUT_RANDOM_SEED),
            "curandSetPseudoRandomGeneratorSeed");

        std::cout << "\nRunning matmul_shmem" << std::endl;
        std::cout << "Template parameters:" << std::endl;
        std::cout << "  TILE_SIZE = " << SHMEM_BLOCK_SIZE << std::endl;

        for (int iteration = 0; iteration < BENCHMARK_WARMUP_ITERATIONS; ++iteration) {
            check_curand(
                curandGenerateUniform(random_generator, d_A, a_elements),
                "curandGenerateUniform(A)");
            check_curand(
                curandGenerateUniform(random_generator, d_B, b_elements),
                "curandGenerateUniform(B)");
            launch_flush_l2_cache<L2_FLUSH_BLOCK_SIZE>(
                d_l2_flush, flush_elements, iteration + 1);
            launch_matmul_shmem<SHMEM_BLOCK_SIZE>(
                d_A, d_B, d_C, MATRIX_M, MATRIX_N, MATRIX_K);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int iteration = 0; iteration < BENCHMARK_MEASURED_ITERATIONS; ++iteration) {
            check_curand(
                curandGenerateUniform(random_generator, d_A, a_elements),
                "curandGenerateUniform(A)");
            check_curand(
                curandGenerateUniform(random_generator, d_B, b_elements),
                "curandGenerateUniform(B)");
            launch_flush_l2_cache<L2_FLUSH_BLOCK_SIZE>(
                d_l2_flush, flush_elements, iteration + 1);

            if (iteration == BENCHMARK_PROFILED_ITERATION) {
                nvtxRangePushA("matmul_shmem iteration 23");
            }
            CUDA_CHECK(cudaEventRecord(start));
            launch_matmul_shmem<SHMEM_BLOCK_SIZE>(
                d_A, d_B, d_C, MATRIX_M, MATRIX_N, MATRIX_K);
            CUDA_CHECK(cudaEventRecord(stop));
            if (iteration == BENCHMARK_PROFILED_ITERATION) {
                nvtxRangePop();
            }

            CUDA_CHECK(cudaEventSynchronize(stop));
            float elapsed_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
            elapsed_times.push_back(elapsed_ms);
        }
        CUDA_CHECK(cudaGetLastError());

        print_performance("matmul_shmem", elapsed_times, MATRIX_M, MATRIX_N, MATRIX_K);
        success &= check_result(
            check_mode,
            cublas_handle,
            d_A,
            d_B,
            d_C,
            d_reference,
            MATRIX_M,
            MATRIX_N,
            MATRIX_K);
    }

    // ------------------------ matmul_thread_tiling ------------------------
    
    // 2x2 thread tiling without shared memory. Each thread computes four
    // output elements and reuses two A and two B values from registers.
    {
        constexpr int THREAD_TILING_TILE_SIZE = 16;
        constexpr int THREAD_TILING_THREAD_TILE_SIZE = 2;

        std::vector<float> elapsed_times;
        elapsed_times.reserve(BENCHMARK_MEASURED_ITERATIONS);
        check_curand(
            curandSetPseudoRandomGeneratorSeed(
                random_generator,
                INPUT_RANDOM_SEED),
            "curandSetPseudoRandomGeneratorSeed");

        std::cout << "\nRunning matmul_thread_tiling" << std::endl;
        std::cout << "Template parameters:" << std::endl;
        std::cout << "  TILE_SIZE = "
                  << THREAD_TILING_TILE_SIZE << std::endl;
        std::cout << "  THREAD_TILE_SIZE = "
                  << THREAD_TILING_THREAD_TILE_SIZE << std::endl;

        for (int iteration = 0; iteration < BENCHMARK_WARMUP_ITERATIONS; ++iteration) {
            check_curand(
                curandGenerateUniform(random_generator, d_A, a_elements),
                "curandGenerateUniform(A)");
            check_curand(
                curandGenerateUniform(random_generator, d_B, b_elements),
                "curandGenerateUniform(B)");
            launch_flush_l2_cache<L2_FLUSH_BLOCK_SIZE>(
                d_l2_flush, flush_elements, iteration + 1);
            launch_matmul_thread_tiling<
                THREAD_TILING_TILE_SIZE,
                THREAD_TILING_THREAD_TILE_SIZE>(
                    d_A, d_B, d_C, MATRIX_M, MATRIX_N, MATRIX_K);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int iteration = 0; iteration < BENCHMARK_MEASURED_ITERATIONS; ++iteration) {
            check_curand(
                curandGenerateUniform(random_generator, d_A, a_elements),
                "curandGenerateUniform(A)");
            check_curand(
                curandGenerateUniform(random_generator, d_B, b_elements),
                "curandGenerateUniform(B)");
            launch_flush_l2_cache<L2_FLUSH_BLOCK_SIZE>(
                d_l2_flush, flush_elements, iteration + 1);

            if (iteration == BENCHMARK_PROFILED_ITERATION) {
                nvtxRangePushA("matmul_thread_tiling iteration 23");
            }
            CUDA_CHECK(cudaEventRecord(start));
            launch_matmul_thread_tiling<
                THREAD_TILING_TILE_SIZE,
                THREAD_TILING_THREAD_TILE_SIZE>(
                    d_A, d_B, d_C, MATRIX_M, MATRIX_N, MATRIX_K);
            CUDA_CHECK(cudaEventRecord(stop));
            if (iteration == BENCHMARK_PROFILED_ITERATION) {
                nvtxRangePop();
            }

            CUDA_CHECK(cudaEventSynchronize(stop));
            float elapsed_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
            elapsed_times.push_back(elapsed_ms);
        }
        CUDA_CHECK(cudaGetLastError());

        print_performance("matmul_thread_tiling", elapsed_times, MATRIX_M, MATRIX_N, MATRIX_K);
        success &= check_result(
            check_mode,
            cublas_handle,
            d_A,
            d_B,
            d_C,
            d_reference,
            MATRIX_M,
            MATRIX_N,
            MATRIX_K);
    }

    // ------------------------ matmul_thread_tiling_tune ------------------------

    // The same thread tiling kernel, but the parameters can be tuned.
    {
        constexpr int THREAD_TILING_TUNE_BLOCK_TILE_M = 32;
        constexpr int THREAD_TILING_TUNE_BLOCK_TILE_N = 32;
        constexpr int THREAD_TILING_TUNE_THREAD_TILE_M = 2;
        constexpr int THREAD_TILING_TUNE_THREAD_TILE_N = 2;

        std::vector<float> elapsed_times;
        elapsed_times.reserve(BENCHMARK_MEASURED_ITERATIONS);
        check_curand(
            curandSetPseudoRandomGeneratorSeed(
                random_generator,
                INPUT_RANDOM_SEED),
            "curandSetPseudoRandomGeneratorSeed");

        std::cout << "\nRunning matmul_thread_tiling_tune" << std::endl;
        std::cout << "Template parameters:" << std::endl;
        std::cout << "  BLOCK_TILE_M = "
                  << THREAD_TILING_TUNE_BLOCK_TILE_M << std::endl;
        std::cout << "  BLOCK_TILE_N = "
                  << THREAD_TILING_TUNE_BLOCK_TILE_N << std::endl;
        std::cout << "  THREAD_TILE_M = "
                  << THREAD_TILING_TUNE_THREAD_TILE_M << std::endl;
        std::cout << "  THREAD_TILE_N = "
                  << THREAD_TILING_TUNE_THREAD_TILE_N << std::endl;

        for (int iteration = 0; iteration < BENCHMARK_WARMUP_ITERATIONS; ++iteration) {
            check_curand(
                curandGenerateUniform(random_generator, d_A, a_elements),
                "curandGenerateUniform(A)");
            check_curand(
                curandGenerateUniform(random_generator, d_B, b_elements),
                "curandGenerateUniform(B)");
            launch_flush_l2_cache<L2_FLUSH_BLOCK_SIZE>(
                d_l2_flush, flush_elements, iteration + 1);
            launch_matmul_thread_tiling_tune<
                THREAD_TILING_TUNE_BLOCK_TILE_M,
                THREAD_TILING_TUNE_BLOCK_TILE_N,
                THREAD_TILING_TUNE_THREAD_TILE_M,
                THREAD_TILING_TUNE_THREAD_TILE_N>(
                    d_A, d_B, d_C, MATRIX_M, MATRIX_N, MATRIX_K);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int iteration = 0; iteration < BENCHMARK_MEASURED_ITERATIONS; ++iteration) {
            check_curand(
                curandGenerateUniform(random_generator, d_A, a_elements),
                "curandGenerateUniform(A)");
            check_curand(
                curandGenerateUniform(random_generator, d_B, b_elements),
                "curandGenerateUniform(B)");
            launch_flush_l2_cache<L2_FLUSH_BLOCK_SIZE>(
                d_l2_flush, flush_elements, iteration + 1);

            if (iteration == BENCHMARK_PROFILED_ITERATION) {
                nvtxRangePushA("matmul_thread_tiling_tune iteration 23");
            }
            CUDA_CHECK(cudaEventRecord(start));
            launch_matmul_thread_tiling_tune<
                THREAD_TILING_TUNE_BLOCK_TILE_M,
                THREAD_TILING_TUNE_BLOCK_TILE_N,
                THREAD_TILING_TUNE_THREAD_TILE_M,
                THREAD_TILING_TUNE_THREAD_TILE_N>(
                    d_A, d_B, d_C, MATRIX_M, MATRIX_N, MATRIX_K);
            CUDA_CHECK(cudaEventRecord(stop));
            if (iteration == BENCHMARK_PROFILED_ITERATION) {
                nvtxRangePop();
            }

            CUDA_CHECK(cudaEventSynchronize(stop));
            float elapsed_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
            elapsed_times.push_back(elapsed_ms);
        }
        CUDA_CHECK(cudaGetLastError());

        print_performance(
            "matmul_thread_tiling_tune", elapsed_times, MATRIX_M, MATRIX_N, MATRIX_K);
        success &= check_result(
            check_mode,
            cublas_handle,
            d_A,
            d_B,
            d_C,
            d_reference,
            MATRIX_M,
            MATRIX_N,
            MATRIX_K);
    }

    // ------------------------ matmul_shmem_thread_tiling ------------------------

    // Shared-memory loading combined with 2x2 thread tiling.
    {
        constexpr int SHMEM_THREAD_TILING_TILE_SIZE = 16;
        constexpr int SHMEM_THREAD_TILING_THREAD_TILE_SIZE = 2;

        std::vector<float> elapsed_times;
        elapsed_times.reserve(BENCHMARK_MEASURED_ITERATIONS);
        check_curand(
            curandSetPseudoRandomGeneratorSeed(
                random_generator,
                INPUT_RANDOM_SEED),
            "curandSetPseudoRandomGeneratorSeed");

        std::cout << "\nRunning matmul_shmem_thread_tiling" << std::endl;
        std::cout << "Template parameters:" << std::endl;
        std::cout << "  TILE_SIZE = "
                  << SHMEM_THREAD_TILING_TILE_SIZE << std::endl;
        std::cout << "  THREAD_TILE_SIZE = "
                  << SHMEM_THREAD_TILING_THREAD_TILE_SIZE << std::endl;

        for (int iteration = 0; iteration < BENCHMARK_WARMUP_ITERATIONS; ++iteration) {
            check_curand(
                curandGenerateUniform(random_generator, d_A, a_elements),
                "curandGenerateUniform(A)");
            check_curand(
                curandGenerateUniform(random_generator, d_B, b_elements),
                "curandGenerateUniform(B)");
            launch_flush_l2_cache<L2_FLUSH_BLOCK_SIZE>(
                d_l2_flush, flush_elements, iteration + 1);
            launch_matmul_shmem_thread_tiling<
                SHMEM_THREAD_TILING_TILE_SIZE,
                SHMEM_THREAD_TILING_THREAD_TILE_SIZE>(
                    d_A, d_B, d_C, MATRIX_M, MATRIX_N, MATRIX_K);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int iteration = 0; iteration < BENCHMARK_MEASURED_ITERATIONS; ++iteration) {
            check_curand(
                curandGenerateUniform(random_generator, d_A, a_elements),
                "curandGenerateUniform(A)");
            check_curand(
                curandGenerateUniform(random_generator, d_B, b_elements),
                "curandGenerateUniform(B)");
            launch_flush_l2_cache<L2_FLUSH_BLOCK_SIZE>(
                d_l2_flush, flush_elements, iteration + 1);

            if (iteration == BENCHMARK_PROFILED_ITERATION) {
                nvtxRangePushA("matmul_shmem_thread_tiling iteration 23");
            }
            CUDA_CHECK(cudaEventRecord(start));
            launch_matmul_shmem_thread_tiling<
                SHMEM_THREAD_TILING_TILE_SIZE,
                SHMEM_THREAD_TILING_THREAD_TILE_SIZE>(
                    d_A, d_B, d_C, MATRIX_M, MATRIX_N, MATRIX_K);
            CUDA_CHECK(cudaEventRecord(stop));
            if (iteration == BENCHMARK_PROFILED_ITERATION) {
                nvtxRangePop();
            }

            CUDA_CHECK(cudaEventSynchronize(stop));
            float elapsed_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
            elapsed_times.push_back(elapsed_ms);
        }
        CUDA_CHECK(cudaGetLastError());

        print_performance(
            "matmul_shmem_thread_tiling", elapsed_times, MATRIX_M, MATRIX_N, MATRIX_K);
        success &= check_result(
            check_mode,
            cublas_handle,
            d_A,
            d_B,
            d_C,
            d_reference,
            MATRIX_M,
            MATRIX_N,
            MATRIX_K);
    }

    // ------------------------ matmul_shmem_thread_tiling_tune ------------------------

    // Independently tunable shared-memory, block and thread tiles.
    {
        constexpr int SHMEM_THREAD_TILING_TUNE_BLOCK_TILE_M = 32;
        constexpr int SHMEM_THREAD_TILING_TUNE_BLOCK_TILE_N = 32;
        constexpr int SHMEM_THREAD_TILING_TUNE_BLOCK_TILE_K = 16;
        constexpr int SHMEM_THREAD_TILING_TUNE_THREAD_TILE_M = 2;
        constexpr int SHMEM_THREAD_TILING_TUNE_THREAD_TILE_N = 2;

        std::vector<float> elapsed_times;
        elapsed_times.reserve(BENCHMARK_MEASURED_ITERATIONS);
        check_curand(
            curandSetPseudoRandomGeneratorSeed(
                random_generator,
                INPUT_RANDOM_SEED),
            "curandSetPseudoRandomGeneratorSeed");

        std::cout << "\nRunning matmul_shmem_thread_tiling_tune"
                  << std::endl;
        std::cout << "Template parameters:" << std::endl;
        std::cout << "  BLOCK_TILE_M = "
                  << SHMEM_THREAD_TILING_TUNE_BLOCK_TILE_M << std::endl;
        std::cout << "  BLOCK_TILE_N = "
                  << SHMEM_THREAD_TILING_TUNE_BLOCK_TILE_N << std::endl;
        std::cout << "  BLOCK_TILE_K = "
                  << SHMEM_THREAD_TILING_TUNE_BLOCK_TILE_K << std::endl;
        std::cout << "  THREAD_TILE_M = "
                  << SHMEM_THREAD_TILING_TUNE_THREAD_TILE_M << std::endl;
        std::cout << "  THREAD_TILE_N = "
                  << SHMEM_THREAD_TILING_TUNE_THREAD_TILE_N << std::endl;

        for (int iteration = 0;
             iteration < BENCHMARK_WARMUP_ITERATIONS;
             ++iteration) {
            check_curand(
                curandGenerateUniform(random_generator, d_A, a_elements),
                "curandGenerateUniform(A)");
            check_curand(
                curandGenerateUniform(random_generator, d_B, b_elements),
                "curandGenerateUniform(B)");
            launch_flush_l2_cache<L2_FLUSH_BLOCK_SIZE>(
                d_l2_flush, flush_elements, iteration + 1);
            launch_matmul_shmem_thread_tiling_tune<
                SHMEM_THREAD_TILING_TUNE_BLOCK_TILE_M,
                SHMEM_THREAD_TILING_TUNE_BLOCK_TILE_N,
                SHMEM_THREAD_TILING_TUNE_BLOCK_TILE_K,
                SHMEM_THREAD_TILING_TUNE_THREAD_TILE_M,
                SHMEM_THREAD_TILING_TUNE_THREAD_TILE_N>(
                    d_A, d_B, d_C, MATRIX_M, MATRIX_N, MATRIX_K);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int iteration = 0;
             iteration < BENCHMARK_MEASURED_ITERATIONS;
             ++iteration) {
            check_curand(
                curandGenerateUniform(random_generator, d_A, a_elements),
                "curandGenerateUniform(A)");
            check_curand(
                curandGenerateUniform(random_generator, d_B, b_elements),
                "curandGenerateUniform(B)");
            launch_flush_l2_cache<L2_FLUSH_BLOCK_SIZE>(
                d_l2_flush, flush_elements, iteration + 1);

            if (iteration == BENCHMARK_PROFILED_ITERATION) {
                nvtxRangePushA(
                    "matmul_shmem_thread_tiling_tune iteration 23");
            }
            CUDA_CHECK(cudaEventRecord(start));
            launch_matmul_shmem_thread_tiling_tune<
                SHMEM_THREAD_TILING_TUNE_BLOCK_TILE_M,
                SHMEM_THREAD_TILING_TUNE_BLOCK_TILE_N,
                SHMEM_THREAD_TILING_TUNE_BLOCK_TILE_K,
                SHMEM_THREAD_TILING_TUNE_THREAD_TILE_M,
                SHMEM_THREAD_TILING_TUNE_THREAD_TILE_N>(
                    d_A, d_B, d_C, MATRIX_M, MATRIX_N, MATRIX_K);
            CUDA_CHECK(cudaEventRecord(stop));
            if (iteration == BENCHMARK_PROFILED_ITERATION) {
                nvtxRangePop();
            }

            CUDA_CHECK(cudaEventSynchronize(stop));
            float elapsed_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
            elapsed_times.push_back(elapsed_ms);
        }
        CUDA_CHECK(cudaGetLastError());

        print_performance(
            "matmul_shmem_thread_tiling_tune",
            elapsed_times,
            MATRIX_M,
            MATRIX_N,
            MATRIX_K);
        success &= check_result(
            check_mode,
            cublas_handle,
            d_A,
            d_B,
            d_C,
            d_reference,
            MATRIX_M,
            MATRIX_N,
            MATRIX_K);
    }

    // ------------------------ cuBLAS SGEMM ------------------------

    if (check_mode == CheckMode::Cublas) {
        std::vector<float> elapsed_times;
        elapsed_times.reserve(BENCHMARK_MEASURED_ITERATIONS);
        check_curand(
            curandSetPseudoRandomGeneratorSeed(
                random_generator,
                INPUT_RANDOM_SEED),
            "curandSetPseudoRandomGeneratorSeed");

        std::cout << "\nRunning cuBLAS SGEMM" << std::endl;
        std::cout << "Template parameters: none" << std::endl;

        for (int iteration = 0;
             iteration < BENCHMARK_WARMUP_ITERATIONS;
             ++iteration) {
            check_curand(
                curandGenerateUniform(random_generator, d_A, a_elements),
                "curandGenerateUniform(A)");
            check_curand(
                curandGenerateUniform(random_generator, d_B, b_elements),
                "curandGenerateUniform(B)");
            launch_flush_l2_cache<L2_FLUSH_BLOCK_SIZE>(
                d_l2_flush, flush_elements, iteration + 1);
            launch_cublas_matmul(
                cublas_handle,
                d_A,
                d_B,
                d_reference,
                MATRIX_M,
                MATRIX_N,
                MATRIX_K);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int iteration = 0;
             iteration < BENCHMARK_MEASURED_ITERATIONS;
             ++iteration) {
            check_curand(
                curandGenerateUniform(random_generator, d_A, a_elements),
                "curandGenerateUniform(A)");
            check_curand(
                curandGenerateUniform(random_generator, d_B, b_elements),
                "curandGenerateUniform(B)");
            launch_flush_l2_cache<L2_FLUSH_BLOCK_SIZE>(
                d_l2_flush, flush_elements, iteration + 1);

            if (iteration == BENCHMARK_PROFILED_ITERATION) {
                nvtxRangePushA("cuBLAS SGEMM iteration 23");
            }
            CUDA_CHECK(cudaEventRecord(start));
            launch_cublas_matmul(
                cublas_handle,
                d_A,
                d_B,
                d_reference,
                MATRIX_M,
                MATRIX_N,
                MATRIX_K);
            CUDA_CHECK(cudaEventRecord(stop));
            if (iteration == BENCHMARK_PROFILED_ITERATION) {
                nvtxRangePop();
            }

            CUDA_CHECK(cudaEventSynchronize(stop));
            float elapsed_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
            elapsed_times.push_back(elapsed_ms);
        }

        print_performance(
            "cuBLAS SGEMM",
            elapsed_times,
            MATRIX_M,
            MATRIX_N,
            MATRIX_K);
    }

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

    if (success) {
        return EXIT_SUCCESS;
    } else {
        return EXIT_FAILURE;
    }
}
