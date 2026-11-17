#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <nvtx3/nvToolsExt.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "common.cuh"

void check_cublas(cublasStatus_t status, const char* operation)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS error in " << operation
                  << ": status " << status << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void check_curand(curandStatus_t status, const char* operation)
{
    if (status != CURAND_STATUS_SUCCESS) {
        std::cerr << "cuRAND error in " << operation
                  << ": status " << status << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

__global__ void convert_float_to_half(
    const float* input,
    __half* output,
    std::size_t element_count)
{
    for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < element_count;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        output[index] = __float2half_rn(input[index]);
    }
}

template<int BLOCK_SIZE>
void launch_convert_float_to_half(
    const float* input,
    __half* output,
    std::size_t element_count)
{
    const int block_count = static_cast<int>(
        (element_count + BLOCK_SIZE - 1) / BLOCK_SIZE);
    convert_float_to_half<<<block_count, BLOCK_SIZE>>>(
        input, output, element_count);
}

void launch_cublas_sgemm(
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

    // GEMM interface
    // https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm
    // Computes C = (alpha * A) @ B + (beta * C)
    // All matrices are expected to be column-major
    // A = MxK
    // B = KxN
    // C = MxN

    // Current problem:
    // A, B, C are row-major
    // A = MxK
    // B = KxN
    // C = MxN
    
    // Observation row-major(A) = column-major(A.T)
    // Since C = A @ B <=> C.T = B.T @ A.T
    // We get C in column major if we pass B.T as A and A.T as B.

    // Thus,
    // A operand will be B, shape is NxK,   leading dim is N
    // B operand will be A, shape is KxM,   leading dim is K
    // C operand stays C, shape is NxM,     leading dim is N
    // --> M and N dim are flipped
    check_cublas(
        cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,       // transa, transb: OP_N means non-transposed
            MATRIX_N, MATRIX_M, MATRIX_K,   // m, n, k sizes
            &alpha,
            B, MATRIX_N,                    // A, lda
            A, MATRIX_K,                    // B, ldb
            &beta,
            C, MATRIX_N),                   // C, ldc
        "cublasSgemm");
}

void launch_cublas_dgemm(
    cublasHandle_t handle,
    const double* A,
    const double* B,
    double* C,
    int MATRIX_M,
    int MATRIX_N,
    int MATRIX_K)
{
    const double alpha = 1.0;
    const double beta = 0.0;

    check_cublas(
        cublasDgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            MATRIX_N, MATRIX_M, MATRIX_K,
            &alpha,
            B, MATRIX_N,
            A, MATRIX_K,
            &beta,
            C, MATRIX_N),
        "cublasDgemm");
}

void launch_cublas_gemm_ex(
    cublasHandle_t handle,
    const __half* A,
    const __half* B,
    float* C,
    int MATRIX_M,
    int MATRIX_N,
    int MATRIX_K)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    check_cublas(
        cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            MATRIX_N, MATRIX_M, MATRIX_K,
            &alpha, 
            B, CUDA_R_16F, MATRIX_N,        // A storage type
            A, CUDA_R_16F, MATRIX_K,        // B storage type
            &beta,
            C, CUDA_R_32F, MATRIX_N,        // C storage type
            CUBLAS_COMPUTE_32F,             // Computation type
            CUBLAS_GEMM_DEFAULT),
        "cublasGemmEx");
}

int main()
{
    constexpr int MATRIX_M = 2048;
    constexpr int MATRIX_N = 2048;
    constexpr int MATRIX_K = 2048;
    constexpr int WARMUP_ITERATIONS = 10;
    constexpr int MEASURED_ITERATIONS = 100;
    constexpr int PROFILED_ITERATION = 22;
    constexpr int L2_FLUSH_BLOCK_SIZE = 256;
    constexpr int CONVERSION_BLOCK_SIZE = 256;
    constexpr unsigned long long RANDOM_SEED = 1;

    const std::size_t a_elements =
        static_cast<std::size_t>(MATRIX_M) * MATRIX_K;
    const std::size_t b_elements =
        static_cast<std::size_t>(MATRIX_K) * MATRIX_N;
    const std::size_t c_elements =
        static_cast<std::size_t>(MATRIX_M) * MATRIX_N;

    int device = 0;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    const std::size_t l2_bytes = properties.l2CacheSize;
    const std::size_t flush_bytes = 2 * l2_bytes;
    const std::size_t flush_elements =
        (flush_bytes + sizeof(std::uint32_t) - 1) /
        sizeof(std::uint32_t);

    float* d_A_float = nullptr;
    float* d_B_float = nullptr;
    float* d_C_float = nullptr;
    double* d_A_double = nullptr;
    double* d_B_double = nullptr;
    double* d_C_double = nullptr;
    __half* d_A_half = nullptr;
    __half* d_B_half = nullptr;
    std::uint32_t* d_l2_flush = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A_float, a_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B_float, b_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_float, c_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A_double, a_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B_double, b_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C_double, c_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_A_half, a_elements * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_B_half, b_elements * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(
        &d_l2_flush,
        flush_elements * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMemset(
        d_l2_flush,
        0,
        flush_elements * sizeof(std::uint32_t)));

    cublasHandle_t cublas_handle;
    check_cublas(cublasCreate(&cublas_handle), "cublasCreate");

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
    std::cout << "L2 cache: "
              << l2_bytes / 1024.0 / 1024.0 << " MiB" << std::endl;
    std::cout << "L2 flush buffer: "
              << flush_bytes / 1024.0 / 1024.0 << " MiB" << std::endl;
    std::cout << "Warmup iterations: " << WARMUP_ITERATIONS << std::endl;
    std::cout << "Measured iterations: " << MEASURED_ITERATIONS << std::endl;

    std::vector<float> elapsed_times;
    elapsed_times.reserve(MEASURED_ITERATIONS);

    // Measure SGEMM
    // FP32 input, FP32 multiplication, FP32 accumulation, FP32 output.
    check_cublas(
        cublasSetMathMode(cublas_handle, CUBLAS_PEDANTIC_MATH),
        "cublasSetMathMode");
    check_curand(
        curandSetPseudoRandomGeneratorSeed(random_generator, RANDOM_SEED),
        "curandSetPseudoRandomGeneratorSeed");

    for (int iteration = 0; iteration < WARMUP_ITERATIONS; ++iteration) {
        check_curand(
            curandGenerateUniform(random_generator, d_A_float, a_elements),
            "curandGenerateUniform(A_float)");
        check_curand(
            curandGenerateUniform(random_generator, d_B_float, b_elements),
            "curandGenerateUniform(B_float)");
        launch_flush_l2_cache<L2_FLUSH_BLOCK_SIZE>(
            d_l2_flush, flush_elements, iteration + 1);
        launch_cublas_sgemm(
            cublas_handle,
            d_A_float,
            d_B_float,
            d_C_float,
            MATRIX_M,
            MATRIX_N,
            MATRIX_K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int iteration = 0; iteration < MEASURED_ITERATIONS; ++iteration) {
        check_curand(
            curandGenerateUniform(random_generator, d_A_float, a_elements),
            "curandGenerateUniform(A_float)");
        check_curand(
            curandGenerateUniform(random_generator, d_B_float, b_elements),
            "curandGenerateUniform(B_float)");
        launch_flush_l2_cache<L2_FLUSH_BLOCK_SIZE>(
            d_l2_flush, flush_elements, iteration + 1);

        if (iteration == PROFILED_ITERATION) {
            nvtxRangePushA("cublasSgemm iteration 23");
        }
        CUDA_CHECK(cudaEventRecord(start));
        launch_cublas_sgemm(
            cublas_handle,
            d_A_float,
            d_B_float,
            d_C_float,
            MATRIX_M,
            MATRIX_N,
            MATRIX_K);
        CUDA_CHECK(cudaEventRecord(stop));
        if (iteration == PROFILED_ITERATION) {
            nvtxRangePop();
        }

        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        elapsed_times.push_back(elapsed_ms);
    }
    print_performance(
        "cublasSgemm (FP32)",
        elapsed_times,
        MATRIX_M,
        MATRIX_N,
        MATRIX_K);

    // Measure DGEMM:
    // FP64 input, FP64 multiplication, FP64 accumulation, FP64 output.
    elapsed_times.clear();
    check_curand(
        curandSetPseudoRandomGeneratorSeed(random_generator, RANDOM_SEED),
        "curandSetPseudoRandomGeneratorSeed");

    for (int iteration = 0; iteration < WARMUP_ITERATIONS; ++iteration) {
        check_curand(
            curandGenerateUniformDouble(
                random_generator, d_A_double, a_elements),
            "curandGenerateUniformDouble(A_double)");
        check_curand(
            curandGenerateUniformDouble(
                random_generator, d_B_double, b_elements),
            "curandGenerateUniformDouble(B_double)");
        launch_flush_l2_cache<L2_FLUSH_BLOCK_SIZE>(
            d_l2_flush, flush_elements, iteration + 1);
        launch_cublas_dgemm(
            cublas_handle,
            d_A_double,
            d_B_double,
            d_C_double,
            MATRIX_M,
            MATRIX_N,
            MATRIX_K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int iteration = 0; iteration < MEASURED_ITERATIONS; ++iteration) {
        check_curand(
            curandGenerateUniformDouble(
                random_generator, d_A_double, a_elements),
            "curandGenerateUniformDouble(A_double)");
        check_curand(
            curandGenerateUniformDouble(
                random_generator, d_B_double, b_elements),
            "curandGenerateUniformDouble(B_double)");
        launch_flush_l2_cache<L2_FLUSH_BLOCK_SIZE>(
            d_l2_flush, flush_elements, iteration + 1);

        if (iteration == PROFILED_ITERATION) {
            nvtxRangePushA("cublasDgemm iteration 23");
        }
        CUDA_CHECK(cudaEventRecord(start));
        launch_cublas_dgemm(
            cublas_handle,
            d_A_double,
            d_B_double,
            d_C_double,
            MATRIX_M,
            MATRIX_N,
            MATRIX_K);
        CUDA_CHECK(cudaEventRecord(stop));
        if (iteration == PROFILED_ITERATION) {
            nvtxRangePop();
        }

        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        elapsed_times.push_back(elapsed_ms);
    }
    print_performance(
        "cublasDgemm (FP64)",
        elapsed_times,
        MATRIX_M,
        MATRIX_N,
        MATRIX_K);

    // Measure mixed precision GEMM
    // FP16 input, FP16 multiplication, FP32 accumulation, FP32 output.
    elapsed_times.clear();
    check_cublas(
        cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH),
        "cublasSetMathMode");
    check_curand(
        curandSetPseudoRandomGeneratorSeed(random_generator, RANDOM_SEED),
        "curandSetPseudoRandomGeneratorSeed");

    for (int iteration = 0; iteration < WARMUP_ITERATIONS; ++iteration) {
        check_curand(
            curandGenerateUniform(random_generator, d_A_float, a_elements),
            "curandGenerateUniform(A_half_source)");
        check_curand(
            curandGenerateUniform(random_generator, d_B_float, b_elements),
            "curandGenerateUniform(B_half_source)");
        launch_convert_float_to_half<CONVERSION_BLOCK_SIZE>(
            d_A_float, d_A_half, a_elements);
        launch_convert_float_to_half<CONVERSION_BLOCK_SIZE>(
            d_B_float, d_B_half, b_elements);
        launch_flush_l2_cache<L2_FLUSH_BLOCK_SIZE>(
            d_l2_flush, flush_elements, iteration + 1);
        launch_cublas_gemm_ex(
            cublas_handle,
            d_A_half,
            d_B_half,
            d_C_float,
            MATRIX_M,
            MATRIX_N,
            MATRIX_K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int iteration = 0; iteration < MEASURED_ITERATIONS; ++iteration) {
        check_curand(
            curandGenerateUniform(random_generator, d_A_float, a_elements),
            "curandGenerateUniform(A_half_source)");
        check_curand(
            curandGenerateUniform(random_generator, d_B_float, b_elements),
            "curandGenerateUniform(B_half_source)");
        launch_convert_float_to_half<CONVERSION_BLOCK_SIZE>(
            d_A_float, d_A_half, a_elements);
        launch_convert_float_to_half<CONVERSION_BLOCK_SIZE>(
            d_B_float, d_B_half, b_elements);
        launch_flush_l2_cache<L2_FLUSH_BLOCK_SIZE>(
            d_l2_flush, flush_elements, iteration + 1);

        if (iteration == PROFILED_ITERATION) {
            nvtxRangePushA("cublasGemmEx iteration 23");
        }
        CUDA_CHECK(cudaEventRecord(start));
        launch_cublas_gemm_ex(
            cublas_handle,
            d_A_half,
            d_B_half,
            d_C_float,
            MATRIX_M,
            MATRIX_N,
            MATRIX_K);
        CUDA_CHECK(cudaEventRecord(stop));
        if (iteration == PROFILED_ITERATION) {
            nvtxRangePop();
        }

        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        elapsed_times.push_back(elapsed_ms);
    }
    print_performance(
        "cublasGemmEx (FP16 input, FP32 accumulation)",
        elapsed_times,
        MATRIX_M,
        MATRIX_N,
        MATRIX_K);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    check_curand(
        curandDestroyGenerator(random_generator),
        "curandDestroyGenerator");
    check_cublas(cublasDestroy(cublas_handle), "cublasDestroy");
    CUDA_CHECK(cudaFree(d_l2_flush));
    CUDA_CHECK(cudaFree(d_B_half));
    CUDA_CHECK(cudaFree(d_A_half));
    CUDA_CHECK(cudaFree(d_C_double));
    CUDA_CHECK(cudaFree(d_B_double));
    CUDA_CHECK(cudaFree(d_A_double));
    CUDA_CHECK(cudaFree(d_C_float));
    CUDA_CHECK(cudaFree(d_B_float));
    CUDA_CHECK(cudaFree(d_A_float));

    return 0;
}
