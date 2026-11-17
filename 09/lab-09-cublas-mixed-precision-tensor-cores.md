Lab 09
===

FP16 numerics
---
The sample `09/fp16-conversion.cu` converts a float `x` to fp16, with nearest rounding, then converts back to float. Validate the step-size around 0, 1 and 50000.

CuBLAS gemm interface
---

Revisit the `09/cublas-gemm-ex.cu` file, check the GEMM parameterization to compute C = A @ B, all matrices are in row major. BLAS libaries, however expect matrices A, B, C to be in colum-major.

https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm
Computes C = (alpha * A) @ B + (beta * C)
All matrices are expected to be column-major:
*  A = MxK
*  B = KxN
*  C = MxN

But we want to compute C = A @ B:
A, B, C are row-major:
*  A = MxK
*  B = KxN
*  C = MxN

1. Observation row-major A = column-major A.T
2. Since C = A @ B <=> C.T = B.T @ A.T

We get the result (C.T) as column major if we pass B.T as A and A.T as B, thus the buffer C will contain C in row major.

Thus, the proper cuBLAS parameterization:
*  A operand will be B, shape is NxK,   leading dim is N
*  B operand will be A, shape is KxM,   leading dim is K
*  C operand stays C, shape is NxM,     leading dim is N
 --> M and N dim are flipped

```cuda
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
```

What about the other combinations? E.g. A is row major, C is colum major, we want the result in column major?

See: https://leimao.github.io/blog/cuBLAS-Transpose-Column-Major-Relationship/

CuBLAS mixed precision performance
---
Tune the computation type parameters of GemmEx in the code.
https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex

One should give:
* operand types
* output types
* result type
* accumulation type

Example: A, B: fp16; C: fp32. Compute type can be `CUBLAS_COMPUTE_32F` and `CUBLAS_COMPUTE_32F_PEDANTIC`.