Matrix multiplication
=====================

Build the matmul.cu with ninja: `ninja`. 

See install instructions for ninja in `01` or build manually: `nvcc -O3 -lineinfo -I../include matmul.cu -o build/matmul -lnvToolsExt -lcurand -lcublas`.

Use the fast cuBLAS result check with `./build/matmul --check cublas` (this is the default), or the original CPU reference with `./build/matmul --check cpu`.

If we execute the naive kernel (`matmul_naive`) on 3060, we see that the perf is 788 GFLOPS/s. The peak perf of 3060 is 9.462 TFLOPS/s (without boost), thus we are under 10% of the peak performance.

Important: why are we interested in GFLOPS/s instead of GB/s in the matmul experiment?

Create an ncu report to see why we get only 10% of the peak perf: `ncu --set full --nvtx --nvtx-include "regex:matmul_naive iteration 23/" -o build/matmul-report -f ./build/matmul`.

Rooflines tell us that perf depends on the matrix size. Try to adjust the matrix size, M=N=K=64 to multiple thousands.

We introduce shared memory loading in the code (`matmul_shmem(TILE_SIZE)`). Why the perf improvement is minimal (although significant)? The template parameter `TILE_SIZE` is the square root of the block size.

* Try to increase the thread-level reuse without shared memory and check the perf. Kernel: `matmul_thread_tiling(TILE_SIZE, THREAD_TILE_SIZE)`.

* Tune the block tile size parameters. Kernel: `matmul_thread_tiling_tune`. Template parameters:
    * `BLOCK_TILE_M`,
    * `BLOCK_TILE_N`,
    * `THREAD_TILE_M`,
    * `THREAD_TILE_N`.

* Try to use both shared memory prefetch and thread-level reuse. Kernel: `matmul_shmem_thread_tiling(TILE_SIZE, THREAD_TILE_SIZE)`.

* Tune the block tile size parameters. Kernel: `matmul_shmem_thread_tiling_tune`. Template parameters:
    * `BLOCK_TILE_M`, 
    * `BLOCK_TILE_N`, 
    * `BLOCK_TILE_K`, 
    * `THREAD_TILE_M`, 
    * `THREAD_TILE_N`.
    
    The number of threads is determined by considering all the template arguments. How?
