# Lab 01

## Goal

* Implement the copy kernel `copy-kernel.cu` that copies from device buffer A to device buffer B.

* Intentionally overindex the inpur buffer in `copy-kernel-error.cu`, then check the problems with `compute-sanitizer`. We should see an error.

* Add proper error handling to `copy-kernel` (`copy-kernel-error-handling.cu`). Use the `CUDA_CHECK` macro provided by the Nvidia documentation (already put in `common.cuh`).

* Add wall-clock time measurement with cuda events (`copy-kernel-timing.cu`).
Proper time measurement:
    * Warmup: execute the kernel several times ensuring that the GPU reached the necessary clock speed.
    * L2 flush: write data to the global memory of size at least the L2.
    * Compute the median execution time, or compute the mean ignoring the best and worst `k`.

* Profile the kernel with ncu:
```bash
nvcc \
    -O3 -lineinfo \
    copy-kernel-timing.cu \
    -o copy-kernel-timing
```

```bash
ncu \
    --set detailed \
    --launch-skip 1 \
    --launch-count 10 \
    --export copy-kernel-timing \
    --force-overwrite \
    ./copy-kernel-timing
```

* Add nvtx range to the 23rd iteration and profile only this kernel call with ncu (`copy-kernel-nvtx.cu`).
```bash
nvcc \
    -O3 -lineinfo \
    copy-kernel-nvtx.cu \
    -lnvToolsExt \
    -o copy-kernel-nvtx
```

```bash
ncu \
    --set detailed \
    --nvtx \
    --nvtx-include "copy-kernel iteration 23/" \
    --launch-count 1 \
    --export 02/copy-kernel-iteration-23 \
    --force-overwrite \
    ./copy-kernel-nvtx
```