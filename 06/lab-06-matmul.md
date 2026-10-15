Matrix multiplication
=====================

Build the matmul.cu with ninja: `ninja`. 

See install instructions for ninja in `01` or build manually: `nvcc -O3 -lineinfo -I../include matmul.cu -o build/matmul -lnvToolsExt -lcurand`.

If we execute the code on 3060, we see that the perf is 788 GFLOPS/s. The peak perf of 3060 is 9.462 TFLOPS/s (without boos), thus we are under 10% of the peak performance.

Important: why are we interested in GFLOPS/s instead of GB/s in the matmul experiment?

Create an ncu report to see why we get only 10% of the peak perf: `ncu --set full --nvtx --nvtx-include 'regex:matmul_naive iteration 23/' -o build/matmul-report -f ./build/matmul`.