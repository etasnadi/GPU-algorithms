# Lab 04

## Goal

Consider the `transpose-shmem.cu` in detail. Reason about the problematic parts, and how it is fixed.

Execute the `transpose-shmem-timing` code and reason about the perf differences.

The `transpose-shmem-timing` application puts the 23rd kernel call into an nvtx range. Profile the nvtx range with ncu and profile it with the graphical `ncu-ui` application. Profiling command: `ncu --set detailed --nvtx --nvtx-include "regex:.* iteration 23/" -o build/transpose-report -f ./build/transpose-shmem-timing`.