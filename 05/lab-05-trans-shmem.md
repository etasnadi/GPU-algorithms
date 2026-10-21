# Lab 05 – Matrix Transpose with Shared Memory

## Goals

- Study `transpose-shmem.cu` in detail.
- Identify the problematic memory-access patterns and understand how they are
  fixed.
- Execute `transpose-shmem-timing` and explain the performance differences
  between its kernels.
- Profile the selected kernel launches with NVIDIA Nsight Compute.

## Nsight Compute profiling

The `transpose-shmem-timing` application places the 23rd invocation of each
kernel inside an NVTX range. Profile these ranges with:

```bash
ncu --set full --nvtx \
    --nvtx-include "regex:.* iteration 23/" \
    -o build/transpose-shmem-timing-report \
    -f ./build/transpose-shmem-timing
```

Open the resulting report in the graphical `ncu-ui` application.

## Interpreting the report

### GPU Speed of Light Throughput

Open **Details → GPU Speed of Light Throughput** and inspect:

- **Compute (SM) Throughput**
- **L1/TEX Throughput**
- **L2 Throughput**
- **DRAM Throughput**

The memory-throughput values represent the actual data transferred by the
memory subsystem, not only the bytes useful to the application.

### Kernel duration from elapsed cycles

Given the following values:

```text
SM Frequency   = 1.32 GHz
Elapsed Cycles = 1,155,442 cycles
```

The duration of one cycle is approximately:

```text
1 / 1.32 GHz = 0.75 ns/cycle
```

Therefore, the kernel duration is approximately:

```text
0.75 ns/cycle × 1,155,442 cycles
    = 866,581.5 ns
    = 866.58 µs
```

### Issue-slot utilization

`Issue Slots Busy = 6.51%` means that the schedulers used only 6.51% of their
instruction-issue capacity. An SM has four warp schedulers, so the average
issue rate per SM is:

```text
4 × 0.0651 = 0.2604 warp-instructions/cycle
```

The **Summary** tab reports a block size of `(16, 16, 1)`, which corresponds to
256 threads or 8 warps per block. The **Occupancy** section reports 35.56 active
warps per SM. Assuming an even distribution, an active warp issues an
instruction approximately once every:

```text
35.56 / 0.2604 = 136.56 cycles
```

The reported value of `Warp Cycles Per Executed Instruction = 135.59` is close
to this estimate.

### Warp stall statistics

The stall reasons approximately sum to the reported warp cycles per executed
instruction:

```text
55.76 + 52.26 + 10.18 + 9.52 + 2.46 + 2.43 + 1.00
    + 0.50 + 0.45 + 0.33 + 0.14 + 0.09 + 0.05
    = 135.17 cycles
```

This is close to the reported `135.59 cycles`.

Important stall reasons:

- **Stall Drain (`55.76 cycles`)**: the warp is waiting for pending memory
  operations to complete before it can finish and leave the SM.
- **Stall Long Scoreboard**: the warp is waiting for the result of a memory
  operation that uses the L1/TEX path, such as a global-memory access.
- **Stall Short Scoreboard**: the warp is waiting for a dependency on a MIO
  operation that does not use the L1/TEX path, such as a shared-memory load.

A scoreboard tracks dependencies created by asynchronous or long-latency
instructions. Independent instructions may continue executing, but an
instruction that consumes a pending result cannot issue until its scoreboard
dependency is cleared.

## Why is the shared-memory kernel faster?

The naive and shared-memory kernels transfer the same number of sectors to and
from DRAM. However, between L1/TEX and L2, the naive kernel generates roughly:

- 6× as many write requests;
- 3.5× as many affected sectors.

Measured affected-sector counts:

```text
naive: 6,420,562 sectors
shmem: 1,048,576 sectors
```

Consequently, the naive kernel shows much higher L1/TEX and L2 throughput. This
does not represent useful additional work: the memory subsystem is processing
extra transactions caused by the non-optimal access pattern. The shared-memory
kernel reorganizes the accesses so that global-memory writes are coalesced,
substantially reducing this overhead.
