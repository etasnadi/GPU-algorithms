---
marp: true
theme: minimal-white
paginate: true
size: 16:9
footer: GPU Programming for AI · Lecture 5
---

<!-- _paginate: false -->
<!-- _footer: "" -->

# Latency Hiding and Occupancy

## Warp scheduling, resident warps, and resource limits

GPU Programming for AI  
Lecture 5 · 60 minutes

---

# Learning outcomes

By the end of this lecture, you should be able to:

- explain how warp scheduling hides latency
- distinguish latency, throughput, occupancy, and utilization
- define occupancy using resident warps
- identify the resources that limit block residency
- use CUDA's occupancy API as a starting point
- reason about block size instead of guessing it

---

# Two different performance questions

**Latency** asks:

> How long does one operation take?

**Throughput** asks:

> How many operations finish in a given time?

A GPU primarily targets **high throughput** by keeping many operations in flight.

---

# Why can a warp stop making progress?

A warp may be unable to issue its next instruction because it is waiting for:

- data from global memory
- the result of a previous instruction
- a synchronization barrier
- an available execution unit

The warp is **stalled**, but the whole SM does not have to stall.

---

# A useful model of warp state

| State | Meaning |
|---|---|
| **ready** | its next instruction can be issued |
| **executing** | an instruction is being processed |
| **stalled** | its next instruction cannot yet be issued |
| **finished** | no instructions remain |

The scheduler can only select a **ready** warp.

---

# The warp scheduler hides waiting time

If one warp is stalled, the scheduler can issue an instruction from another ready warp.

```text
warp A: issue load ───────── waiting ───────── use value
warp B:       issue arithmetic ── issue store
warp C:              issue load ───── waiting ────────
warp D:                    issue arithmetic
```

Useful work from other warps fills otherwise idle issue opportunities.

---

# Latency hiding does not reduce latency

The load of warp A does **not** return sooner.

Instead, its waiting time overlaps work from warps B, C, and D.

```text
lower latency     = one operation finishes sooner
latency hiding    = other work proceeds while it waits
higher throughput = more total work finishes per unit time
```

---

# Hiding latency requires independent work

The SM needs enough ready instructions from other warps.

Latency hiding becomes difficult when:

- too few warps are resident
- many warps wait for the same reason
- a block-wide barrier stops every warp in the block
- dependencies form long instruction chains

More resident warps create **opportunity**, not a guarantee.

---

# From blocks to resident warps

When a block becomes resident on an SM:

- all of its warps become resident there
- its registers and shared memory are reserved
- it stays on that SM until the block finishes

Several blocks may be resident on the same SM if resources allow it.

---

# Occupancy

**Occupancy** is the ratio of active warps to the maximum number of active warps supported by one SM.

$$
\text{occupancy} =
\frac{\text{active warps per SM}}
     {\text{maximum active warps per SM}}
$$

Here, **active** means **resident**—not necessarily executing right now.

---

# What occupancy is not

| Metric | Question |
|---|---|
| occupancy | How many warp slots contain resident warps? |
| SM utilization | How often is the SM doing useful work? |
| lane efficiency | How many lanes participate in issued instructions? |
| memory throughput | How many bytes are transferred per second? |

High occupancy does not automatically imply high values for the other metrics.

---

# What limits residency?

The number of resident blocks is bounded by several SM resources:

- maximum blocks per SM
- maximum threads or warps per SM
- register capacity
- shared-memory capacity

The **first exhausted resource** determines the limit.

---

# Threads and block size

A block with more threads contains more warps.

```text
64 threads/block  →  2 warps/block
128 threads/block →  4 warps/block
256 threads/block →  8 warps/block
```

Larger blocks consume the SM's thread and warp capacity in larger pieces.

---

# Registers can limit resident blocks

Registers are allocated to the threads of resident blocks.

```text
registers per thread × threads per block × resident blocks
```

More registers per thread can reduce the number of blocks that fit on an SM.

Compiler report:

```bash
nvcc --ptxas-options=-v kernel.cu
```

---

# Shared memory can limit resident blocks

Shared memory is reserved **per resident block**.

```text
shared memory per block × resident blocks
```

For the tiled transpose:

**Device (GPU) code**

```cpp
__shared__ float tile[32][33];
```

The padding avoids bank conflicts, but the allocation still consumes shared memory.

---

<!-- _class: compact -->

# A simplified occupancy example

Assume an SM supports at most:

- 1536 resident threads
- 8 resident blocks
- enough registers and shared memory for this kernel

| Block size | Thread limit | Block limit | Resident blocks | Resident warps |
|---:|---:|---:|---:|---:|
| 128 | 12 blocks | 8 blocks | 8 | 32 |
| 256 | 6 blocks | 8 blocks | 6 | 48 |
| 512 | 3 blocks | 8 blocks | 3 | 48 |

The numbers are hypothetical: always use the limits of the target GPU.

---

<!-- _class: compact -->

# Ask the CUDA runtime

**Host (CPU) code**

```cpp
int minGridSize;
int blockSize;

cudaOccupancyMaxPotentialBlockSize(
    &minGridSize,
    &blockSize,
    myKernel,
    0,
    0);
```

This suggests a launch configuration for **occupancy**. It does not benchmark the kernel.

---

# Calculate active blocks directly

**Host (CPU) code**

```cpp
int activeBlocks;

cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &activeBlocks,
    myKernel,
    blockSize,
    dynamicSharedBytes);
```

Convert active blocks to active warps:

```text
active warps = active blocks × ceil(block size / 32)
```

---

# 100% occupancy is not the goal

Higher occupancy often helps until enough latency is hidden.

After that point, more occupancy may add no benefit.

Reducing register use merely to raise occupancy can even create extra local-memory traffic.

> Occupancy is a constraint and a diagnostic—not a performance score.

---

# A practical block-size strategy

1. Use a multiple of 32 threads.
2. Start with 128 or 256 threads per block.
3. Check registers and shared memory per block.
4. Use the occupancy API or calculator.
5. Benchmark realistic inputs.

The best block size depends on both the kernel and the GPU.

---

# Practical session: tiled transpose

Start from the global-memory transpose from Lecture 4.

Then:

1. load a tile coalescently into shared memory
2. synchronize the block with `__syncthreads()`
3. store the transposed tile coalescently
4. add padding to remove bank conflicts
5. measure effective bandwidth after every change

Compare **naive**, **tiled**, and **padded tiled** versions.

---

# Summary

- Latency is the time for one operation; throughput is work per unit time.
- A scheduler hides stalls by issuing instructions from other ready warps.
- Latency hiding needs enough independent resident work.
- Occupancy is active warps divided by the SM's maximum active warps.
- Threads, registers, shared memory, and block slots limit residency.
- High occupancy is useful only when it improves measured performance.
