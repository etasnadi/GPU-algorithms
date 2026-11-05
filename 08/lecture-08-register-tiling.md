---
marp: true
theme: minimal-white
paginate: true
size: 16:9
math: mathjax
footer: GPU Programming for AI · Lecture 8
---

# Register Tiling and Thread-Level Reuse

## Multiple outputs per thread in matrix multiplication

GPU Programming for AI  
Lecture 8

---

# The current GEMM kernels

| Kernel | Data reuse | Output per thread |
|---|---|---:|
| `matmul_naive` | implicit caches (L1, L2) | one element |
| `matmul_shmem` | explicit block-level reuse | one element |

The shared-memory kernel reduces global-memory traffic.

It still performs the inner product with one accumulator per thread.

```cpp
sum += A_tile[threadIdx.y][k]
     * B_tile[k][threadIdx.x];
```

---

# How loop unrolling works

<style scoped>
section { font-size: 24px; padding-bottom: 42px; padding-top: 44px; }
h1 { font-size: 41px; margin-bottom: 16px; }
pre { font-size: 17px; line-height: 1.24; padding: 12px 16px; }
li { margin: 4px 0; }
</style>

When the iteration count is known at compile time, the compiler can replace the loop with repeated copies of its body.

```cpp
#pragma unroll
for (int k = 0; k < 4; ++k)
    sum += a[k] * b[k];
```

Conceptually becomes:

```cpp
sum += a[0] * b[0];
sum += a[1] * b[1];
sum += a[2] * b[2];
sum += a[3] * b[3];
```

- Loop-control instructions disappear, and constant indices help optimization.
- The same thread performs the same work; unrolling does not create CUDA threads.
- Larger code and increased register pressure are possible costs.

---

# One thread computes one output

**Device (GPU) code**

```cpp
float sum = 0.0f;

#pragma unroll
for (int k = 0; k < TILE_SIZE; ++k) {
    float a = A_tile[threadIdx.y][k];
    float b = B_tile[k][threadIdx.x];
    sum += a * b;
}
```

Every iteration reads two shared-memory values and updates the same register.

The next update of `sum` depends on the previous update.

---

# Registers

Registers have **thread-local scope** and reside on the SM (threads have private registers).

The compiler normally places scalar local variables in registers:

```cpp
float sum;
float a;
float b;
```

Registers provide low-latency storage for:

- input values reused by one thread
- partial results
- loop state and addresses

Threads cannot directly read another thread's registers.

---

# A thread tile

Instead of one output, a thread computes a small rectangular tile:

```text
              output columns
             0    1    2    3
output row  ┌────┬────┬────┬────┐
          0 │c00 │c01 │c02 │c03 │
            └────┴────┴────┴────┘
```

The thread keeps four accumulators:

```cpp
float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
```

One loaded `A` value contributes to all four outputs.

---

# A two-dimensional thread tile

<style scoped>
section { font-size: 24px; padding-bottom: 42px; padding-top: 44px; }
h1 { font-size: 41px; margin-bottom: 16px; }
pre { font-size: 17px; line-height: 1.24; padding: 12px 16px; }
</style>

**Device (GPU) code**

```cpp
float accum[THREAD_M][THREAD_N] = {};

#pragma unroll
for (int k = 0; k < TILE_K; ++k) {
    float a[THREAD_M];
    float b[THREAD_N];

    #pragma unroll
    for (int i = 0; i < THREAD_M; ++i)
        a[i] = A_tile[local_row + i][k];

    #pragma unroll
    for (int j = 0; j < THREAD_N; ++j)
        b[j] = B_tile[k][local_column + j];

    #pragma unroll
    for (int i = 0; i < THREAD_M; ++i)
        #pragma unroll
        for (int j = 0; j < THREAD_N; ++j)
            accum[i][j] += a[i] * b[j];
}
```

Each loop iteration computes a small outer product.

---

# Independent accumulators in a thread tile

<style scoped>
section { font-size: 24px; padding-bottom: 42px; padding-top: 44px; }
h1 { font-size: 41px; margin-bottom: 16px; }
pre { font-size: 17px; line-height: 1.24; padding: 12px 16px; }
</style>

Before register tiling, one thread computes one output with one accumulator:

```text
sum ← FMA(sum) ← FMA(sum) ← FMA(sum)
```

A $T_M\times T_N$ thread tile computes $T_MT_N$ output elements. It therefore keeps $T_MT_N$ accumulators, one for each output.

For a $1\times4$ thread tile:

```text
accum[0] ← FMA(accum[0])
accum[1] ← FMA(accum[1])
accum[2] ← FMA(accum[2])
accum[3] ← FMA(accum[3])
```

Each accumulator belongs to a different output element, so the four dependency chains are independent. The scheduler can issue arithmetic from one chain while another result is pending.

This is **instruction-level parallelism**, or ILP.

---

# Reuse at the shared-memory boundary

For one value of `k`, a `THREAD_M × THREAD_N` thread tile performs:

$$
2\,T_M T_N\ \text{FLOPs}
$$

and loads:

$$
4(T_M+T_N)\ \text{bytes from shared memory}
$$

| Thread tile | Arithmetic Intensity |
|---:|---:|
| $1\times1$ | 0.25 |
| $2\times2$ | 0.50 |
| $4\times4$ | 1.00 |

Larger thread tiles reuse each loaded value in more FMA instructions.

---

# Register pressure and occupancy

<style scoped>
section { font-size: 24px; padding-bottom: 42px; padding-top: 44px; }
h1 { font-size: 41px; margin-bottom: 16px; }
.flow-chain { align-items: center; display: flex; font-size: 19px; gap: 7px; margin: 34px 0; }
.flow-chain .step { flex: 1 1 0; line-height: 1.15; text-align: center; }
.flow-chain .arrow { flex: 0 0 auto; font-size: 26px; }
</style>

**Question:** We want to increase arithmetic intensity. Are larger thread tiles always better?

For a fixed number of threads per block, a larger thread tile produces a larger block output tile:

<div class="flow-chain">
<span class="step">larger thread tile</span>
<span class="arrow">→</span>
<span class="step">larger block output tile</span>
<span class="arrow">→</span>
<span class="step">larger A and B tiles in shared memory</span>
<span class="arrow">→</span>
<span class="step">more shared memory per block</span>
<span class="arrow">→</span>
<span class="step">fewer resident blocks and warps per SM</span>
<span class="arrow">→</span>
<span class="step">less potential latency hiding</span>
</div>

The best tile size balances higher data reuse against enough resident warps to hide latency.

---

# Register spilling

<style scoped>
section { font-size: 24px; padding-bottom: 42px; padding-top: 44px; }
h1 { font-size: 41px; margin-bottom: 16px; }
.flow-chain { align-items: center; display: flex; font-size: 20px; gap: 9px; margin: 36px 0; }
.flow-chain .step { flex: 1 1 0; line-height: 1.15; text-align: center; }
.flow-chain .arrow { flex: 0 0 auto; font-size: 27px; }
</style>

Likewise, larger thread tiles need more registers for operands and accumulators:

<div class="flow-chain">
<span class="step">larger thread tile</span>
<span class="arrow">→</span>
<span class="step">more live operands and accumulators</span>
<span class="arrow">→</span>
<span class="step">higher register usage per thread</span>
<span class="arrow">→</span>
<span class="step">insufficient registers</span>
<span class="arrow">→</span>
<span class="step">values spilled to local memory</span>
</div>

The compiler generates local-memory loads and stores for spilled values.

Local memory has thread-local addressing, but its physical storage is in device memory. Its traffic is cached, but it remains much slower than register access.

Spilling can remove the performance benefit of register tiling.

---

# Which memory traffic changes?

| Level | Effect of thread tiling |
|---|---|
| global memory | may remain unchanged for the same block tile |
| shared memory | each loaded value feeds more arithmetic |
| registers | more inputs and accumulators stay live |

Thread tiling directly improves register and shared-memory reuse.

It reduces global traffic only when the overall block tile or loading strategy also changes.

---

# Mapping a 2 × 2 thread tile

Assume:

```text
block output tile: 16 × 16 elements
thread block:        8 × 8 threads
thread output tile:  2 × 2 elements
```

Thread `(tx, ty)` owns:

```cpp
int first_row    = block_row + 2 * ty;
int first_column = block_col + 2 * tx;

C[first_row + i][first_column + j]
```

where `i, j ∈ {0, 1}`.

All 64 thread tiles cover the 16 × 16 block tile exactly once.

---

# How many registers does a kernel use?

<style scoped>
section { font-size: 24px; padding-bottom: 42px; padding-top: 44px; }
h1 { font-size: 41px; margin-bottom: 16px; }
pre { font-size: 17px; line-height: 1.24; padding: 12px 16px; }
li { margin: 4px 0; }
</style>

Shared-memory usage is easy to determine because the kernel explicitly allocates and manages it.

**But how many registers does each thread use?**

The compiler decides the register allocation after optimizing the kernel. Ask the compiler:

```bash
nvcc -O3 -lineinfo --resource-usage matmul.cu
```

Or inspect the kernel's **Launch Statistics** in Nsight Compute:

```bash
ncu --section LaunchStats ./matmul
```

Both methods report the register usage per thread. Use this value to analyze occupancy and possible register spilling.

