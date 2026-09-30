---
marp: true
theme: minimal-white
paginate: true
size: 16:9
footer: GPU Programming for AI · Lecture 4
---

<!-- _paginate: false -->
<!-- _footer: "" -->

# Global and Shared Memory Access

## Coalescing, memory transactions, and shared-memory banks

GPU Programming for AI  
Lecture 4 · 90 minutes


---

<!-- _class: compact -->

# Learning outcomes

By the end of this lecture, you should be able to:

- explain coalesced global-memory access
- predict the effect of alignment and stride
- map a row-major matrix to CUDA threads
- explain why naive transpose has an inefficient store pattern
- describe shared-memory banks
- identify and avoid basic bank conflicts
- calculate effective bandwidth


---

# A warp issues memory requests

**Device (GPU) code**

```cpp
float value = input[i];
```

Each active lane computes an address.

```text
lane 0  requests address A0
lane 1  requests address A1
...
lane 31 requests address A31
```

The memory subsystem combines the warp's addresses into the transactions required to serve them.


---

# Requested data and transferred data

The kernel requests individual words.

The memory system transfers aligned memory regions.

```text
requested words  →  memory transactions  →  cache or DRAM
```

Efficient access uses most of the bytes in each transferred region.

Inefficient access moves bytes that the warp never requested.


---

# Coalesced access

A global-memory access is **coalesced** when the addresses requested by a warp can be served with few memory transactions.

The common efficient pattern is:

```text
consecutive lanes access consecutive words
```

For 32-bit values:

```text
lane k accesses element base + k
```


---

# Contiguous tensor access

**Device (GPU) code**

```cpp
int i =
    blockIdx.x * blockDim.x +
    threadIdx.x;

if (i < n) {
    output[i] = input[i];
}
```

```text
lane:      0   1   2   3   4   ... 31
element:   i  i+1 i+2 i+3 i+4  ... i+31
```

Adjacent lanes access adjacent elements.

---

# Why contiguous access is efficient

On modern NVIDIA GPUs, global-memory accesses are serviced using aligned 32-byte transactions.

A warp reading 32 consecutive `float` values requests 128 useful bytes.

When the first address is suitably aligned, the request fits into four adjacent 32-byte regions.

```text
32 floats × 4 bytes = 128 bytes
128 bytes ÷ 32 bytes = 4 transactions
```

---

# Alignment

An aligned access starts at a natural transaction boundary.

```text
aligned:     [ requested requested requested requested ]

misaligned:      [ requested requested requested requested ]
```

A misaligned contiguous warp request may touch an additional memory region.

The access remains sequential, but it can require more transferred bytes.

---

# Offset access

**Device (GPU) code**

```cpp
int i =
    blockIdx.x * blockDim.x +
    threadIdx.x;

if (i < n - offset) {
    output[i] = input[i + offset];
}
```

The lanes still access consecutive elements.

The offset changes alignment, not the distance between neighboring lane addresses.

---

# Strided access

**Device (GPU) code**

```cpp
if (i < output_size) {
    output[i] = input[i * stride];
}
```

For `stride = 8`:

```text
lane:      0   1    2    3    4
element:   0   8   16   24   32
```

The warp touches many separated memory regions and uses only a small part of each transaction.


---

# Coalescing is a warp-level property

These statements are insufficient:

- each thread reads one `float`
- every address is valid
- the array is contiguous in memory

The important question is:

> What addresses do adjacent lanes request in the same memory instruction?

Correct indexing determines correctness. Thread-to-data mapping also determines memory efficiency.


---

# Row-major matrix storage

For a matrix with `width` columns:

$$
\operatorname{index}(y,x)=y\cdot\text{width}+x
$$

```text
row 0: a00 a01 a02 a03 ...
row 1: a10 a11 a12 a13 ...
row 2: a20 a21 a22 a23 ...
```

Elements in one row occupy consecutive addresses.

Elements in one column are separated by `width` elements.


---

# A two-dimensional thread mapping

**Device (GPU) code**

```cpp
int x =
    blockIdx.x * blockDim.x +
    threadIdx.x;

int y =
    blockIdx.y * blockDim.y +
    threadIdx.y;
```

The x dimension changes fastest inside a warp for the usual two-dimensional block layout.

Map `threadIdx.x` to the contiguous matrix dimension.


---

# Matrix copy as a baseline

**Device (GPU) code**

```cpp
if (x < width && y < height) {
    int index = y * width + x;
    output[index] = input[index];
}
```

Adjacent lanes vary `x`:

- input reads are contiguous
- output writes are contiguous

The copy kernel provides a bandwidth-oriented baseline for transpose.


---

<!-- _class: compact -->

# Naive matrix transpose

We can either use 1D or 2D indexing.

**Device (GPU) code**

```cpp
__global__ void transpose_naive(
    const float* input,
    float* output,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        output[x * height + y] =
            input[y * width + x];
    }
}
```


---

# Read pattern of naive transpose

The input index is:

$$
y\cdot\text{width}+x
$$

Within a warp, `y` is fixed while `x` changes across adjacent lanes.

```text
input[y, x+0], input[y, x+1], ...
```

The global-memory reads are contiguous and coalesced.


---

# Write pattern of naive transpose

The output index is:

$$
x\cdot\text{height}+y
$$

Adjacent lanes change `x`, so their output addresses differ by `height` elements.

```text
output[x+0, y], output[x+1, y], ...
```

The stores use a large stride and require many memory transactions.


---

# Correctness and access efficiency

The naive transpose is correct.

```text
output[x, y] = input[y, x]
```

Its performance is limited by the output address pattern.

This distinction is central to GPU programming:

1. derive correct indices
2. inspect warp-level addresses
3. measure the resulting traffic and runtime


---

# Effective bandwidth

Copy and out-of-place transpose each read and write every matrix element once.

For `N` elements:

$$
B_{\text{useful}}=2N\cdot\operatorname{sizeof}(\texttt{float})
$$

$$
BW_{\text{effective}}=\frac{B_{\text{useful}}}{t}
$$

The metric counts useful bytes, not every byte moved internally by caches and DRAM.


---

# Bandwidth calculation

**Host (CPU) code**

```cpp
double useful_bytes =
    2.0 * width * height * sizeof(float);

double bandwidth_gb_s =
    useful_bytes /
    (milliseconds * 1.0e-3) /
    1.0e9;
```

Use a warm-up launch and average several repetitions.

CUDA events measure elapsed time on the CUDA stream timeline.


---

# Caches do not repair the mapping

L1/TEX and L2 can reuse cache lines and reduce some DRAM traffic.

They cannot change which addresses the warp requests.

A large-stride store still spreads the warp's useful words across many memory regions.

The first optimization question remains the thread-to-data mapping.


---

# Shared memory as a programmer-managed tile

Shared memory allows threads in one block to exchange data through on-chip storage.

**Device (GPU) code**

```cpp
__shared__ float tile[32][32];
```

Shared memory can change the order in which data is consumed or written without changing the global-memory layout.

Its performance depends on **banks**, not global-memory coalescing.


---

# Shared-memory banks

Shared memory is divided into independently accessible banks.

For the common 32-bit model:

- there are 32 banks
- consecutive 32-bit words map to consecutive banks
- bank numbers repeat every 32 words

$$
\operatorname{bank}(i)=i\bmod 32
$$


---

# Conflict-free row access

**Device (GPU) code**

```cpp
float value =
    tile[threadIdx.y][threadIdx.x];
```

For a warp with fixed `threadIdx.y` and `threadIdx.x = 0..31`:

```text
lane:   0  1  2  3  ... 31
bank:   0  1  2  3  ... 31
```

Each lane accesses a different bank, so the request can proceed in parallel.


---

# Shared-memory bank conflict

**Device (GPU) code**

```cpp
float value =
    tile[threadIdx.x][fixed_column];
```

For `float tile[32][32]`, consecutive lanes access words 32 positions apart.

```text
word:   c  c+32  c+64  c+96  ...
bank:   b    b     b     b    ...
```

The requests target the same bank and must be split into conflict-free operations.


---

# Why bank conflicts reduce throughput

A bank can serve limited data per cycle.

When different lanes request different words from the same bank, the hardware serializes the request into multiple phases.

```text
different banks  → parallel service
same bank        → serialized service
```

The code remains correct, but effective shared-memory bandwidth decreases.


---

# Broadcast is a special case

If several lanes read the same shared-memory word, the hardware can broadcast that word.

```text
lane 0  reads shared[k]
lane 1  reads shared[k]
lane 2  reads shared[k]
...
```

This does not create the same conflict as reading different words from one bank.


---

# Simple resolution for bank conflicts: padding

**Device (GPU) code**

```cpp
__shared__ float tile[32][33];
```

The extra column changes the row stride from 32 to 33 words.

Column accesses now advance by one bank between neighboring rows:

```text
bank:  b, b+1, b+2, b+3, ...
```

Padding can remove the 32-way conflict without changing the logical 32 by 32 tile.

Drawback: we waste space for 33 elements. Thus, we can use less shared memory for caching, that 
can reduce the performance in other stages of the kernel.

---

# Two different optimization rules

| Memory space | Efficient warp access |
|---|---|
| global memory | adjacent lanes access adjacent words |
| shared memory | adjacent lanes access different banks |

For global memory, minimize the number of memory transactions.

For shared memory, minimize bank conflicts.

The address pattern must be evaluated separately for each memory space.


---

<!-- _class: compact -->

# Practical session: global-memory transpose

The practical session implements two kernels:

1. a two-dimensional matrix copy
2. a naive out-of-place matrix transpose

Shared memory is deliberately excluded from this implementation.


---

# The next optimization step

The naive transpose exposes a global-memory store problem.

A later version will:

- load a tile from global memory with coalesced access
- reorder the tile in shared memory
- store the transposed tile with coalesced access
- add padding to avoid shared-memory bank conflicts

Shared-memory usage also affects how many blocks can reside on an SM.

---

# Summary

- Coalescing combines a warp's global-memory requests into transactions.
- Adjacent lanes should usually access adjacent global-memory words.
- Strided accesses waste transferred bandwidth.
- Naive transpose reads coalescently but writes with a large stride.
- Shared memory uses banks rather than global-memory coalescing rules.
- Different words in one bank create a bank conflict.
- Padding can change the bank mapping.

