---
marp: true
theme: minimal-white
paginate: true
size: 16:9
math: mathjax
footer: GPU Programming for AI · Lecture 10
---

# Combining Results Across Blocks

## Split-K matrix multiplication, reduction kernels, and atomic updates

GPU Programming for AI  
Lecture 10

---

# One output tile normally belongs to one block

In a tiled GEMM, one block owns a tile of $C$ and traverses the full inner dimension:

$$
C_{mn}=\sum_{k=0}^{K-1} A_{mk}B_{kn}
$$

```text
block (tile_m, tile_n)
    visits every K tile
    accumulates one C tile
    writes the final tile once
```

This mapping gives exclusive ownership of each output element.

It may expose too few blocks when $M$ and $N$ are small but $K$ is large.

---

# Splitting the inner dimension

Divide $K$ into $S$ disjoint ranges:

$$
[0,K) = K_0 \cup K_1 \cup \cdots \cup K_{S-1}
$$

Each range produces a partial matrix:

$$
P^{(s)}_{mn}
=\sum_{k\in K_s} A_{mk}B_{kn}
$$

The final output is:

$$
C_{mn}=\sum_{s=0}^{S-1}P^{(s)}_{mn}
$$

The arithmetic remains a matrix multiplication. The ownership of each output changes.

---

# The grid gains a split-K dimension

```text
grid.x: output-column tiles
grid.y: output-row tiles
grid.z: K ranges
```

For one output tile:

| Block | K range | Result |
|---|---|---|
| $(x,y,0)$ | $K_0$ | $P^{(0)}$ tile |
| $(x,y,1)$ | $K_1$ | $P^{(1)}$ tile |
| $\vdots$ | $\vdots$ | $\vdots$ |
| $(x,y,S-1)$ | $K_{S-1}$ | $P^{(S-1)}$ tile |

The number of blocks grows by approximately the split factor $S$.

---

# Partial results need separate storage

Several blocks contribute to the same coordinates of $C$.

The first phase gives every split a private output plane:

```text
partial[0, :, :]   P^(0)
partial[1, :, :]   P^(1)
partial[2, :, :]   P^(2)
       ...
partial[S-1, :, :] P^(S-1)
```

Conceptual workspace size:

$$
S\times M\times N\times \text{sizeof(element)}
$$

Private planes avoid write conflicts during the matrix-multiplication phase.

---

# Phase 1: partial matrix multiplication

**Device-side scheme**

```cpp
split = blockIdx.z
(k_begin, k_end) = range_of(split, K)

accumulator_tile = 0

for k_tile in [k_begin, k_end):
    load A tile and B tile
    accumulate products

write accumulator_tile to partial[split, :, :]
```

Each block follows the familiar tiled-GEMM structure.

The difference is that its loop covers one $K$ range and its result goes to a workspace plane.

---

# Phase 2: reduction across the split dimension

The reduction kernel owns output coordinates of $C$:

```cpp
(row, column) = output_coordinate()

sum = 0
for split in [0, S):
    sum += partial[split, row, column]

C[row, column] = sum
```

For each output element, the reduction runs along the new workspace dimension.

$$
C_{mn}=P^{(0)}_{mn}+P^{(1)}_{mn}+\cdots+P^{(S-1)}_{mn}
$$

This phase performs no matrix multiplication. It only combines partial outputs.

---

# Two kernel launches form one algorithm

```text
A, B
  │
  ▼
split-K GEMM kernel
  │ writes S partial matrices
  ▼
partial[S, M, N]
  │
  ▼
K-split reduction kernel
  │ writes one final matrix
  ▼
C[M, N]
```

The kernel boundary provides the global phase boundary that blocks cannot create with `__syncthreads()`.

---

# The kernels belong to the same stream

**Host-side scheme**

```cpp
split_k_gemm<<<gemm_grid, gemm_block, ..., stream>>>(
    A, B, partial, ...);

reduce_k_splits<<<reduce_grid, reduce_block, ..., stream>>>(
    partial, C, ...);
```

Operations in one stream execute in enqueue order:

```text
stream:  split-K GEMM  ─────────  reduction
```

The host does not need to wait between the launches.

The reduction starts only after the matrix-multiplication kernel completes.

---

# The two phases cannot run in parallel

The reduction reads every partial result produced by the first kernel.

```text
producer: partial[0], partial[1], ... partial[S-1]
consumer: all partial planes for each output coordinate
```

Launching the kernels independently in different streams would not establish this dependency.

Possible failure:

```text
reduction reads partial[s, m, n]
before the producing block writes it
```

For this algorithm, a single stream expresses the required order directly.

---

# Same stream does not block the CPU

Kernel launches remain asynchronous with respect to the host:

```text
CPU:  enqueue GEMM  enqueue reduction  continue host work

GPU stream:
      run GEMM      run reduction
```

Synchronize only when the host needs the final matrix.

Examples of later completion points:

- an asynchronous copy of $C$ followed by stream synchronization
- an event recorded after the reduction
- another kernel in the same stream that consumes $C$

---

# Why split K?

Split K creates more independent blocks when the output grid is too small to occupy the GPU well.

It is most plausible when:

- $M$ and $N$ create few output tiles
- $K$ is large enough to divide into useful ranges
- each split still performs enough work to amortize a block and kernel launch

It is less attractive when the ordinary $M\times N$ tiling already exposes enough blocks.

---

# The extra parallelism has a cost

| Added resource or work | Consequence |
|---|---|
| partial-result workspace | extra device memory |
| writes of $S$ partial matrices | extra global-memory traffic |
| reads by the reduction kernel | extra global-memory traffic |
| second kernel launch | launch overhead |
| final additions | additional arithmetic |

Split K trades more parallel work in the GEMM phase for a separate merge phase.

Measure the complete pair of kernels, not only the first phase.

---

# Returning to block-level reduction

The earlier example in `03/block-reduce.cu` launches one block:

```text
128 inputs
    │
    ▼
one shared-memory reduction tree
    │
    ▼
out[0]
```

The block produces a complete result only because the entire input fits in that block.

With several blocks, each block produces only a block sum.

---

# Several blocks create several contributors

```text
input chunk 0 ── block 0 ── block_sum_0 ─┐
input chunk 1 ── block 1 ── block_sum_1 ─┤
input chunk 2 ── block 2 ── block_sum_2 ─┼─ final sum
input chunk 3 ── block 3 ── block_sum_3 ─┘
```

Inside each block:

1. threads load and accumulate input values
2. the block reduces its values in shared memory
3. one thread holds the block sum

The remaining question is how block sums update one global result safely.

---

# A normal addition can lose updates

Suppose two blocks execute this logical sequence on the same address:

```text
read output
add block_sum
write output
```

Possible interleaving:

| Block 0 | Block 1 |
|---|---|
| reads 10 | reads 10 |
| adds 4 | adds 7 |
| writes 14 | writes 17 |

The correct result is 21, but one update disappears.

This is a read-modify-write race.

---

# Atomic addition

An atomic operation makes one read-modify-write indivisible with respect to competing atomic updates at its scope.

Schematic operation:

```cpp
atomicAdd(output, block_sum)
```

Conceptually:

```text
old = output
output = old + block_sum
```

No other atomic update to the same address can interleave inside that transaction.

Atomicity prevents lost updates. It does not create a grid-wide barrier.

---

# One atomic update per block

```text
many input values
      │
      ▼
shared-memory reduction inside each block
      │
      ▼
one block sum
      │
      ▼
one atomicAdd to the global result
```

For $B$ blocks, this design performs approximately $B$ global atomic additions.

Using one atomic addition for every input value would create much more contention at the same address.

The block-level reduction acts as an aggregation step before the global atomic update.

---

# Initialization and completion are separate concerns

The global accumulator must contain the identity value before blocks update it:

```text
same stream:
    set output to 0
    launch multi-block reduction
    consume or copy the final output
```

The atomic operation protects concurrent updates during the kernel.

Stream ordering establishes when initialization finishes and when the final result is ready.

These mechanisms solve different problems.

---

# Atomic contention

Atomic updates to one address must be serialized at that address.

The cost grows when:

- many blocks finish at nearly the same time
- every block targets the same output value
- the work inside each block is small compared with the atomic update

Aggregation reduces the number of contenders:

```text
per-element atomic updates   many updates
per-block atomic updates     one update per block
```

Atomics are most useful when the number of global updates stays small.

---

# Floating-point addition depends on order

Block completion order is not specified.

Atomic updates may therefore add block sums in different orders across runs:

$$
(a+b)+c \neq a+(b+c)
$$

for some floating-point values.

The result can be mathematically valid within rounding error without being bitwise identical.

A separate reduction kernel can enforce a more structured reduction order when reproducibility matters.

---

# Two merge strategies

| Strategy | Intermediate storage | Kernel phases | Main limitation |
|---|---:|---:|---|
| workspace and reduction kernel | partial results | two | memory traffic and launch overhead |
| per-block atomic update | one global accumulator | one | contention and update order |

The split-K GEMM in this lecture uses a workspace and a separate reduction kernel.

The multi-block scalar reduction uses one atomic addition per block.

The output shape and expected contention determine which merge strategy is appropriate.

---

# Summary

- Split-K GEMM assigns several blocks to different ranges of the inner dimension.
- The first kernel writes private partial matrices to a workspace.
- A second kernel reduces the workspace along the split dimension.
- Both kernels belong to one stream because the reduction consumes the first kernel's results.
- A multi-block scalar reduction can merge block sums with one atomic addition per block.
- Atomicity prevents lost updates but can introduce contention and order-dependent rounding.
- Cross-block algorithms require an explicit merge phase and an explicit dependency.

