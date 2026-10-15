---
marp: true
theme: minimal-white
paginate: true
size: 16:9
footer: GPU Algorithms · Quiz
---

<style>
section.quiz {
  font-size: 18px;
  line-height: 1.2;
  padding: 30px 48px 34px;
}

section.quiz h1 {
  font-size: 34px;
  margin-bottom: 10px;
}

section.quiz p {
  margin: 7px 0;
}

section.quiz pre {
  font-size: 13px;
  line-height: 1.15;
  margin: 7px 0;
  padding: 9px 12px;
}

section.quiz ul,
section.quiz ol {
  margin: 6px 0;
}

section.quiz li {
  margin: 3px 0;
}
</style>

<!-- _paginate: false -->
<!-- _footer: "" -->

# GPU Algorithms - Quiz

Each question has exactly one correct answer.


---

<!-- _class: quiz -->

# Useful information

**Kernel call syntax:**

```cpp
__global__ void kernel_name(){
    // define static shared memory
    __shared__ float tmp[32];
}
```

**Kernel launch:**

```cpp
kernel_name<<<num_blocks, elems_per_block, dynamic_shmem_size, stream>>>();
```


---

<!-- _class: quiz -->

# Question 1

Consider the following kernel. The last block may contain threads for which `i >= n`.

```cpp
__global__ void neighbor_sum(const float* input, float* output, int n) {
    __shared__ float tile[128];
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    // (a)
    if (i < n) {
        tile[t] = input[i];
    } else {
        tile[t] = 0.0f;
    }
    // (b)
    if (i < n) {
        // (c)
        int neighbor = (t + 1) % blockDim.x;
        output[i] = tile[t] + tile[neighbor];
    }
}
```

The kernel is launched with 128 threads per block. A thread may read a value written into shared memory by another thread.

Which operation should be inserted, and at which marked position?

A. Insert `__syncthreads()` at position **(a)**.  
B. Insert `__syncthreads()` at position **(b)**.  
C. Insert `__syncthreads()` at position **(c)**.  
D. Insert `__syncwarp()` at position **(c)**.

---

<!-- _class: quiz -->

# Question 2

Each block has already computed one partial sum. The following kernel attempts to combine them:

```cpp
__global__ void combine_partials(const float* partial, float* result) {
    if (threadIdx.x == 0) {
        *result += partial[blockIdx.x];
    }
}
```

The kernel is launched with multiple blocks. The `*result += ...` expression consists of an ordinary, non-atomic read followed by a write.

Which modification provides a correct and scalable solution?

A. Execute a block-level barrier before each block updates `result`.  
B. Let each block write to a separate location, then use another kernel to combine the partial results.  
C. Assume that block 0 executes before the other blocks.  
D. Move `result` to faster memory without changing the update operation.

---

<!-- _class: quiz -->

# Question 3

`S1` and `S2` are two different non-default streams.

```cpp
producer<<<grid, block, 0, S1>>>(data);
consumer<<<grid, block, 0, S2>>>(data);
```

The `consumer` kernel can produce a correct result only after `producer` has finished writing `data`. The streams should remain separate, and the CPU should not wait.

Which solution provides the required ordering?

A. No modification is needed because the CPU submits `producer` first.  
B. Use a block-level barrier at the beginning of `consumer`.  
C. Record an event in `S1` after `producer`, then make `S2` wait for that event before `consumer`.  
D. Synchronize `S2` before submitting `consumer`.

---

<!-- _class: quiz -->

# Question 4

The following kernels operate on independent data:

```cpp
kernel_A<<<gridA, blockA, 0, S1>>>(data_A);
kernel_B<<<gridB, blockB, 0, S1>>>(data_B);

kernel_C<<<gridC, blockC, 0, S2>>>(data_C);
```

`S1` and `S2` are two different non-default streams.

Which statement is correct?

A. All three kernels are guaranteed to execute at the same time.  
B. `kernel_B` may begin before `kernel_A` finishes because kernel launches are asynchronous with respect to the CPU.  
C. `kernel_A` must finish before `kernel_B` begins, while `kernel_C` may overlap with either of them if sufficient GPU resources are available.  
D. Using different streams guarantees that `kernel_C` overlaps with another kernel.

---

<!-- _class: quiz -->

# Question 5

The following kernel is launched with exactly one block containing 32 threads. For this question, these 32 threads form one warp.

```cpp
__global__ void read_with_stride(
    const float* input, float* output, int stride) {
    int k = threadIdx.x;
    output[k] = input[k * stride];
}
```

Assume that:

- one `float` occupies 4 bytes;
- the starting address of `input` is aligned to a 32-byte boundary;
- global-memory reads are served using aligned 32-byte sectors;
- only the read from `input` is considered;
- previous cache contents are ignored.

The kernel is executed twice:

```cpp
read_with_stride<<<1, 32>>>(input, output, 1);
read_with_stride<<<1, 32>>>(input, output, 8);
```

How many different memory sectors does the warp’s load instruction touch?

A. `stride = 1`: 4 sectors; `stride = 8`: 4 sectors  
B. `stride = 1`: 4 sectors; `stride = 8`: 32 sectors  
C. `stride = 1`: 16 sectors; `stride = 8`: 32 sectors  
D. `stride = 1`: 32 sectors; `stride = 8`: 32 sectors

---

<!-- _class: quiz -->

# Answer Key - Questions 1-3

1. **B** - The barrier must appear after every thread has written its shared-memory value and before any thread reads another thread’s value. Position (a) is too early. Position (c) is inside a condition that may not be entered by every thread in the final block.

2. **B** - Updates from different blocks may overlap, causing partial sums to be lost. Separate output locations followed by another kernel create two globally ordered phases.

3. **C** - Submission order on the CPU does not establish a dependency between different streams. An event can order the required GPU operations without blocking the CPU.


---

<!-- _class: quiz -->

# Answer Key - Questions 4-5

4. **C** - Operations in one stream execute in order. Operations in different streams may overlap, but concurrency depends on available hardware resources.

5. **B** - With `stride = 1`, the warp reads 128 consecutive bytes, covering four 32-byte sectors. With `stride = 8`, adjacent threads access addresses 32 bytes apart, so every thread touches a different sector.
