---
marp: true
theme: minimal-white
paginate: true
size: 16:9
footer: GPU Programming for AI · Lecture 2
---

<!-- _paginate: false -->
<!-- _footer: "" -->

# The CUDA Execution Model

## From an AI tensor operation to thousands of GPU threads

GPU Programming for AI  
Lecture 2 · 90 minutes


---

# The system around a CUDA GPU

<div class="hardware-model">
  <div class="hardware-label">Lab PC</div>
  <div class="hardware-path">
    <div class="hardware-node">System RAM</div><div class="hardware-link">memory bus</div>
    <div class="hardware-node">CPU</div><div class="hardware-link">PCIe 4.0</div>
    <div class="hardware-node">RTX 3060</div><div class="hardware-link">memory bus</div>
    <div class="hardware-node">GDDR6</div>
  </div>

  <div class="hardware-label">DGX B200</div>
  <div class="hardware-path">
    <div class="hardware-node">DDR5 system memory</div><div class="hardware-link">memory channels</div>
    <div class="hardware-node">2 Xeon CPUs</div><div class="hardware-link">PCIe</div>
    <div class="hardware-node dgx-gpus">8 Blackwell GPUs<br><small>local HBM3e<br>GPU ↔ GPU: NVLink + NVSwitch</small></div>
  </div>

  <div class="hardware-label">GB200 Superchip<br><small>one module</small></div>
  <div class="hardware-path">
    <div class="hardware-node">LPDDR5X</div><div class="hardware-link">memory channels</div>
    <div class="hardware-node">Grace CPU</div><div class="hardware-link">NVLink-C2C</div>
    <div class="hardware-node dgx-gpus">2 Blackwell GPUs<br><small>local HBM3e</small></div>
  </div>
</div>

The programming model remains recognizable. High-end systems replace isolated links with faster coherent and multi-GPU fabrics. A GB200 NVL72 rack combines 36 Grace CPUs and 72 Blackwell GPUs.

<p class="source-note">RTX 3060: PCIe Gen 4, GDDR6, no NVLink. DGX B200: 8 GPUs. One GB200 Superchip: 1 Grace CPU + 2 GPUs. GB200 NVL72: 36 Grace CPUs + 72 GPUs.</p>


---

# Learning outcomes

By the end of this lecture, you should be able to:

- distinguish host code from device code
- describe the lifecycle of a simple CUDA program
- explain the grid–block–thread hierarchy
- compute a global thread index
- choose a launch configuration for a 1D tensor
- explain the purpose of a boundary check
- identify asynchronous execution and basic CUDA errors


---

# From AI operations to parallel work

Consider an elementwise activation:

$$y_i = \max(0, x_i)$$

Sequential implementation:

**Host (CPU) code**

```cpp
for (int i = 0; i < n; ++i) {
    y[i] = std::max(0.0f, x[i]);
}
```

The loop iterations are independent. CUDA assigns them to different threads.


---

# The basic CUDA mapping

```text
thread 0  processes tensor element 0
thread 1  processes tensor element 1
thread 2  processes tensor element 2
...
thread i  processes tensor element i
```

Every thread executes the same kernel function.

The thread index determines which data element it processes.

> CUDA replaces explicit loop iterations with a logical collection of threads.


---

# Host and device

| Term | Meaning |
|---|---|
| host | the CPU and its memory |
| device | the GPU and its memory |
| host code | code executed by the CPU |
| device code | code executed by the GPU |
| kernel | a device function launched across many threads |

Host and device memory are separate address spaces in the basic CUDA model.


---

# Lifecycle of a simple CUDA program

<div class="cuda-flow">
  <div class="cpu-step"><b>1</b><span>Create host input</span><small>CPU</small></div>
  <div class="flow-arrow">→</div>
  <div class="gpu-step"><b>2</b><span>Allocate device memory</span><small>CPU calls CUDA</small></div>
  <div class="flow-arrow">→</div>
  <div class="gpu-step"><b>3</b><span>Copy input to device</span><small>CPU initiates transfer</small></div>
  <div class="flow-arrow">→</div>
  <div class="gpu-step"><b>4</b><span>Launch kernel</span><small>GPU executes</small></div>
</div>

<div class="cuda-flow second-row">
  <div class="gpu-step"><b>5</b><span>Copy result to host</span><small>CPU initiates transfer</small></div>
  <div class="flow-arrow">→</div>
  <div class="cpu-step"><b>6</b><span>Verify result</span><small>CPU</small></div>
  <div class="flow-arrow">→</div>
  <div class="cpu-step"><b>7</b><span>Release allocations</span><small>CPU calls CUDA</small></div>
</div>


---

# A minimal CUDA source file

**Mixed host and device source**

```cpp
#include <cuda_runtime.h>

__global__ void my_kernel(float* data, int n)
{
    // Executed on the GPU.
}

int main()
{
    // Executed on the CPU.
    my_kernel<<<1, 1>>>(nullptr, 0);
    cudaDeviceSynchronize();
}
```

A `.cu` file may contain both host and device code.


---

# Compiling CUDA code

**Host shell command**

```bash
nvcc program.cu -o program
./program
```

`nvcc` coordinates the compilation of:

- host code for the CPU
- device code for the GPU
- CUDA runtime calls used by the host program

For now, treat `nvcc` as the compiler driver for a mixed CPU–GPU program.


---

# A kernel function

**Device (GPU) code**

```cpp
__global__ void copy_kernel(
    const float* input,
    float* output,
    int n)
{
    // Each thread executes this function.
}
```

`__global__` means:

- the function executes on the device
- host code can launch it
- the function returns `void`


---

# The logical execution hierarchy

```text
Grid
├── Block 0
│   ├── Thread 0
│   ├── Thread 1
│   └── ...
├── Block 1
│   ├── Thread 0
│   ├── Thread 1
│   └── ...
└── ...
```

The grid represents one kernel launch. Threads are grouped into blocks.


---

# Launch configuration

**Host (CPU) code**

```cpp
copy_kernel<<<4, 256>>>(d_input, d_output, n);
```

This launch creates:

- one grid
- four blocks
- 256 threads per block
- 1,024 logical threads in total

$$N_{threads}=N_{blocks}\times N_{threads/block}$$


---

# Built-in indices

Every thread can read CUDA-provided variables:

| Identifier | Meaning |
|---|---|
| `threadIdx.x` | thread position inside its block |
| `blockIdx.x` | block position inside the grid |
| `blockDim.x` | number of threads in a block |
| `gridDim.x` | number of blocks in the grid |

The `.x` component describes the first dimension.


---

# Computing a global thread index

**Device (GPU) code**

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

Example:

```text
blockDim.x  = 256
blockIdx.x  = 3
threadIdx.x = 17
```

$$i = 3 \times 256 + 17 = 785$$


---

# Indexing exercise

Assume `blockDim.x = 128`.

1. What is the global index of thread 5 in block 0?
2. What is the first global index in block 1?
3. Which thread processes element 300?
4. How many threads are launched with `<<<8, 128>>>`?


---

# Choosing the grid size

**Host (CPU) code**

```cpp
int block_size = 256;
int grid_size = (n + block_size - 1) / block_size;
```

This is integer ceiling division:

$$
N_{blocks}=\left\lceil\frac{n}{N_{threads/block}}\right\rceil
$$

For `n = 1000` and `block_size = 256`:

```text
grid_size = 4
threads launched = 4 × 256 = 1024
```


---

# The boundary check

**Device (GPU) code**

```cpp
__global__ void copy_kernel(
    const float* input,
    float* output,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        output[i] = input[i];
    }
}
```

The grid is rounded up to full blocks. Extra threads must not access elements beyond `n`.


---

# A complete kernel launch

**Host (CPU) code**

```cpp
int block_size = 256;
int grid_size = (n + block_size - 1) / block_size;

copy_kernel<<<grid_size, block_size>>>(
    d_input,
    d_output,
    n
);
```

The launch specifies the amount of parallel work. The kernel uses its indices to locate that work.


---

# One-dimensional and two-dimensional data

For a vector:

**Device (GPU) code**

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

For an image or matrix:

**Device (GPU) code**

```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```

Multidimensional indexing is a convenience. It makes code follow the logical shape of the data, but the same data can also be addressed through a linear index.


---

# Blocks and threads can execute in any order

CUDA does not guarantee block execution order.

```text
Block 3 may execute before Block 0.
Block 1 and Block 7 may execute concurrently.
The same grid can run on GPUs with different numbers of SMs.
```

Therefore, the first version of an algorithm should treat blocks as independent.

CUDA also does not guarantee the relative progress or completion order of individual threads. A program must not assume that thread `i` executes before thread `i + 1`.


---

# A practical initial block size

For the first kernels, use:

**Host (CPU) code**

```cpp
int block_size = 256;
```

Reasons:

- it is valid for common CUDA devices
- it is a conventional starting point for simple 1D kernels
- it keeps the first launch configuration easy to inspect

The best block size depends on the kernel and must eventually be measured.


---

# Kernel launches are asynchronous

**Host (CPU) code**

```cpp
copy_kernel<<<grid_size, block_size>>>(
    d_input, d_output, n
);

// The CPU may continue while the GPU is still working.
```

Explicit host-side wait:

**Host (CPU) code**

```cpp
cudaDeviceSynchronize();
```

Synchronization is required before the host relies on completed device work.


---

# Checking CUDA errors

**Host (CPU) code**

```cpp
copy_kernel<<<grid_size, block_size>>>(
    d_input, d_output, n
);

cudaError_t launch_error = cudaGetLastError();
cudaError_t runtime_error = cudaDeviceSynchronize();
```

- `cudaGetLastError()` detects launch configuration errors.
- `cudaDeviceSynchronize()` can reveal errors that occur during execution.
- Every CUDA runtime call returns a status value.


---

# Correctness before performance

A GPU result needs a trusted reference.

**Host (CPU) code**

```cpp
bool correct = true;

for (int i = 0; i < n; ++i) {
    if (h_output[i] != h_input[i]) {
        correct = false;
        break;
    }
}
```

The initial workflow is:

```text
implement → synchronize → check errors → compare results
```


---

# Practical session: manual copy kernel

You will:

1. compile and run a minimal CUDA program
2. allocate host and device arrays
3. copy the input from host to device
4. launch a manually written device-to-device copy kernel
5. copy the result back to the host
6. verify every output element
7. release all allocations


---

# The kernel used in the practical session

**Device (GPU) code**

```cpp
__global__ void copy_kernel(
    const float* input,
    float* output,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        output[i] = input[i];
    }
}
```

The exercise exposes indexing, launch configuration, boundary checks, and host–device coordination.


---

# Summary

- Host code launches device kernels.
- One kernel launch creates a grid of thread blocks.
- Each thread computes its own logical index.
- The grid size is rounded up to full blocks.
- A boundary check protects the valid data range.
- Kernel launches are asynchronous with respect to the host.
- Correctness and error checking come before optimization.

