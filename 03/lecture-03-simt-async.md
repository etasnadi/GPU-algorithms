---
marp: true
theme: minimal-white
paginate: true
size: 16:9
footer: GPU Programming for AI · Lecture 3
---

<!-- _paginate: false -->
<!-- _footer: "" -->

# SIMT Execution on NVIDIA GPUs

## SMs, execution groups, memory hierarchy, and synchronization

GPU Programming for AI  
Lecture 3 · 90 minutes


---

<!-- _class: compact -->

# Learning outcomes

By the end of this lecture, you should be able to:

- distinguish SIMD from SIMT
- connect grids and blocks to SM execution
- explain thread-group scheduling on an SM
- describe the GPU memory hierarchy
- use `__syncthreads()` correctly
- explain the lack of a general grid-wide barrier
- outline a block-level reduction


---

# SIMD

**Single Instruction, Multiple Data** applies one vector instruction to several lanes.

```text
instruction: add

lane 0: a[0] + b[0]
lane 1: a[1] + b[1]
lane 2: a[2] + b[2]
lane 3: a[3] + b[3]
```

The instruction explicitly operates on a vector of values.

Inactive lanes may be masked.


---

# SIMT

**Single Instruction, Multiple Threads** groups independent logical threads for execution.

**Device (GPU) code**

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
output[i] = input[i] + bias;
```

Each thread has its own:

- index and register state
- data values
- control-flow decisions

The programmer writes scalar thread code. The hardware groups thread operations for execution.


---

# Warp and lane

A **warp** is the basic scheduling and execution group on an NVIDIA GPU.

The hardware issues an instruction for a warp, rather than issuing an independent instruction for one thread.

An NVIDIA warp contains **32 threads**.

The individual positions inside a warp are called **lanes**:

```text
warp: lane 0, lane 1, lane 2, ... lane 31
```


---

# A block is divided into warps

For a one-dimensional block with 128 threads:

```text
warp 0: threadIdx.x   0–31
warp 1: threadIdx.x  32–63
warp 2: threadIdx.x  64–95
warp 3: threadIdx.x 96–127
```

On current NVIDIA CUDA GPUs, a warp contains 32 threads.

The final warp may contain inactive lanes when the block size is not a multiple of 32.


---

# Branching and the active mask

**Device (GPU) code**

```cpp
if ((threadIdx.x % 2) == 0) {
    output[i] = 2.0f * input[i];
} else {
    output[i] = input[i];
}
```

The **active mask** records which lanes participate in the currently issued instruction.

- Active lanes execute the instruction together in lockstep.
- Inactive lanes are disabled for that instruction and produce no result.


---

# SIMD and SIMT

| SIMD | SIMT |
|---|---|
| one vector instruction | many logical threads |
| explicit vector lanes | scalar thread program |
| vector registers | per-thread logical state |
| instruction controls lane mask | control flow determines active threads |

SIMT gives each thread a CUDA identity such as `threadIdx.x`.

The warp remains an important hardware execution group.


---

<!-- _class: compact -->

# Streaming Multiprocessors

A GPU contains several **Streaming Multiprocessors**, or SMs responsible for executing instructions.

Main components of the SM:
- Multiple execution units (**SM sub partitions** -- SMSPs). 
    - An SMSP executes an instruction for a single warp.
- A **register file**. 
    - A private set of registers are held for each thread of the warp (e.g. 64x4 bytes) running on the SMSP.
- Integrated **shared memory** and **L1/TEX** resources
    - Shared memory is a small (~96 KiB) addressable private memory for the threads of a single block.
    - L1 is the level 1 (non-addressable) cache for the SM.

---

<!-- _class: compact -->

# Blocks are assigned to SMs

The GPU selects blocks using an internal scheduling policy and assigns each selected block to one SM.

A **resident block** is a block that has been assigned to an SM. All warps of the block become resident on the same SM.

- Depending on available SM resources, several blocks may be resident on one SM.
- One block cannot be resident on more than one SM.
- A block remains resident on the same SM until all its threads finish.

CUDA does not specify the block selection order or the exact SM assignment.

---

# The complete hierarchy

```text
Grid
├── Block 0 ── resident on SM 0
│   ├── Warp 0: threads 0–31
│   └── Warp 1: threads 32–63
│
└── Block 1 ── resident on SM 1
    ├── Warp 0: threads 0–31
    └── Warp 1: threads 32–63
```

Thread IDs are defined within a block.

Warps never contain threads from different blocks.


---

# The warp scheduler

An SM keeps several warps resident (from a single block or multiple blocks).

```text
warp 0: waiting for memory
warp 1: ready to execute the next instruction
warp 2: ready to execute the next instruction
warp 3: waiting for an operand
warp 4: waiting for an operand
warp 5: ready to execute the next instruction
...
```

Warps are divided between SMSPs.

Each SMSP has a **warp scheduler** that can select from the set of warps assigned to it.

Switching between resident warps helps the SM make progress while another warp waits.

---

# Residency is limited by resources

A block consumes SM resources:

- thread slots (maximum number of resident threads is limited)
- registers
- shared memory
- block slots (maximum number of resident blocks is limited)

Large resource requirements can reduce the number of simultaneously resident blocks.

---

# SIMT control flow

**Device (GPU) code**

```cpp
if ((threadIdx.x % 2) == 0) {
    output[i] = 2.0f * input[i];
} else {
    output[i] = input[i];
}
```

Threads in one warp may choose different branches.

The GPU executes the required paths with different active-thread masks.

This behavior is called **warp divergence**.


---

# SIMT does not promise instruction-level lock-step

The CUDA programming model groups threads into warps.

Correctness must not rely on an assumption that every warp thread always reaches every machine instruction at exactly the same time.

Since Volta, NVIDIA GPUs use **Independent Thread Scheduling**.

Use explicit synchronization whenever threads communicate.


---

<!-- _class: compact -->

# Memory hierarchy and scope

| Storage | Scope | Managed by |
|---|---|---|
| registers | one thread | compiler and hardware |
| shared memory | one block | CUDA program |
| L1/TEX cache | one SM | hardware |
| L2 cache | whole GPU | hardware |
| GPU memory | whole GPU | CUDA program and hardware |

Scope answers which threads can address the storage directly.

Caching does not create a safe communication mechanism by itself.


---

# Registers

**Device (GPU) code**

```cpp
float x = input[i];
float y = x * scale;
output[i] = y;
```

Values such as `x` and `y` typically live in the thread's logical register state.

The SM provides the physical register file.

High register use per thread can reduce the number of resident warps and blocks.

---

<!-- _class: compact -->

# Shared memory

**Device (GPU) code**

```cpp
// Uses dynamic shared memory, shared buffer size is defined in launch configuration.
extern __shared__ float shared[];

shared[threadIdx.x] = input[i];
```

Shared memory is:

- allocated for one block
- accessible by every thread in that block
- located on the block's SM
- valid for the lifetime of the block

It acts as a programmer-managed scratchpad.


---

<!-- _class: compact -->

# L1/TEX and shared memory

Both resources are associated with an SM (SMSPs share its resources).

| L1/TEX | Shared memory |
|---|---|
| hardware-managed cache | explicitly addressed storage |
| caches eligible memory accesses | filled and read by thread instructions |
| contents may be replaced automatically | contents remain until the block finishes |
| unsuitable as a synchronization protocol | designed for block collaboration |

Their exact physical organization and capacity depend on GPU architecture.


---

# L2 and GPU memory

```text
SM load/store request
        │
        ▼
    L1/TEX path (L1 miss stage)
        │
        ▼
       L2 (L2 miss stage)
        │
        ▼
   GPU memory
```

L2 is shared across the GPU's SMs.

Global memory load and store traffic passes through the L2 subsystem on its way to or from GPU memory.


---

# Who reads and writes L2?

Threads execute global load and store instructions.

The GPU memory subsystem then:

- checks the relevant cache path for a load
- obtains missing data from GPU memory
- moves store data toward L2 and GPU memory
- manages cache-line allocation and replacement

The program does not normally issue an instruction that means “write this value into L2.”


---

# Dependencies inside one thread

**Device (GPU) code**

```cpp
float x = input[i];
float y = 2.0f * x;
output[i] = y;
```

The operations have data dependencies inside one thread.


---

# A block-level barrier

**Device (GPU) code**

```cpp
shared[threadIdx.x] = input[i];

__syncthreads();

float neighbor = shared[threadIdx.x ^ 1];
```

There is a dependency between independently executing threads.

`__syncthreads()` waits until ALL participating threads in the block reach the barrier.

Writes before the barrier become visible to block threads after the barrier.

It DOES NOT synchronize threads of different blocks.


---

# Every thread must reach the block barrier

**Device (GPU) code, incorrect**

```cpp
if (threadIdx.x < 32) {
    __syncthreads();
}
```

Some threads in the block wait at the barrier while others never arrive.

The program may hang or behave unpredictably.

A conditional barrier is safe only when the condition evaluates identically for the entire block.


---

# Warp-level synchronization

`__syncwarp()` synchronizes the participating lanes of a warp.

**Device (GPU) code**

```cpp
// nask selects the thread-combination that participates in the operation.
// Only participating threads are synchronized. 
unsigned int mask = __activemask();

// Warp-local communication.

__syncwarp(mask);
```

The function makes warp-level synchronization explicit.

---

# Why blocks cannot use a normal global barrier

Not every grid block must be resident at the same time.

```text
resident blocks wait at a barrier
             ↓
they keep SM resources occupied
             ↓
waiting blocks cannot become resident
```

A software spin barrier can therefore deadlock.

An ordinary CUDA kernel provides no general `__syncthreads()` equivalent for the entire grid.

---

<!-- _class: compact -->

# Inter-block synchronization can be achieved with multiple kernel launches with cuda streams

**Host (CPU) code**

```cpp
reduce_blocks<<<grid_size, block_size,
                dynamic_shared_bytes>>>(
    d_input, d_partial, n
);

reduce_final<<<1, block_size,
               dynamic_shared_bytes>>>(
    d_partial, d_output, grid_size
);
```

The second kernel begins after the first kernel completes. Why? Both use the default **stream**.

Thus, all blocks in the first kernel are completed before executing any block of the second kernel starts.

Thus, all blocks in the second kernel can see the global memory writes made by any blocks of the first kernel.

---

# Kernels are launched in parallel

Kernel launches are normally asynchronous with respect to the host.

```cpp
cudaStream_t stream_a, stream_b;
cudaStreamCreate(&stream_a);
cudaStreamCreate(&stream_b);

kernel_a<<<grid_a, block_a, 0, stream_a>>>(d_a);
kernel_b<<<grid_b, block_b, 0, stream_b>>>(d_b);
```

Kernels submitted to different non-default streams are independent and **may execute concurrently**.

Concurrency is possible only when dependencies and available GPU resources permit it.

---

<!-- _class: compact -->

# Defining kernel dependencies with streams

Operations submitted to the same stream execute in order. This does not mean that we need to put all dependent kernel launches into the same stream. 

For a dependency between *different streams*, use a **CUDA event**:

```cpp
cudaEvent_t data_ready;
cudaEventCreate(&data_ready);

producer<<<grid, block, 0, stream_a>>>(d_data);
cudaEventRecord(data_ready, stream_a);

cudaStreamWaitEvent(stream_b, data_ready);
consumer<<<grid, block, 0, stream_b>>>(d_data);
```

```text
stream_a:  producer ── record event
                            │
stream_b:             wait event ── consumer
```

The wait delays `consumer` without blocking the host thread.

---

# Fine-grained synchronizations

These functions block the calling host thread, but have different scopes:

| Function | The host waits until... |
|---|---|
| `cudaDeviceSynchronize()` | all preceding work on the device completes |
| `cudaStreamSynchronize(stream)` | all preceding work in `stream` completes |
| `cudaEventSynchronize(event)` | the recorded `event` completes |

Use the narrowest synchronization scope that provides the result needed by the host.

Unlike `cudaStreamWaitEvent()`, all three calls above make the **host** wait.

---

# Fine-grained synchronization example

```cpp
cudaEvent_t done;
cudaEventCreate(&done);

kernel_a<<<grid, block, 0, stream>>>(d_data);
cudaEventRecord(done, stream);       // checkpoint after A
kernel_b<<<grid, block, 0, stream>>>(d_data);

cudaEventSynchronize(done);    // A is done; B may still run
cudaStreamSynchronize(stream); // A and B are done
cudaDeviceSynchronize();       // all preceding device work is done
```

An event marks one precise point in a stream. Work submitted after that point does not have to complete before `cudaEventSynchronize()` returns.

---

<!-- _class: compact -->

# Be careful with the default stream

In **legacy default-stream mode**, stream 0 is a special synchronizing stream.

```cpp
kernel_a<<<grid, block, 0, stream_a>>>(d_a);
kernel_default<<<grid, block>>>(d_data);  // stream 0
kernel_b<<<grid, block, 0, stream_b>>>(d_b);
```

In legacy mode, `kernel_default` waits for earlier work in ordinary streams, and later work in those streams waits for `kernel_default`. This serializes the example.

---

<!-- _class: compact -->

# Stream-ordered memory allocation

`cudaMalloc()` is a synchronous host operation and is not ordered in a stream. Traditional allocation and deallocation may introduce device-wide synchronization.

When allocation is part of an asynchronous workflow, use the stream-ordered allocator:

```cpp
float* d_buffer;

cudaMallocAsync((void**)&d_buffer, bytes, stream);
kernel<<<grid, block, 0, stream>>>(d_buffer);
cudaFreeAsync(d_buffer, stream);
```

- Allocation, kernel use, and deallocation follow stream order.
- The pointer is safe for work ordered after the allocation and before the free.
- CUDA can reuse allocations through a driver-managed memory pool.
- Cross-stream use requires explicit ordering with events.

An allocation performed once, before concurrent work begins, can still use `cudaMalloc()` without reducing overlap inside the main execution phase.

---

<!-- _class: compact -->

# Asynchronous memory transfers

| `cudaMemcpy()` | `cudaMemcpyAsync()` |
|---|---|
| synchronous copy API | enqueues a copy into a stream |
| has no stream argument | takes an explicit stream argument |
| unsuitable for expressing overlap | returns control after enqueueing |
| synchronization depends on direction and memory type | ordered with other work in the stream |

```cpp
// Synchronous API
cudaMemcpy(d_data, h_data, bytes,
           cudaMemcpyHostToDevice);

// Stream-ordered asynchronous API
cudaMemcpyAsync(d_data, h_data, bytes,
                cudaMemcpyHostToDevice, stream);
```

A truly asynchronous host–device transfer requires pinned host memory. Use `cudaMallocHost()` or `cudaHostAlloc()` when transfer and computation should overlap.

---

<!-- _class: compact -->

# Virtual memory: address translation

Operating systems use **virtual addresses**. The MMU (Memory Management Unit) translates each virtual page to a physical memory frame using the TLB and page tables.

```text
virtual address                   physical address
[ virtual page | offset ]         [ frame | offset ]
         │                                ▲
         ▼                                │
        TLB ── hit ───────────────────────┘
         │ miss
         ▼
     page table ── resident mapping ──────┘
```

- Memory is managed in fixed-size **pages**; physical frames need not be contiguous.
- Each process gets its own virtual address space for isolation.
- The *page table stores* virtual-page-to-frame mappings and access permissions.
- The *TLB* (Translation Lookaside Buffer) caches recent translations to avoid a page-table lookup on every access.
- A contiguous virtual allocation may map to non-contiguous physical frames.

---

<!-- _class: compact -->

# Virtual memory: page faults and replacement

```text
access to a non-resident page
              │
              ▼
          page fault
              │
              ▼
 choose a free frame or victim page
       │                     │
       │              dirty victim?
       │                     └── write it to disk / swap
       ▼
load requested page ──► update page table ──► resume process
```

- Page faults are slow because storage access is much slower than RAM access.
- Replacement policies include FIFO, approximate LRU, and Clock/Second Chance.
- A clean victim can usually be discarded; a dirty victim must be saved first.
- A **pinned** page is locked in RAM and cannot be selected for replacement.

---

<!-- _class: compact -->

# Direct memory access: control and data flow

```text
1. CPU submits a transfer descriptor

   [ CPU ] ── {source, destination, size} ──► [ GPU copy engine ]

2. The copy engine performs the transfer

   [ pinned host RAM ] ═══ PCIe / NVLink ═══► [ GPU memory ]
                              ▲
                       controlled by DMA

3. The CPU continues and later observes completion
```

- The CPU initiates the operation but does not copy the individual bytes.
- The copy engine acts as a bus master and issues the memory transactions.
- Completion is observed through stream or event status and synchronization.
- DMA may overlap with kernel execution when dependencies and hardware resources permit it.

---

<!-- _class: compact -->

# Pinned host memory for DMA

```text
pageable allocation:

[ malloc buffer ] ── CPU staging copy ──► [ temporary pinned buffer ]
                                                  │
                                                 DMA
                                                  ▼
                                            [ GPU memory ]

pinned allocation:

[ cudaMallocHost buffer ] ─────── DMA ───────► [ GPU memory ]
```

- DMA needs host pages whose physical location remains stable for the transfer.
- `cudaMallocHost()` pins the pages so the OS cannot relocate or page them out.
- Pageable memory may require the extra staging copy shown above.
- Release pinned memory with `cudaFreeHost()` after all transfers using it complete.

Pinned memory is limited: excessive pinning leaves less pageable RAM available to the operating system.


---

# Example: block reduction

```text
8 values
  ↓ pairwise additions
4 partial sums
  ↓
2 partial sums
  ↓
1 block sum
```


---

# One result per block

**Device (GPU) code**

```cpp
if (tid == 0) {
    partial_sums[blockIdx.x] = shared[0];
}
```

Each block produces one partial sum in global memory.

The final sum requires another phase:

- copy the partial sums to the CPU and finish there, or
- launch another reduction kernel


---

# Practical session

The practical session builds a correct block-level reduction:

1. compute a CPU reference sum
2. load one input value per GPU thread
3. reduce values in shared memory
4. write one partial sum per block
5. finish the partial sums and verify the result

The required concepts are shared memory, block barriers, and multi-phase execution.


---

<!-- _class: compact -->

# Summary

- CUDA exposes grids, blocks, and threads.
- NVIDIA GPUs execute block threads in 32-thread warps on SMs.
- Warp schedulers select ready resident warps.
- Shared memory enables communication inside a block.
- `__syncthreads()` is a block barrier and must be reached by all threads in a block.
- Ordinary kernels provide no general grid-wide barrier (we will return to this in advanced kernel programming).
- Multi-block reductions use separate globally ordered phases (multiple kernel launches).
- Kernels are executed in parallel. Serialization only happens in a single stream.
