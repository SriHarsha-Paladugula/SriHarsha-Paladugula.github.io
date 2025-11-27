---
title: "Understanding GPU Memory Hierarchy: A Deep Dive into Modern Architecture"
date: 2024-12-15 14:00:00 +0800
categories: ["Efficient Deep Learning", "Model Efficiency"]
tags: ["Deep Learning", "GPU Architecture", "Memory Hierarchy", "Performance Optimization", "Hardware", "NVIDIA A100"]
description: A comprehensive guide to GPU memory types, access patterns, and their impact on deep learning computations
math: true
---

Modern deep learning depends heavily on GPUs, but understanding **how** GPUs manage memory is crucial for optimizing model performance. This guide explores the memory hierarchy in GPUs, focusing on the NVIDIA A100 architecture, and explains how different memory types impact computation speed and efficiency.

## What is GPU Memory Hierarchy?

GPU memory hierarchy is a structured system of different memory types, each with distinct characteristics in terms of speed, capacity, and energy consumption. Think of it like a library system:

- **Registers**: Your desk (immediate access, tiny capacity)
- **Shared Memory**: Books on your desk (very fast, limited space)
- **L2 Cache**: Bookshelf in your room (fast, moderate space)
- **Global Memory/HBM**: Library shelves (large capacity, slower access)

Understanding this hierarchy is essential because **memory access time often dominates computation time** in deep learning workloads.

## Why Memory Hierarchy Matters for Deep Learning

**The Performance Paradox**:
Modern GPUs can perform trillions of operations per second, but if data isn't available when needed, those compute cores sit idle. This is the **memory bottleneck problem**.

**Key Insight**: A model that efficiently uses the memory hierarchy can run 10-100× faster than one that doesn't, even with identical FLOPs (floating-point operations).

**Real-World Impact**:
- **Training time**: Hours vs days
- **Inference latency**: 10ms vs 100ms  
- **Energy consumption**: Dollars vs cents per inference
- **Deployment feasibility**: Runs on device vs requires cloud

## GPU Memory Types: The Complete Hierarchy

Let's explore each memory type in the GPU, from fastest to slowest, using the NVIDIA A100 as our reference architecture.

### 1. Registers: The Fastest Memory

**What are Registers?**
Registers are the smallest, fastest memory units directly accessible by GPU cores. Each thread has its own private set of registers.

**Characteristics**:
- **Access Speed**: Fastest - essentially instantaneous
- **Access Time**: 1-2 cycles
- **Energy Consumption**: Lowest - minimal power per access
- **Capacity**: Limited to several KB per thread block
- **Scope**: Private to individual threads

**How Registers Work**:
Think of registers as the calculator buttons you're currently pressing. Every intermediate calculation during a computation gets stored here temporarily.

**Example in Transformers**:
During self-attention computation, when calculating the dot product between a query vector and key vector:

$$
\text{score} = q_i \cdot k_j = \sum_{d=1}^{d_k} q_i^{(d)} \times k_j^{(d)}
$$

Each individual multiplication ($q_i^{(d)} \times k_j^{(d)}$) and the running sum are stored in registers.

**Why This Matters**: 
Keeping data in registers avoids expensive memory reads. Well-optimized CUDA kernels maximize register usage for frequently accessed values.

**Limitations**:
- Very limited capacity (typically 64KB per SM)
- If a thread needs more registers than available, "register spilling" occurs (data moves to slower memory)
- Register pressure can limit parallelism

---

### 2. Shared Memory (L1 Cache): Fast Collaborative Storage

**What is Shared Memory?**
Shared memory is on-chip memory shared by all threads within a thread block. It's explicitly managed by programmers for optimal performance.

**Characteristics**:
- **Access Speed**: Very fast
- **Access Time**: 4-8 cycles
- **Energy Consumption**: Low
- **Capacity**: Up to 100 KB per SM (Streaming Multiprocessor)
- **Scope**: Shared within a thread block

**How Shared Memory Works**:
Imagine a whiteboard in a team meeting room. All team members (threads in a block) can read and write to it, enabling fast data sharing without going to the main database (global memory).

**Example in Transformers**:
During multi-head attention, the Query (Q), Key (K), and Value (V) matrices are loaded into shared memory:

```python
# Conceptual CUDA-like pseudocode
__shared__ float shared_Q[BLOCK_SIZE][D_MODEL];
__shared__ float shared_K[BLOCK_SIZE][D_MODEL];

# Load Q and K into shared memory
shared_Q[threadIdx.x] = Q[blockIdx.x * BLOCK_SIZE + threadIdx.x];
shared_K[threadIdx.x] = K[blockIdx.x * BLOCK_SIZE + threadIdx.x];

# All threads in block can now access shared_Q and shared_K efficiently
```

**Why This Matters**:
Loading frequently reused data (like attention matrices) into shared memory once and reusing it multiple times dramatically reduces global memory accesses.

**Optimization Techniques**:
- **Tiling**: Break large matrix multiplications into tiles that fit in shared memory
- **Data reuse**: Load data once, use many times
- **Coalescing**: Coordinate memory access patterns across threads

**Limitations**:
- Limited capacity (100 KB on A100)
- Requires explicit programming (not automatic)
- Bank conflicts can reduce performance if not careful

---

### 3. L2 Cache: The Smart Intermediary

**What is L2 Cache?**
L2 cache is a larger, automatically managed cache shared across all Streaming Multiprocessors (SMs) on the GPU.

**Characteristics**:
- **Access Speed**: Moderate (faster than global memory, slower than shared memory)
- **Access Time**: 30-50 cycles
- **Energy Consumption**: Moderate
- **Capacity**: ~6 MB on A100 (shared across all SMs)
- **Scope**: Shared globally across the entire GPU

**How L2 Cache Works**:
L2 cache automatically stores recently accessed global memory data. Think of it as a "recently used items" shelf in a warehouse - items you grabbed recently are kept nearby for faster re-access.

**Example in Transformers**:
During the forward pass, if layer normalization parameters ($\gamma$ and $\beta$) are accessed multiple times:

$$
\text{LayerNorm}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

The $\gamma$ and $\beta$ parameters, once loaded from global memory, are cached in L2 for subsequent accesses.

**Why This Matters**:
L2 cache provides a significant speedup for frequently accessed data without requiring explicit programming (unlike shared memory).

**Cache Hit vs Cache Miss**:
- **Cache Hit**: Data found in L2 → 30-50 cycles
- **Cache Miss**: Must fetch from HBM → 200-400 cycles

**Optimization Strategy**:
Structure your algorithms to maximize data reuse within a short time window to keep data "hot" in the L2 cache.

---

### 4. Global Memory (Device Memory / HBM): The Main Storage

**What is Global Memory?**
Global memory, implemented as High Bandwidth Memory (HBM) on modern GPUs like the A100, is the largest memory pool accessible to all threads.

**Characteristics**:
- **Access Speed**: Slow compared to on-chip memory
- **Access Time**: 200-400 cycles
- **Energy Consumption**: High (proportional to data movement)
- **Capacity**: 40 GB or 80 GB on A100
- **Bandwidth**: 1555 GB/s on A100 (extremely high throughput)
- **Scope**: Accessible globally by all SMs

**How Global Memory Works**:
Global memory is like the main warehouse for all data. It holds model weights, activations, gradients, and input/output tensors.

**Example in Transformers**:
All major tensors are stored in global memory:

```python
# Model weights in global memory
W_Q = torch.randn(d_model, d_k)  # Query projection weights
W_K = torch.randn(d_model, d_k)  # Key projection weights
W_V = torch.randn(d_model, d_v)  # Value projection weights

# Activations during forward pass
Q = input @ W_Q  # Query computation
K = input @ W_K  # Key computation
V = input @ W_V  # Value computation

# Attention scores
scores = Q @ K.T / sqrt(d_k)  # Stored in global memory
```

**Why This Matters**:
While global memory is slow per-access, its massive bandwidth (1555 GB/s) means it can transfer huge amounts of data when access patterns are optimized.

**The Bandwidth vs Latency Trade-off**:
- **Latency**: Time to fetch a single piece of data (high)
- **Bandwidth**: Total data transfer rate (very high)
- **Key insight**: Batch operations together to maximize bandwidth utilization

**Memory Access Patterns**:

**Coalesced Access** (Good):
```
Thread 0: Read address 0
Thread 1: Read address 4
Thread 2: Read address 8
Thread 3: Read address 12
→ Single memory transaction
```

**Uncoalesced Access** (Bad):
```
Thread 0: Read address 0
Thread 1: Read address 100
Thread 2: Read address 200
Thread 3: Read address 300
→ Four separate memory transactions
```

---

### 5. High Bandwidth Memory (HBM): Understanding the Physical Layer

**What is HBM?**
HBM (High Bandwidth Memory 2 on A100) is the physical memory technology used for global memory. It's a 3D-stacked memory architecture directly connected to the GPU die.

**Characteristics**:
- **Access Speed**: High bandwidth, moderate latency
- **Access Time**: 100-150 cycles (optimized for throughput)
- **Energy Consumption**: Moderate to high
- **Capacity**: 40 GB or 80 GB on A100
- **Bandwidth**: 1555 GB/s (A100)

**HBM vs Traditional GDDR**:
HBM provides:
- 3× higher bandwidth than GDDR6
- 50% lower energy per bit transferred
- Shorter physical distance to GPU cores

**Why HBM is Critical for Deep Learning**:
Training large models like GPT-3 requires moving billions of parameters. HBM's massive bandwidth ensures the compute cores aren't starved for data.

**Example Calculation**:
For a Transformer with 1B parameters (4 GB at FP32):
- Forward pass: 4 GB read
- Backward pass: 4 GB read + 4 GB write = 12 GB total
- At 1555 GB/s: ~7.7ms just for memory transfers

**Memory Hierarchy Relationship**:
```
HBM (Physical) → Global Memory (Logical) → L2 Cache → Shared Memory → Registers
```

---

### 6. Texture Memory: Specialized for Spatial Data

**What is Texture Memory?**
Texture memory is a specialized read-only memory optimized for 2D spatial locality, primarily used for image-based computations.

**Characteristics**:
- **Access Speed**: Similar to global memory (200-400 cycles)
- **Access Time**: 200-400 cycles
- **Energy Consumption**: Moderate (efficient for spatial patterns)
- **Capacity**: Managed within global memory (HBM)
- **Specialty**: Hardware interpolation and caching for 2D data

**How Texture Memory Works**:
Texture memory includes hardware support for:
- **Spatial caching**: 2D locality-aware caching
- **Hardware interpolation**: Bilinear/trilinear filtering
- **Boundary handling**: Automatic clamping/wrapping

**Example in Vision Transformers**:
When processing image patches in Vision Transformers (ViT), texture memory can accelerate:

```python
# Image patch extraction with 2D spatial locality
image = load_image()  # 224x224x3
patches = extract_patches(image, patch_size=16)  
# Creates 196 patches (14x14 grid)
```

**Why This Matters**:
For vision-based models, texture memory's spatial caching can improve performance when accessing image data with 2D locality.

**When NOT to Use**:
Standard Transformer models (text-based) don't benefit from texture memory since they process 1D sequences without spatial structure.

---

### 7. Unified Memory: Bridging CPU and GPU

**What is Unified Memory?**
Unified Memory allows both CPU and GPU to access the same memory address space, simplifying programming but with performance trade-offs.

**Characteristics**:
- **Access Speed**: Variable (depends on data location)
- **Access Time**: Variable (includes potential PCIe transfer overhead)
- **Energy Consumption**: High (due to CPU-GPU transfers)
- **Capacity**: Can exceed GPU memory (uses system RAM)
- **Bandwidth**: Limited by PCIe (32 GB/s PCIe 4.0 x16)

**How Unified Memory Works**:
The CUDA runtime automatically migrates data between CPU and GPU memory on-demand:

```python
# Unified Memory allocation
data = torch.randn(10000, 10000).pin_memory()  # CPU memory
data = data.cuda()  # Migrates to GPU when accessed

# Automatic migration happens transparently
result = model(data)  # Data moved to GPU if needed
```

**Why This Matters**:
Unified Memory simplifies programming for:
- Models larger than GPU memory
- CPU-GPU collaborative algorithms
- Prototyping and debugging

**Performance Trade-offs**:

**Advantages**:
- Simpler programming model
- Handles out-of-memory scenarios gracefully
- Useful for sparse data structures

**Disadvantages**:
- Page faults cause significant overhead
- PCIe bandwidth is much lower than HBM bandwidth
- Unpredictable performance

**Example Scenario**:
Training a 100B parameter model on a 40GB A100:
- Parameters: 400 GB (FP32)
- Options: Model parallelism OR Unified Memory with CPU offloading
- Unified Memory allows training but with 10-50× slowdown for offloaded portions

**Best Practice**:
Use Unified Memory for convenience during development, but optimize for explicit memory management in production.

---

## Memory Hierarchy Comparison Table

| Memory Type | Access Time | Energy | Capacity (A100) | Bandwidth | Scope | Usage Pattern |
|-------------|-------------|--------|-----------------|-----------|-------|---------------|
| **Registers** | 1-2 cycles | Lowest | ~64 KB/SM | N/A | Thread-private | Scalar values, loop counters |
| **Shared Memory** | 4-8 cycles | Low | 100 KB/SM | ~20 TB/s | Thread block | Explicitly managed tiles |
| **L2 Cache** | 30-50 cycles | Moderate | 6 MB | ~4 TB/s | GPU-wide | Automatic caching |
| **Global Memory (HBM)** | 200-400 cycles | High | 40/80 GB | 1555 GB/s | GPU-wide | Model weights, activations |
| **Texture Memory** | 200-400 cycles | Moderate | Part of HBM | 1555 GB/s | GPU-wide | 2D spatial data |
| **Unified Memory** | Variable | Highest | System RAM | 32 GB/s (PCIe) | CPU+GPU | Oversubscription, prototyping |

**Key Insights from the Table**:

**Speed vs Capacity Trade-off**:
- Fastest memory (registers): Smallest capacity
- Largest memory (HBM): Slowest access time
- This is a fundamental hardware constraint

**Energy Efficiency**:
Moving data from HBM costs ~100× more energy than accessing shared memory. Algorithms that minimize data movement are critical for energy-efficient AI.

**Bandwidth Hierarchy**:
```
Shared Memory (20 TB/s) > L2 Cache (4 TB/s) > HBM (1.5 TB/s) > PCIe (32 GB/s)
```

---

## Memory Hierarchy in Action: Transformer Example

Let's trace how different memory types are used during a single attention computation:

**Step 1: Load Model Weights**
```python
# Weights stored in Global Memory (HBM)
W_Q, W_K, W_V = load_weights_from_hbm()
```
**Memory Access**: Global Memory → 200-400 cycles per weight

**Step 2: Compute Q, K, V Projections**
```python
# Matrix multiplication: input @ W_Q
Q = matmul(input, W_Q)
```
**Optimization**: 
- Load tiles of `W_Q` into Shared Memory
- Each thread block processes a tile
- Intermediate results stored in Registers

**Step 3: Compute Attention Scores**
$$
\text{scores} = \frac{QK^T}{\sqrt{d_k}}
$$

**Memory Flow**:
1. Load Q tile into Shared Memory (4-8 cycles)
2. Load K tile into Shared Memory (4-8 cycles)
3. Compute dot products using Registers (1-2 cycles per op)
4. Store results back to Global Memory (200-400 cycles)

**Step 4: Apply Softmax**
```python
attention_weights = softmax(scores)
```
**Cache Behavior**:
- If `scores` recently written, likely in L2 Cache (30-50 cycles)
- Avoids full global memory round-trip

**Step 5: Compute Weighted Values**
$$
\text{output} = \text{attention\_weights} \times V
$$

**Final Memory Access**:
- Attention weights: L2 Cache or Global Memory
- V matrix: Global Memory
- Output: Written to Global Memory

**Total Memory Hierarchy Usage**:
- **Registers**: ~90% of arithmetic operations
- **Shared Memory**: Tile loading for matrix multiplications
- **L2 Cache**: Frequently reused intermediate results
- **Global Memory**: Input/output and weight storage

---

## Why Device Memory and HBM Have Different Access Speeds

**Common Confusion**:
"Aren't Device Memory and HBM the same thing?"

**Technical Clarification**:

**Device Memory** (Logical Concept):
- Refers to the entire addressable memory space on the GPU
- Includes global memory, caches, and the memory hierarchy
- Access speed varies depending on cache hits/misses

**HBM** (Physical Hardware):
- The actual memory chips (HBM2 on A100)
- Provides the physical storage for global memory
- Raw bandwidth: 1555 GB/s

**Why Access Speeds Differ**:

**HBM Access (Best Case)**:
Data requested → Found in L2 Cache → 30-50 cycles

**Device Memory Access (Worst Case)**:
Data requested → L2 miss → Fetch from HBM → 200-400 cycles

**Analogy**:
- **HBM**: The highway system (high capacity, high speed limit)
- **Device Memory**: Your actual commute time (includes traffic, stoplights, parking)

**Practical Implication**:
Optimizing memory access patterns to maximize cache hits can make "device memory" perform 4-8× faster than raw HBM access.

---

## Memory Bandwidth: The Hidden Bottleneck

**Understanding the Roofline Model**:

GPU performance is limited by either:
1. **Compute Bound**: Limited by FLOPS capacity
2. **Memory Bound**: Limited by memory bandwidth

**A100 Specifications**:
- Peak Compute: 312 TFLOPS (FP16 with Tensor Cores)
- Memory Bandwidth: 1555 GB/s

**Arithmetic Intensity**:
$$
\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Transferred}}
$$

**Example: Matrix Multiplication**:
For $C = A \times B$ where $A, B, C$ are $N \times N$ matrices:

- **FLOPs**: $2N^3$ (multiply-add operations)
- **Bytes**: $3N^2 \times 4$ bytes (loading A, B, writing C in FP32)
- **Arithmetic Intensity**: $\frac{2N^3}{12N^2} = \frac{N}{6}$

**For N=1024**:
- Arithmetic Intensity = 170 FLOPs/byte
- This is **compute-bound** ✓

**For N=64**:
- Arithmetic Intensity = 10 FLOPs/byte
- This is **memory-bound** ⚠️

**Implication for Deep Learning**:
Small matrix multiplications (common in batch size 1 inference) are memory-bound, not compute-bound. Optimizing memory access is more important than adding more compute.

---

## Optimizing Memory Access: Best Practices

### 1. Minimize Global Memory Transfers

**Bad Pattern**:
```python
for i in range(num_layers):
    x = layer[i](x)  # Each layer reads/writes to global memory
```

**Good Pattern** (Operator Fusion):
```python
x = fused_layers(x)  # Multiple operations fused, fewer memory round-trips
```

**Speedup**: 2-3×

### 2. Maximize Data Reuse in Shared Memory

**Tiled Matrix Multiplication**:
```cuda
__global__ void matmul_tiled(float *A, float *B, float *C, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    // Load tile once, reuse TILE_SIZE times
    for (int tile = 0; tile < N/TILE_SIZE; tile++) {
        tileA[ty][tx] = A[...];  // Shared memory load
        tileB[ty][tx] = B[...];
        __syncthreads();
        
        // Reuse data in shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[ty][k] * tileB[k][tx];
        }
    }
}
```

**Benefit**: Load data once from global memory, reuse many times from shared memory.

### 3. Coalesce Memory Accesses

**Uncoalesced** (Slow):
```python
# Strided access pattern
for i in range(N):
    x[i * stride] = value  # Causes multiple memory transactions
```

**Coalesced** (Fast):
```python
# Contiguous access pattern
for i in range(N):
    x[i] = value  # Single memory transaction for multiple threads
```

**Speedup**: 5-10×

### 4. Use Mixed Precision Training

**FP32** (Standard):
- 4 bytes per parameter
- 1555 GB/s → 388B parameters/second

**FP16** (Half Precision):
- 2 bytes per parameter
- 1555 GB/s → 776B parameters/second

**Speedup**: 2× memory bandwidth efficiency

### 5. Enable Persistent Kernels

Keep data in faster memory across kernel launches:
```python
# Traditional: Data moves to global memory between kernels
output1 = kernel1(input)  # Write to global memory
output2 = kernel2(output1)  # Read from global memory

# Persistent kernel: Data stays in cache/shared memory
output = fused_kernel(input)  # No intermediate global memory transfer
```

---

## Memory Hierarchy and Model Architecture Design

**Design Principle**: Architecture choices should be informed by memory hierarchy constraints.

### Example 1: Attention Mechanism Memory Footprint

**Standard Attention**:
$$
\text{Memory} = O(N^2) \text{ for attention matrix}
$$

For sequence length $N=1024$:
- Attention matrix: $1024^2 \times 4$ bytes = 4 MB (fits in L2 cache ✓)

For sequence length $N=16384$:
- Attention matrix: $16384^2 \times 4$ bytes = 1 GB (exceeds cache, lives in global memory ⚠️)

**Solution**: Sparse attention, Flash Attention (memory-efficient algorithms)

### Example 2: Batch Size Selection

**Small Batch (Batch Size = 1)**:
- Low arithmetic intensity
- Memory-bound
- GPU underutilized

**Large Batch (Batch Size = 256)**:
- High arithmetic intensity
- Compute-bound
- GPU well-utilized

**Optimal Batch Size**: Balance memory capacity and arithmetic intensity

---

## Measuring Memory Performance in Practice

### Tool 1: NVIDIA Nsight Compute

```bash
ncu --metrics dram__bytes_read,dram__bytes_write python train.py
```

**Key Metrics**:
- `dram__bytes_read`: Bytes read from HBM
- `dram__bytes_write`: Bytes written to HBM
- `l2_cache_hit_rate`: L2 cache effectiveness
- `sm__sass_average_data_bytes_per_sector`: Memory access efficiency

### Tool 2: PyTorch Profiler

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    with_stack=True
) as prof:
    model(input)

print(prof.key_averages().table(sort_by="cuda_memory_usage"))
```

### Tool 3: Memory Bandwidth Utilization

```python
achieved_bandwidth = (bytes_transferred / elapsed_time)
memory_bandwidth_utilization = achieved_bandwidth / peak_bandwidth_HBM

# Goal: >70% utilization for memory-bound kernels
```

---

## Key Takeaways


**Summary of GPU Memory Hierarchy**: Modern GPUs use 7 distinct memory types, from ultra-fast registers (1-2 cycles) to slower unified memory (variable latency).

**Why Memory Hierarchy is Important**: Memory access time often dominates computation time in deep learning. Improving memory efficiency can have a greater impact than increasing compute power.

**Applications and Relevance**: Memory hierarchy affects all aspects of deep learning—training large models, optimizing inference, deploying on edge devices, and building energy-efficient AI systems.

**When to Focus on Memory Optimization**:
- When models exceed GPU memory capacity
- When inference latency is too high
- When training is slow
- When energy costs are significant

**Who Benefits from Memory Optimization**:
- ML engineers seeking better performance
- Researchers designing efficient architectures
- System architects deploying at scale
- Anyone working with large models (LLMs, Vision Transformers)

**Practical Strategies for Optimization**:
1. Minimize global memory transfers (operator fusion)
2. Maximize data reuse in shared memory (tiling)
3. Coalesce memory accesses (contiguous patterns)
4. Use mixed precision (FP16/BF16)
5. Understand arithmetic intensity (roofline model)

**Key Insights**:
- Memory is often the main bottleneck in modern GPUs
- Hierarchy matters: 100× speed difference between registers and global memory
- Energy efficiency: Data movement costs more energy than computation
- Architecture design should consider memory constraints

---

## What's Next?

Now that you understand GPU memory hierarchy, explore related topics:

- **Flash Attention**: Memory-efficient attention algorithms that reduce memory from $O(N^2)$ to $O(N)$
- **Model Quantization**: Reduce memory footprint and bandwidth requirements
- **Neural Network Pruning**: Decrease model size to fit in faster memory tiers
- **Tensor Core Programming**: Leverage specialized hardware for mixed-precision compute
- **CUDA Programming**: Write custom kernels optimized for memory hierarchy

**Deep Dive Resources**:
- NVIDIA CUDA C++ Programming Guide: Memory Hierarchy sections
- "Making Deep Learning Go Brrrr From First Principles" by Horace He
- NVIDIA Nsight Compute Documentation: Memory profiling

---

**References:**
- [MIT 6.5940: TinyML and Efficient Deep Learning Computing (Fall 2024)](https://hanlab.mit.edu/courses/2024-fall-65940)
- [NVIDIA A100 GPU Architecture Whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf)
- [CUDA C++ Programming Guide: Memory Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy)
- [Understanding the A100 GPU Memory Subsystem](https://developer.nvidia.com/blog/understanding-the-a100-gpu-memory-subsystem/)
- Harris, M. (2013). "Optimizing Parallel Reduction in CUDA" - NVIDIA Developer Blog
- "Roofline: An Insightful Visual Performance Model" - Williams et al., Communications of the ACM 2009
