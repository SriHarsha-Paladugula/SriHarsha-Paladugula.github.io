---
title: "Neural Network Pruning Part 1: Why Pruning Matters"
date: 2025-05-12 10:00:00 +0800
categories: ["Efficient Deep Learning", "Model Efficiency"]
tags: ["Deep Learning", "Model Compression", "Pruning", "Neural Networks", "Efficiency"]
math: true
---

## The Growing Challenge of Model Size

Imagine building a house and ordering 10,000 bricks, but you only use 2,000 of them. The remaining 8,000 bricks sit in your yard, taking up space, costing money, and serving no purpose. This is exactly what happens with modern neural networks—they're massively over-parameterized.

### The Exponential Growth of AI Models

Deep learning models have grown exponentially over the years. Let's look at how model sizes have exploded:

<div align="center">
  <img src="/assets/img/Pruning/pruning_page_02.png" alt="Model size and accuracy comparison" />
  <p><em>Model size vs accuracy: Larger models don't always mean better performance</em></p>
</div>

**The Evolution of Model Sizes:**

| Model | Year | Parameters | Approximate Size |
|-------|------|-----------|------------------|
| LeNet-5 | 1998 | <1M | ~4 MB |
| AlexNet | 2012 | 61M | ~240 MB |
| VGG-16 | 2014 | 138M | ~528 MB |
| ResNet-50 | 2015 | 26M | ~100 MB |
| GPT-3 | 2020 | 175B | ~700 GB |
| GPT-4 | 2023 | ~1.7T | ~7 TB (estimated) |

### Why This Growth is Problematic

**The Supply-Demand Gap:**

<div align="center">
  <img src="/assets/img/Pruning/pruning_page_03.png" alt="GPU memory vs model size growth" />
  <p><em>Model sizes are growing faster than GPU memory capacity</em></p>
</div>

The graph above shows a critical problem: **model sizes are growing exponentially faster than available GPU memory**. Even the most advanced GPUs struggle to keep up:

- **A100 GPU (2020)**: 80 GB memory
- **GPT-3 (2020)**: 175B parameters ≈ 700 GB (in FP16)
- **MT-NLG (2021)**: 530B parameters ≈ 2.1 TB (in FP16)

This means we can't even load these models into a single GPU, let alone train or run inference efficiently.

## What is Neural Network Pruning?

**Pruning** is the process of removing unnecessary connections (weights) and neurons from a neural network to make it smaller and faster, without significantly hurting its performance.

### Inspiration from the Human Brain

Neural network pruning isn't just a computational trick—it's inspired by how our own brains develop:

<div align="center">
  <img src="/assets/img/Pruning/pruning_page_15.png" alt="Human brain synapse development" />
  <p><em>Synapse development in the human brain from birth to adulthood</em></p>
</div>

**Human Brain Development:**
- **Newborn**: ~2,500 synapses per neuron
- **2-4 years**: ~15,000 synapses per neuron (peak)
- **Adolescence**: Pruning begins
- **Adult**: ~7,000 synapses per neuron (53% reduction!)

The human brain naturally prunes connections that aren't frequently used. This process makes our brain more efficient without reducing our cognitive abilities. In fact, this pruning is essential for healthy brain development.

### The Simple Idea Behind Pruning

Think of a neural network like a city's road network:
- Some roads (connections) carry heavy traffic daily
- Other roads are rarely used
- Removing rarely-used roads saves maintenance costs without affecting most people's commutes

Similarly, in neural networks:
- Some weights contribute significantly to predictions
- Many weights are close to zero or contribute minimally
- Removing these "weak" connections reduces model size without hurting accuracy

## Why Should We Care About Pruning?

### 1. **Energy Efficiency**

<div align="center">
  <img src="/assets/img/Pruning/pruning_page_06.png" alt="Energy cost of operations" />
  <p><em>Energy cost comparison for different computing operations in 45nm CMOS</em></p>
</div>

Memory access is **incredibly expensive** in terms of energy:

| Operation | Energy Cost (pJ) | Relative Cost |
|-----------|------------------|---------------|
| 32-bit INT Addition | 0.1 | 1× |
| 32-bit FP Addition | 0.9 | 9× |
| 32-bit Register File | 1 | 10× |
| 32-bit INT Multiply | 3.1 | 31× |
| 32-bit FP Multiply | 3.7 | 37× |
| 32-bit SRAM Cache | 5 | 50× |
| 32-bit DRAM Memory | 640 | **6,400×** |

**Key Insight:** Reading from DRAM (external memory) costs **200 times more energy** than reading from on-chip SRAM cache. Smaller models fit in faster, more energy-efficient memory.

### 2. **Real-World Impact**

**Mobile Deployment:**
- Running a 1 billion parameter network at 20 Hz requires: $(20 \text{ Hz}) \times (1B) \times (640 \text{ pJ}) = 12.8 \text{ W}$
- This is **beyond the power budget** of most mobile devices
- A 10× pruned model would require only ~1.3W—much more feasible!

**Cloud Costs:**
- Smaller models mean more inferences per GPU
- Reduced memory bandwidth requirements
- Lower electricity costs for data centers

**Environmental Impact:**
- Training large models produces significant CO₂ emissions
- Pruned models require less compute, reducing carbon footprint
- More efficient inference at scale

### 3. **Practical Benefits**

✅ **Faster Inference**: Fewer operations mean faster predictions  
✅ **Lower Latency**: Critical for real-time applications  
✅ **Reduced Storage**: Easier model deployment and updates  
✅ **Better Accessibility**: Run powerful models on edge devices  
✅ **Cost Savings**: Fewer GPUs needed for serving models  

## How Effective is Pruning?

Pruning can dramatically reduce model size with minimal accuracy loss:

<div align="center">
  <img src="/assets/img/Pruning/pruning_page_17.png" alt="Pruning ratio vs accuracy" />
  <p><em>Accuracy loss vs pruning ratio showing that models can maintain accuracy even with 80-90% pruning</em></p>
</div>

The graph above shows that:
- Up to 50% of parameters can be pruned with **no accuracy loss**
- 70-80% pruning still maintains acceptable accuracy
- With proper fine-tuning (iterative pruning), even 90% pruning is possible!

**Real Results from Research:**

<div align="center">
  <img src="/assets/img/Pruning/pruning_page_21.png" alt="Pruning results table" />
  <p><em>Compression ratios achieved on popular neural networks</em></p>
</div>

| Network | Original Params | Pruned Params | Reduction | MAC Reduction |
|---------|----------------|---------------|-----------|---------------|
| AlexNet | 61M | 6.7M | **9×** | 3× |
| VGG-16 | 138M | 10.3M | **12×** | 5× |
| GoogLeNet | 7M | 2.0M | **3.5×** | 5× |
| ResNet-50 | 26M | 7.47M | **3.4×** | 6.3× |
| SqueezeNet | 1M | 0.38M | **3.2×** | 3.5× |

This means we can make AlexNet **9× smaller** while keeping the same accuracy on ImageNet!

## What is Pruning NOT?

Let's clarify some misconceptions:

❌ **Not about reducing model architecture complexity** (like switching from ResNet-50 to MobileNet)  
❌ **Not about quantization** (using lower precision like INT8 instead of FP32)  
❌ **Not about knowledge distillation** (training a smaller model to mimic a larger one)  

✅ **IS about selectively removing parameters from an existing, trained model**

## Real-World Applications

Pruning is not just academic—it's being used in production:

### Industry Adoption

**NVIDIA GPUs** (A100 and newer) have hardware support for structured sparsity:
- **2:4 sparsity pattern**: 50% of weights are zero
- Delivers **2× theoretical speedup**
- Achieves ~1.5× measured speedup on BERT inference

**MLPerf Results** (2024):
- Pruned Llama 2 70B achieved **2.5× speedup** while maintaining 99% accuracy
- Used depth pruning (80 → 32 layers) and width pruning (28,762 → 14,336 dimensions)

## Key Takeaways

1. **Model sizes are growing exponentially**, outpacing hardware improvements
2. **Memory access dominates energy consumption**, not computation
3. **Neural networks are highly redundant**—many parameters contribute little
4. **Pruning can reduce model size by 3-12×** without accuracy loss
5. **The human brain naturally prunes connections**, and it works remarkably well
6. **Industry is adopting pruning** with hardware support and production deployments

## What's Next?

Now that we understand **why** pruning matters, the natural questions are:

- **How** do we actually prune a neural network?
- **Which** connections should we remove?
- **What** patterns of pruning work best?

In [Part 2]({% post_url 2025-02-10-Neural-Network-Pruning-Part2-Pruning-Granularities %}), we'll explore different **pruning granularities**—from removing individual weights to entire channels—and understand the trade-offs between compression ratio and hardware efficiency.

---

**Series Navigation:**
- **Part 1: Why Pruning Matters** (Current)
- [Part 2: Pruning Granularities]({% post_url 2025-02-10-Neural-Network-Pruning-Part2-Pruning-Granularities %})
- [Part 3: Pruning Criteria]({% post_url 2025-02-17-Neural-Network-Pruning-Part3-Pruning-Criteria %})
- [Part 4: Advanced Techniques]({% post_url 2025-02-24-Neural-Network-Pruning-Part4-Advanced-Techniques %})

**References:**
- [Learning Both Weights and Connections for Efficient Neural Network](https://arxiv.org/abs/1506.02626) (Han et al., NeurIPS 2015)
- [Computing's Energy Problem (and What We Can Do About it)](https://ieeexplore.ieee.org/document/6757323) (Horowitz, IEEE ISSCC 2014)
- [Model Compression and Hardware Acceleration: A Comprehensive Survey](https://ieeexplore.ieee.org/document/9043731) (Deng et al., IEEE 2020)
