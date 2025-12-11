---
title: "Neural Architecture Search Part 2: Search Spaces and Strategies"
date: 2025-07-20
categories: [Efficient Deep Learning, Architecture Design]
tags: [NAS, neural-architecture-search, search-space, optimization, deep-learning]
pin: false
---

Now that we understand the foundations of neural architecture search, we can tackle the critical question: **how do we actually search for good architectures?** In this part, we'll explore search spaces—the universe of possible architectures—and the strategies we can use to navigate them.

## What is Neural Architecture Search?

Neural Architecture Search is an automated approach to discovering neural network architectures that are optimized for specific objectives: high accuracy, low latency, minimal memory, or some combination of these.

The NAS problem has three components:

1. **Search Space**: The set of all possible architectures we can build
2. **Search Strategy**: The algorithm we use to explore the search space
3. **Performance Estimation**: How we evaluate architectures without training them from scratch (which would be prohibitively expensive)

Let's explore each component in depth.

## Designing the Search Space

The search space defines what architectures are possible. A well-designed search space should be:

- **Large enough** to contain good architectures
- **Small enough** to be searchable in reasonable time
- **Structured** to encode human knowledge and constraints

### Macro Search Space

The macro search space makes high-level decisions about overall architecture structure:

- **Number of stages**: How many large groups of layers? (typically 3-5)
- **Resolution per stage**: How does spatial resolution change? (typically halves at each stage)
- **Channels per stage**: How many channels in each stage?
- **Depth per stage**: How many blocks per stage?
- **Connections**: How are stages connected? (linear, residual, dense)

### Micro Search Space

The micro search space focuses on the building blocks used within stages:

- **Operation types**: What operations can we use? (convolution sizes, pooling, skip connections)
- **Channel multipliers**: What channel dimensions?
- **Kernel sizes**: $1 \times 1$, $3 \times 3$, $5 \times 5$, or $7 \times 7$?
- **Groups**: For grouped/depthwise convolutions, how many groups?

![Search Space Visualization](/assets/img/Neural_Architecture_Search/search_space_0.png){: .w-75 .shadow}

### Constraining the Search Space

Without constraints, the search space is infinite. We constrain it using:

- **Hardware constraints**: Latency budget, memory limits, energy consumption
- **Architectural constraints**: Maximum depth, channel ranges, operation types
- **Fairness constraints**: Preventing trivial solutions like "just use bigger channels"

## Search Strategies

Once we've defined the search space, we need a strategy to explore it. Different strategies have different trade-offs.

### Strategy 1: Random Search

The simplest approach: randomly sample architectures and evaluate them.

**Pros:**
- Easy to implement
- Provides a baseline
- Embarrassingly parallel

**Cons:**
- Wastes time on poor architectures
- No learning from previous evaluations
- Requires many evaluations to find good architectures

### Strategy 2: Reinforcement Learning-Based Search

Use a reinforcement learning controller to learn which design decisions lead to good architectures.

The process:
1. The controller makes sequential decisions: "add a conv layer", "use kernel size 3", "set channels to 64"
2. A child network is built and trained
3. The validation accuracy is used as reward
4. The controller learns to make better decisions

This is more efficient than random search because the controller learns patterns about what works.

### Strategy 3: Evolutionary Algorithms

Treat architectures like organisms in an evolving population:

1. Start with a random population of architectures
2. Evaluate each architecture
3. Keep the best-performing ones
4. Create offspring by mutating and recombining good architectures
5. Repeat

Evolution can explore diverse regions of the search space and often finds novel solutions that human designers wouldn't consider.

### Strategy 4: Gradient-Based Search

Treat architecture search as a differentiable optimization problem. Rather than discrete choices (e.g., "use a 3×3 conv"), use continuous relaxations where we maintain a probability distribution over operations and optimize it with gradient descent.

This is dramatically faster than discrete searches because gradient-based optimization is extremely efficient.

## Performance Estimation: The Critical Speedup

Here's the problem with NAS: if we train each candidate architecture from scratch to convergence, the search takes months even with thousands of GPUs.

**Solution**: Use tricks to estimate performance without full training.

### Early Stopping

Train architectures for fewer epochs. Final accuracy correlates with early accuracy, so we can discard obviously bad architectures quickly.

### Weight Sharing

Train a single large "super-network" that contains all possible architectures as sub-networks. Each candidate architecture reuses weights from the super-network rather than training from scratch.

The accuracy rank of architectures trained from scratch is usually preserved even with shared weights, so we can use this for comparison.

### Zero-Cost Proxies

Estimate accuracy without any training using properties of the architecture itself:

- **Synaptic saliency**: Are weight gradients large or small?
- **Spectral properties**: Properties of the weight matrix spectrum
- **Input-output distance**: How different are activations across layers?

These metrics correlate with final accuracy and can be computed in seconds.

## Hardware-Aware NAS

One of the most important developments in recent years is hardware-aware NAS: considering the target hardware (GPU, mobile CPU, edge device) during search.

Different architectures have different latency profiles on different hardware:

- A highly parallel architecture might be fast on GPU but slow on CPU
- Operations that are theoretically efficient might have poor kernel implementations
- Memory bandwidth becomes a bottleneck differently on different devices

The solution is to include actual hardware latency measurements in the search objective.

![Hardware-Aware Search Space](/assets/img/Neural_Architecture_Search/hardware_aware_0.png){: .w-75 .shadow}

Rather than optimizing purely for accuracy, we optimize for **accuracy subject to a latency constraint**:

$$\text{maximize} \quad \text{Accuracy}$$
$$\text{subject to} \quad \text{Latency}_{\text{target hardware}} < \text{Budget}$$

This is what separates practical NAS from academic NAS—we design for real deployment scenarios.

## The NAS Workflow

Putting it together, a typical NAS pipeline looks like:

1. **Define search space**: What operations and architectural choices are possible?
2. **Choose search strategy**: Random, RL, evolutionary, gradient-based, etc.
3. **Set constraints**: Hardware targets, computational budgets
4. **Run search**: Explore the search space using the strategy
5. **Validate top candidates**: Train the best architectures from scratch for final evaluation
6. **Deploy**: Use the discovered architecture in production

## Why This Matters

Manual architecture design requires years of experience and intuition. NAS democratizes architecture discovery:

- Teams without expert architects can design state-of-the-art models
- We can rapidly adapt to new constraints (new hardware, new latency budgets)
- We can discover architectures that humans might never think of
- The same approach works across domains: vision, language, generative models

In Part 3, we'll see concrete examples of NAS applications and discover some of the surprising architectures that NAS has found.

---

**Series Navigation:**
- [Part 1: Foundations and Building Blocks]({% post_url 2025-07-15-Neural-Architecture-Search-Part1-Foundations %})
- **Part 2**: Search Spaces and Strategies (this post)
- [Part 3: Applications and Real-World Impact]({% post_url 2025-07-25-Neural-Architecture-Search-Part3-Applications %})
- [Part 4: Efficient Estimation Strategies]({% post_url 2025-07-30-Neural-Architecture-Search-Part4-Efficient-Estimation %})
- [Part 5: Hardware-Aware NAS and Co-Design]({% post_url 2025-08-04-Neural-Architecture-Search-Part5-Hardware-Codesign %})

---

**References:**
- [MIT 6.5940: TinyML and Efficient Deep Learning](https://hanlab.mit.edu/courses/2024-fall-65940) - Lecture 7: Neural Architecture Search Part I (Fall 2024)
- [Neural Architecture Search: A Survey](https://arxiv.org/abs/1808.01377) (Elsken et al., 2018) - Comprehensive NAS survey
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) (Tan & Le, ICML 2019) - Automated scaling of architecture
- [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) (Liu et al., ICLR 2019) - Gradient-based NAS
