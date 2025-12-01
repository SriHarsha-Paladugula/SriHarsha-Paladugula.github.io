---
title: "Mastering LLM Inference Parameters - Part 2B: Advanced Sampling Methods"
date: 2025-05-20 10:00:00 +0000
categories: ["LLM", "Inference"]
tags: ["inference", "top-p", "nucleus-sampling", "typical-sampling", "contrastive-search", "llm-parameters"]
math: true
---

In Part 2A, we explored foundational decoding strategies: greedy, beam search, and top-k sampling. We discovered that top-k's fixed cutoff doesn't adapt to the model's confidence—sometimes including too many low-probability tokens, sometimes excluding reasonable options.

This post introduces **adaptive sampling methods** that solve these limitations: top-p (nucleus) sampling, best-of-N, typical sampling, and contrastive search.

---

## The Limitation Recap

**Top-k Problem:**

```
High Confidence:           Low Confidence:
Token 1: 85%              Token 1: 12%
Token 2:  8%              Token 2: 11%
...                       ...
Top-40: 0.001% (noise)    Top-40: 2% (reasonable)

With k=40: Too many noise tokens | Potentially too restrictive
```

**What we need:** A method that adapts to the distribution's shape.

---

## Top-p Sampling (Nucleus Sampling)

**Approach:** Dynamically select the smallest set of tokens whose cumulative probability exceeds p.

### How It Works

**Step-by-step process:**

1. **Rank tokens** by probability (descending)
2. **Add tokens** until cumulative probability ≥ p
3. **Renormalize** within this "nucleus"
4. **Sample** from the selected tokens

**Visual Example (p=0.90):**

```
Cumulative probabilities:
  by doing projects: 28%  (cumulative: 28%)  ← Include
  through practice:  24%  (cumulative: 52%)  ← Include
  with courses:      15%  (cumulative: 67%)  ← Include
  to start small:    12%  (cumulative: 79%)  ← Include
  using books:        8%  (cumulative: 87%)  ← Include
  from mentors:       6%  (cumulative: 93%)  ← Include (crosses 90%)
  [Stop here - remaining tokens excluded]

Nucleus size: 6 tokens (dynamic!)
```

### Mathematics Behind Top-p

1. **Sort tokens:** $P(w_1) ≥ P(w_2) ≥ ... ≥ P(w_n)$

2. **Find nucleus:**

$$
\mathcal{V}_p = \min \left\{ k : \sum_{i=1}^{k} P(w_i) ≥ p \right\}
$$

3. **Renormalize:**

$$
P'(w_i) = \begin{cases}
\frac{P(w_i)}{\sum_{j \in \mathcal{V}_p} P(w_j)} & \text{if } w_i \in \mathcal{V}_p \\
0 & \text{otherwise}
\end{cases}
$$

### Why Top-p Adapts Better

**Example: High-confidence scenario**

```
Distribution:
  Paris: 92%
  France: 4%
  ...

With p=0.90:
  Nucleus = {Paris} (1 token only)
  Model correctly focuses on the obvious answer
```

**Example: Low-confidence scenario**

```
Distribution:
  could: 8%
  might: 7%
  should: 7%
  would: 6%
  ...

With p=0.90:
  Nucleus = {could, might, should, would, ...} (15+ tokens)
  Model explores multiple reasonable options
```

**Key Insight:** Top-p automatically adjusts the nucleus size based on the model's confidence.

### Parameter Selection Guide

| p Value | Nucleus Size | Use Case |
|---------|--------------|----------|
| 0.5 | Very small | Extremely focused |
| 0.75 | Small | Conservative |
| 0.90 | Medium | Standard (recommended) |
| 0.95 | Large | Creative tasks |
| 1.0 | All tokens | Full sampling |

### Practical Example

```python
# Hugging Face
outputs = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,  # Nucleus sampling
    max_length=100
)

# OpenAI API
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a tagline"}],
    temperature=0.8,
    top_p=0.92
)
```

---

## Best-of-N Sampling

**Approach:** Generate N independent samples and return the highest-scoring one.

### How It Works

1. Generate N complete sequences using sampling (Top-p, Top-k)
2. Score each sequence (log probability, perplexity, reward model)
3. Return the best one

```python
# Generate 10 samples, return best
samples = []
for _ in range(10):
    output = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        max_length=100
    )
    # Score by log probability
    score = model.compute_transition_scores(
        output, 
        normalize_logits=True
    ).sum()
    samples.append((output, score))

# Return highest scoring
best_output = max(samples, key=lambda x: x[1])[0]
```

### Characteristics

**Advantages:**
- **Quality control**: Gets best of multiple attempts
- **Diversity exploration**: Samples different possibilities
- **Flexible scoring**: Can use custom metrics

**Disadvantages:**
- **Expensive**: N× computation cost
- **Latency**: Must generate all N before returning
- **Overkill**: For simple tasks, single sample suffices

### Use Cases

- Creative writing with quality requirements
- Reinforcement learning from human feedback (RLHF)
- When single sample quality is insufficient
- A/B testing different outputs

---

## Typical Sampling (Locally Typical Sampling)

**Approach:** Sample tokens with "typical" information content rather than just high probability.

### The Core Insight

Not all high-probability tokens are equally informative. Consider:

```
Context: "The weather today is"

High probability but boring: "good" (common but uninformative)
Typical: "sunny" (moderately likely, more informative)
```

Typical sampling balances probability with information content.

### Mathematics

Entropy-based selection:

$$
\text{Typical Set} = \left\{ w : \left| -\log P(w) - H(P) \right| < \epsilon \right\}
$$

Where:
- $H(P) = -\sum P(w) \log P(w)$: Entropy of distribution
- $\epsilon$: Threshold for typicality
- Tokens with information content close to expected entropy

### Practical Example

```python
# Hugging Face implementation
outputs = model.generate(
    input_ids,
    do_sample=True,
    typical_p=0.9,        # Typical sampling threshold
    temperature=0.8,
    max_length=100
)
```

### When to Use

- Natural language generation
- Avoiding "obvious but boring" completions
- Better coherence than pure Top-p in some cases
- Dialogue systems (more natural responses)

**Research:** Meister et al. (2022) "Typical Decoding for Natural Language Generation"

---

## Contrastive Search

**Approach:** Balance probability with diversity by penalizing tokens similar to previously generated ones.

### How It Works

**Scoring Function:**

$$
\text{score}(w_i) = (1 - \alpha) \times P(w_i) - \alpha \times \max_{j < i} \text{sim}(w_i, w_j)
$$

Where:
- $P(w_i)$: Model's probability for token $w_i$
- $\text{sim}(w_i, w_j)$: Cosine similarity between token embeddings
- $\alpha$: Balance parameter (typically 0.6)

**Effect:** Tokens similar to already-generated ones get penalized, encouraging diversity.

### Practical Example

```python
# Hugging Face implementation
outputs = model.generate(
    input_ids,
    penalty_alpha=0.6,    # Contrastive search penalty
    top_k=4,              # Candidate pool size
    max_length=100
)
```

### Characteristics

**Advantages:**
- Reduces repetition naturally
- Maintains coherence
- Deterministic (no randomness in selection)
- No temperature parameter needed

**Disadvantages:**
- Requires computing embedding similarities
- Slightly slower than pure sampling
- May avoid valid repetitions

### Use Cases

- Long-form text generation
- Open-ended conversation
- When both quality and diversity matter
- Reducing repetitive patterns

**Research:** Su et al. (2022) "A Contrastive Framework for Neural Text Generation"

---

## Comprehensive Strategy Comparison

| Strategy | Deterministic | Quality | Diversity | Speed | Adaptive |
|----------|---------------|---------|-----------|-------|----------|
| Greedy | Yes | Medium | None | Fast | No |
| Beam Search | Yes | High | Low | Slow | No |
| Top-k | No | Medium | Medium | Fast | No |
| **Top-p** | No | Medium | High | Fast | **Yes** |
| Best-of-N | No | High | High | Very Slow | No |
| Typical | No | High | Medium | Fast | **Yes** |
| Contrastive | Yes | High | Medium | Medium | **Yes** |

### When to Choose Each

**Decision Tree:**

```
Need deterministic output?
├─ Yes: Greedy or Beam Search
│   ├─ Quality critical? → Beam Search
│   └─ Speed critical? → Greedy
│
└─ No: Sampling methods
    ├─ Have computation budget?
    │   ├─ Yes: Best-of-N
    │   └─ No: Continue
    │
    ├─ Long-form text?
    │   ├─ Yes: Contrastive Search
    │   └─ No: Continue
    │
    ├─ Need natural conversation?
    │   ├─ Yes: Typical Sampling
    │   └─ No: Top-p (default choice)
    │
    └─ Fixed creativity level? → Top-k
```

---

## Real-World Configuration Examples

### Chatbot Responses

**Goal:** Natural, varied conversation

**Configuration:**
```python
temperature = 0.7
top_p = 0.90        # Adaptive nucleus
# OR
typical_p = 0.9     # Natural information content
```

**Why:** Adapts to confidence—focused for factual, exploratory for open-ended.

---

### Creative Story Writing

**Goal:** Unique, engaging narratives

**Configuration:**
```python
temperature = 1.0
top_p = 0.95        # Broad exploration
# OR
penalty_alpha = 0.6  # Contrastive for long-form
```

**Why:** High creativity with adaptive diversity.

---

### Code Completion

**Goal:** Correct, idiomatic code

**Configuration:**
```python
temperature = 0.2
top_p = 0.85        # Focused on high-confidence
```

**Why:** Low temperature + tight nucleus = precision.

---

### Professional Email

**Goal:** Varied but appropriate

**Configuration:**
```python
temperature = 0.5
top_p = 0.88
```

**Why:** Moderate creativity, adaptive to context.

---

### Brainstorming

**Goal:** Maximum idea diversity

**Configuration:**
```python
# Option 1: Best-of-N
temperature = 0.9
top_p = 0.95
num_samples = 10

# Option 2: High temperature + presence penalty (Part 3)
temperature = 0.9
top_p = 0.92
presence_penalty = 0.7
```

**Why:** Generate multiple diverse candidates.

---

## Combining Temperature with Sampling

**Generation Pipeline:**

```
1. Model computes raw logits
         ↓
2. Apply temperature scaling
         ↓
3. Convert to probabilities
         ↓
4. Apply selection strategy (Top-k/Top-p/Typical)
         ↓
5. Sample from filtered distribution
```

**Configuration Matrix:**

| Task | Strategy | Temperature | Parameter |
|------|----------|-------------|-----------|
| Factual QA | Greedy | 0.0 | - |
| Translation | Beam | - | num_beams=5 |
| Code | Top-p | 0.2 | p=0.85 |
| Email | Top-p | 0.5 | p=0.88 |
| Marketing | Top-p | 0.8 | p=0.92 |
| Fiction | Top-p | 1.0 | p=0.95 |
| Chatbot | Typical | 0.7 | typical_p=0.9 |
| Long article | Contrastive | - | penalty_alpha=0.6 |

---

## Common Pitfalls

### Mistake 1: Setting p Too Low

**Problem:** p=0.5 for creative writing

**Issue:** Cuts off too many reasonable options

**Solution:** Use p=0.90-0.95 for most tasks

---

### Mistake 2: Using Both Top-k and Top-p

**Problem:** Setting both simultaneously

**Issue:** Most APIs apply top-k first, then top-p (unexpected filtering)

**Solution:** Choose **one** selection strategy

---

### Mistake 3: Ignoring Task Requirements

**Problem:** Using high temperature + broad nucleus for code

**Issue:** Syntax errors, incorrect logic

**Solution:** Match parameters to task needs (code needs precision)

---

## Key Takeaways

**Adaptive sampling advantages:**
- Top-p adjusts nucleus size based on confidence
- Typical sampling balances probability with information
- Contrastive search naturally reduces repetition

**Top-p strengths:**
- Most widely used adaptive method
- Works well across diverse tasks
- Simple to understand and tune
- Generally preferred over fixed top-k

**Advanced method use cases:**
- Best-of-N: When quality matters more than speed
- Typical: Natural conversation and dialogue
- Contrastive: Long-form, coherent text generation

**Practical guidelines:**
- Start with Top-p (p=0.9) + temperature (0.7) as baseline
- Adjust temperature first, then p value
- Use beam search only for tasks with "correct" answers
- Consider contrastive for repetition-prone tasks
- Test configurations on representative examples

**Configuration principles:**
- Lower temperature + lower p = More focused
- Higher temperature + higher p = More exploratory
- Match strategy to task constraints
- Don't over-complicate—simpler often works

---

## What's Next?

We've covered decoding strategies (Part 2A) and sampling methods (Part 2B). In **Part 3**, we'll explore **precision control parameters**:

- **Frequency penalty**: Penalize token repetition proportionally
- **Presence penalty**: Encourage topic diversity
- **Repetition penalty**: Unified anti-repetition mechanism
- **Stop sequences**: Control generation boundaries
- **Logit bias**: Manually boost/suppress specific tokens
- **Length controls**: Min/max token constraints
- **Complete workflows**: Combining all parameters effectively

These parameters provide surgical control over specific output characteristics beyond general sampling behavior.

---

**Series Navigation:**
- [Part 1: Temperature and Randomness Control]({% post_url 2025-05-10-LLM-Inference-Parameters-Part1-Temperature %})
- [Part 2A: Basic Decoding Strategies]({% post_url 2025-05-15-LLM-Inference-Parameters-Part2A-Basic-Strategies %})
- **Part 2B: Advanced Sampling Methods** (Current)
- [Part 3: Advanced Parameters and Practical Applications]({% post_url 2025-05-25-LLM-Inference-Parameters-Part3-Advanced-Parameters %})

**References:**
- Holtzman et al. (2019). "The Curious Case of Neural Text Degeneration." ICLR 2020. (Original nucleus sampling paper)
- Meister et al. (2022). "Typical Decoding for Natural Language Generation." arXiv:2202.00666.
- Su et al. (2022). "A Contrastive Framework for Neural Text Generation." arXiv:2202.06417.
- [Hugging Face - Generation Strategies Guide](https://huggingface.co/docs/transformers/generation_strategies)
- Fan et al. (2018). "Hierarchical Neural Story Generation." ACL 2018.
- [OpenAI API Reference - Chat Completions](https://platform.openai.com/docs/api-reference/chat/create)
