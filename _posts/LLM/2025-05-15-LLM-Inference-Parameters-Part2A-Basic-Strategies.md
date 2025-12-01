---
title: "Mastering LLM Inference Parameters - Part 2A: Basic Decoding Strategies"
date: 2025-05-15 10:00:00 +0000
categories: ["LLM", "Inference"]
tags: ["inference", "greedy-decoding", "beam-search", "top-k", "llm-parameters", "text-generation"]
math: true
---

In Part 1, we learned how temperature controls the "shape" of probability distributions—making the model more focused or more exploratory. But temperature alone doesn't answer a critical question: **Which tokens should we even consider?**

Imagine you're at a restaurant. Temperature is like deciding how adventurous you feel (stick with your favorite dish or try something new?). But **token selection strategies** determine which section of the menu you're looking at in the first place.

This post explores the foundational decoding strategies: greedy decoding, beam search, and top-k sampling.

---

## The Token Selection Problem

Let's revisit the generation process with a concrete example.

**Prompt:** "The best way to learn programming is"

**Model Output (probability distribution):**

```
by doing projects:        28%
through practice:         24%
with online courses:      15%
to start small:           12%
using books:               8%
from mentors:              6%
to give up immediately:    0.8%
to eat pizza:              0.5%
...
[43,000 more tokens with decreasing probability]
```

**The Challenge:** Do we:
1. Always pick the top token? (Deterministic but boring)
2. Sample from all 50,000 tokens? (Creative but chaotic)
3. Pick a middle ground? (This is where selection strategies help)

---

## Strategy 1: Greedy Decoding

**Approach:** Always select the highest-probability token.

### How It Works

```python
# Pseudocode
def greedy_decode(probabilities):
    return argmax(probabilities)  # Pick token with highest probability
```

**Example:**
```
Probabilities:
  by doing projects: 28%  ← Always chosen
  through practice:  24%
  ...

Output: "by doing projects"
Every. Single. Time.
```

### Characteristics

**Advantages:**
- **Fast**: No sampling computation needed
- **Deterministic**: Same input always produces same output
- **Predictable**: Useful for reproducibility

**Disadvantages:**
- **Repetitive**: Falls into loops easily
- **Generic**: Produces "average" text
- **Boring**: No variation across runs

### Optimal Use Cases

**When to use greedy decoding:**
- Factual question answering (you want THE correct answer)
- Code generation (syntax errors are costly)
- Translation (when exactness matters)
- Data extraction (structured output required)

**Example Application:**
```python
# Math problem solving
response = generate(
    prompt="What is 15 * 24?",
    temperature=0.0,  # Greedy decoding
    max_tokens=10
)
# Output: "360" (always correct)
```

---

## Strategy 2: Beam Search

**Approach:** Maintain multiple candidate sequences simultaneously and select the most probable complete sequence.

### How It Works

Unlike greedy decoding (which keeps only 1 candidate) or sampling (which explores randomly), beam search maintains **k** parallel hypotheses and picks the best overall sequence.

**Step-by-step process:**

1. **Start with k beams** (candidate sequences)
2. **For each beam**, generate top k next tokens
3. **Score all k × k combinations** by cumulative log probability
4. **Keep the top k** most probable sequences
5. **Repeat** until all beams end or max length reached
6. **Return highest-scoring** complete sequence

**Visual Example (beam_width=3):**

```
Prompt: "The capital of France is"

Step 1: Generate from prompt
  Beam 1: "The capital of France is Paris" (log_prob: -0.5)
  Beam 2: "The capital of France is the" (log_prob: -1.2)
  Beam 3: "The capital of France is located" (log_prob: -1.8)

Step 2: Expand each beam
  From Beam 1 "...Paris":
    → "...Paris." (log_prob: -0.6)
    → "...Paris," (log_prob: -0.7)
    → "...Paris and" (log_prob: -1.5)
  
  From Beam 2 "...the":
    → "...the capital" (log_prob: -1.5)
    → "...the city" (log_prob: -1.8)
    → "...the main" (log_prob: -2.0)
  
  From Beam 3 "...located":
    → "...located in" (log_prob: -2.0)
    → "...located at" (log_prob: -2.3)
    → "...located near" (log_prob: -2.5)

Step 3: Keep top 3 overall
  Beam 1: "...Paris." (-0.6) ✓
  Beam 2: "...Paris," (-0.7) ✓
  Beam 3: "...the capital" (-1.5) ✓
  [Continue until completion]

Final: Return "The capital of France is Paris."
```

### Mathematics Behind Beam Search

**Scoring Function:**

For a sequence $y = (y_1, y_2, ..., y_T)$:

$$
\text{score}(y) = \log P(y) = \sum_{t=1}^{T} \log P(y_t | y_{<t})
$$

**Length Normalization** (prevents bias toward shorter sequences):

$$
\text{score}_{\text{normalized}}(y) = \frac{1}{T^{\alpha}} \sum_{t=1}^{T} \log P(y_t | y_{<t})
$$

Where:
- $\alpha$: Length penalty coefficient (typically 0.6-1.0)
- $\alpha = 0$: No normalization (favors shorter sequences)
- $\alpha = 1$: Full normalization (uniform treatment)
- $0 < \alpha < 1$: Balanced approach

### Parameter Selection Guide

| Beam Width | Effect | Use Case | Computation Cost |
|------------|--------|----------|------------------|
| k = 1 | Greedy (deterministic) | Fast inference | 1× |
| k = 3-5 | Conservative exploration | Translation, summarization | 3-5× |
| k = 10-20 | Balanced quality | Question answering | 10-20× |
| k = 50+ | Exhaustive search | Research, analysis | 50×+ |

### Practical Example

```python
# Hugging Face Transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")

input_text = "translate English to French: Hello, how are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Beam search decoding
outputs = model.generate(
    input_ids,
    max_length=50,
    num_beams=5,              # Beam width
    length_penalty=0.6,       # Length normalization
    early_stopping=True,      # Stop when all beams finish
    no_repeat_ngram_size=2    # Prevent 2-gram repetition
)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)  # "Bonjour, comment allez-vous?"
```

### Characteristics

**Advantages:**
- **Optimal-seeking**: Finds high-quality sequences
- **Deterministic**: Same input → same output (unlike sampling)
- **Coherent**: Maintains global sequence quality
- **Structured**: Excellent for tasks with "correct" answers

**Disadvantages:**
- **Computationally expensive**: k times slower than greedy
- **Generic**: Can produce safe, boring outputs
- **Repetitive**: May fall into loops without constraints
- **Not diverse**: All beams may converge to similar outputs

### Optimal Use Cases

**When to use beam search:**
- Machine translation (quality critical)
- Text summarization (faithfulness important)
- Question answering (accuracy matters)
- Code generation (correctness essential)
- Image captioning (single best description)

### Beam Search Variants

**Diverse Beam Search:**

Encourages diversity across beams by penalizing similar candidates:

```python
outputs = model.generate(
    input_ids,
    num_beams=6,
    num_beam_groups=3,        # Divide into 3 groups
    diversity_penalty=1.0,    # Penalize similarity
    max_length=50
)
```

**Constrained Beam Search:**

Forces inclusion of specific words/phrases:

```python
from transformers import PhrasalConstraint

constraints = [
    PhrasalConstraint(tokenizer("Paris", add_special_tokens=False).input_ids)
]

outputs = model.generate(
    input_ids,
    num_beams=5,
    constraints=constraints,  # Must include "Paris"
    max_length=50
)
```

---

## Strategy 3: Top-k Sampling

**Approach:** Only consider the top k most likely tokens, ignore the rest.

### How It Works

**Step-by-step process:**

1. **Rank tokens** by probability
2. **Keep only top k tokens** (e.g., k=40)
3. **Renormalize probabilities** to sum to 100%
4. **Sample from this reduced set**

**Visual Example (k=3):**

```
Original distribution:
  by doing projects: 28%
  through practice:  24%
  with courses:      15%
  to start small:    12%
  using books:        8%
  ... (43,000 more)

After Top-k filtering (k=3):
  by doing projects: 41.8%  (28 / 67)
  through practice:  35.8%  (24 / 67)
  with courses:      22.4%  (15 / 67)
  [All other tokens removed]

Sample randomly from these 3 options.
```

### Mathematics Behind Top-k

Given original probabilities $P(w_i)$:

1. **Select top k:** $\mathcal{V}_k = \{w_1, w_2, ..., w_k\}$
2. **Renormalize:**

$$
P'(w_i) = \begin{cases}
\frac{P(w_i)}{\sum_{j=1}^{k} P(w_j)} & \text{if } w_i \in \mathcal{V}_k \\
0 & \text{otherwise}
\end{cases}
$$

3. **Sample:** Choose $w_i$ with probability $P'(w_i)$

### Parameter Selection Guide

| k Value | Effect | Use Case |
|---------|--------|----------|
| k = 1 | Greedy (no randomness) | Factual tasks |
| k = 10-20 | Conservative creativity | Professional writing |
| k = 40-50 | Balanced exploration | General content |
| k = 100+ | High diversity | Creative fiction |

### Practical Example

```python
# Using Hugging Face
outputs = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.8,
    top_k=40,  # Only consider top 40 tokens
    max_length=100
)
```

### Top-k Limitations

**Problem: Fixed Cutoff**

Consider two scenarios:

**Scenario A (Narrow distribution):**
```
Token 1: 85%
Token 2:  8%
Token 3:  3%
...
Top-40: 0.001%
```

**Scenario B (Flat distribution):**
```
Token 1: 12%
Token 2: 11%
Token 3: 10%
...
Top-40: 2%
```

With k=40, Scenario A includes many near-zero probability tokens (noise), while Scenario B might need more than 40 tokens for diversity.

**Key Insight:** A fixed k doesn't adapt to the probability distribution's shape. This limitation led to the development of Top-p sampling, which we'll explore in Part 2B.

---

## Quick Comparison

| Strategy | Deterministic | Quality | Diversity | Speed |
|----------|---------------|---------|-----------|-------|
| Greedy | Yes | Medium | None | Fast |
| Beam Search | Yes | High | Low | Slow (k×) |
| Top-k | No | Medium | Medium | Fast |

**When to Choose:**

```
Greedy: Factual accuracy matters most
Beam Search: Quality > speed, have computational budget
Top-k: Need variation but want consistency
```

---

## Key Takeaways

**Fundamental strategies:**
- Greedy decoding: Fastest, deterministic, but repetitive
- Beam search: High quality, deterministic, but computationally expensive
- Top-k sampling: Balanced approach with fixed diversity threshold

**Greedy characteristics:**
- Zero computational overhead
- Perfect for factual tasks
- Risk of repetitive loops in creative contexts

**Beam search strengths:**
- Optimal for tasks with "correct" answers
- Essential for translation and summarization
- Can be enhanced with diversity penalties
- Length normalization prevents short-sequence bias

**Top-k characteristics:**
- Simple and effective
- Fixed cutoff doesn't adapt to confidence
- Works well with temperature control
- Good baseline for creative tasks

**Practical guidelines:**
- Factual QA: Use greedy (temperature = 0.0)
- Translation: Beam search (num_beams = 3-5)
- General content: Top-k (k = 40-50) with temperature = 0.7
- Always test on your specific use case

---

## What's Next?

In **Part 2B**, we'll explore adaptive and advanced sampling methods:

- **Top-p (nucleus) sampling**: Dynamic token selection based on cumulative probability
- **Best-of-N sampling**: Generate multiple candidates, return the best
- **Typical sampling**: Balance probability with information content
- **Contrastive search**: Penalize similarity for diverse outputs
- **Comprehensive comparisons**: When to use each strategy
- **Real-world configurations**: Complete setup examples by task type

These advanced methods address the limitations of fixed-k sampling and provide more sophisticated control over generation quality.

---

**Series Navigation:**
- [Part 1: Temperature and Randomness Control]({% post_url 2025-05-10-LLM-Inference-Parameters-Part1-Temperature %})
- **Part 2A: Basic Decoding Strategies** (Current)
- [Part 2B: Advanced Sampling Methods]({% post_url 2025-05-20-LLM-Inference-Parameters-Part2B-Advanced-Sampling %})
- [Part 3: Advanced Parameters and Practical Applications]({% post_url 2025-05-25-LLM-Inference-Parameters-Part3-Advanced-Parameters %})

**References:**
- Freitag & Al-Onaizan (2017). "Beam Search Strategies for Neural Machine Translation." NMT@ACL 2017.
- Vijayakumar et al. (2018). "Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models." AAAI 2018.
- Anderson et al. (2017). "Guided Open Vocabulary Image Captioning with Constrained Beam Search." EMNLP 2017.
- [Hugging Face - Generation Strategies Guide](https://huggingface.co/docs/transformers/generation_strategies)
- [OpenAI API Reference - Sampling Parameters](https://platform.openai.com/docs/api-reference/chat/create)
