---
title: "Mastering LLM Inference Parameters - Part 3: Advanced Parameters and Practical Applications"
date: 2025-05-25 10:00:00 +0000
categories: ["LLM", "Inference"]
tags: ["inference", "frequency-penalty", "presence-penalty", "llm-parameters", "text-generation", "best-practices"]
math: true
---

In Parts 1, 2A, and 2B, we explored the foundational parameters—temperature controls randomness, decoding strategies (greedy, beam search) determine how we search for sequences, and sampling methods (top-k, top-p, typical, contrastive) filter which tokens are considered.

But what if you need more surgical precision? What if your model keeps repeating the same words? What if you want to encourage topic diversity? What if you need to ban specific outputs entirely?

This final installment covers advanced parameters that solve these specific challenges, plus best practices for combining all parameters effectively.

---

## The Need for Precision Control

Consider this common frustration:

**Prompt:** "Write a product description for a smartphone"

**Output (with just temperature and Top-p):**
```
This smartphone features a great camera, great battery life, and a great 
display. The great design makes it great for everyday use. It's truly great.
```

**The Problem:** Repetition of "great" despite reasonable temperature settings.

**The Solution:** Advanced penalty parameters that specifically target repetition, diversity, and output constraints.

---

## Parameter 1: Frequency Penalty

**Purpose:** Reduce repetition by penalizing tokens based on how often they've already appeared.

### How It Works

**Mechanism:**

1. **Track token frequency** in the generated text
2. **Apply penalty** proportional to frequency
3. **Adjust logits** before sampling:

$$
\text{logit}'(w_i) = \text{logit}(w_i) - (\alpha \times \text{frequency}(w_i))
$$

Where:
- $\alpha$: Frequency penalty coefficient (typically 0.0 to 2.0)
- $\text{frequency}(w_i)$: Number of times token $w_i$ appeared so far

**Simple Terms:** Each time a word appears, it becomes slightly less likely to appear again.

### Practical Example

**Without frequency penalty (α = 0.0):**
```
The best features are the camera, the battery, and the display. 
The camera is excellent. The battery lasts all day. The display is vibrant.
```
(Repetition of "the" is acceptable but monotonous)

**With moderate penalty (α = 0.5):**
```
The best features include the camera, battery, and display. 
Its camera performs excellently. Battery life lasts throughout the day. 
Display quality appears vibrant.
```
(Natural variation in phrasing)

**With high penalty (α = 1.5):**
```
Outstanding features: camera, battery, display. 
Photography capabilities excel. Power longevity sustains daily usage. 
Screen vibrancy impresses.
```
(Forced variety, potentially awkward)

### Parameter Selection Guide

| α Value | Effect | Use Case |
|---------|--------|----------|
| 0.0 | No penalty (default) | Natural repetition acceptable |
| 0.3-0.5 | Gentle discouragement | Professional writing |
| 0.6-1.0 | Moderate variety | Creative content |
| 1.0-2.0 | Strong avoidance | Poetry, headlines |
| > 2.0 | Excessive (avoid) | Creates unnatural text |

### Code Example

```python
# OpenAI API
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write about cloud computing"}],
    temperature=0.7,
    frequency_penalty=0.5,  # Discourage word repetition
    max_tokens=200
)

# Anthropic Claude API
response = anthropic.messages.create(
    model="claude-3-opus-20240229",
    messages=[{"role": "user", "content": "Write about cloud computing"}],
    temperature=0.7,
    top_p=0.9,
    # Note: Claude uses different penalty mechanisms
    max_tokens=200
)
```

### Common Use Cases

**Ideal For:**
- Marketing copy (avoid repetitive adjectives)
- Blog posts (varied vocabulary)
- Product descriptions (rich language)
- Creative writing (diverse expressions)

**Avoid For:**
- Technical documentation (precision over variety)
- Code generation (natural repetition of keywords)
- Legal text (exact terminology required)

---

## Parameter 2: Presence Penalty

**Purpose:** Encourage topic diversity by penalizing tokens that have appeared at all (regardless of frequency).

### How It Works

**Mechanism:**

$$
\text{logit}'(w_i) = \text{logit}(w_i) - \alpha \times \mathbb{1}[\text{count}(w_i) > 0]
$$

Where:
- $\alpha$: Presence penalty coefficient
- $\mathbb{1}[\cdot]$: Indicator function (1 if token appeared, 0 otherwise)

**Key Difference from Frequency Penalty:**
- **Frequency**: "You said 'great' 5 times? Each repetition makes it less likely."
- **Presence**: "You said 'great' once? Don't say it again."

### Practical Example

**Prompt:** "Explain machine learning"

**Without presence penalty (α = 0.0):**
```
Machine learning is a method of data analysis. Machine learning automates 
analytical model building. It uses algorithms that learn from data. Machine 
learning is widely used in various applications.
```
(Repetition of "machine learning")

**With presence penalty (α = 0.6):**
```
Machine learning is a method of data analysis that automates analytical 
model building. It uses algorithms that iteratively learn from data. This 
technology is widely used across various applications like image recognition 
and natural language processing.
```
(Shifts to "it", "this technology", "the approach")

### Comparison: Frequency vs Presence

| Aspect | Frequency Penalty | Presence Penalty |
|--------|------------------|------------------|
| **Target** | Repetition intensity | Topic stagnation |
| **Penalty Type** | Proportional to count | Binary (appeared or not) |
| **Effect** | Gradual discouragement | Immediate shift |
| **Best For** | Word-level variety | Topic diversity |

**Example Scenario:**

```
Prompt: "Write about AI benefits and risks"

Frequency Penalty (α = 0.7):
  - Discourages repeating specific words heavily
  - Still allows "AI" multiple times (core topic)
  - Output: Varied vocabulary within same topic

Presence Penalty (α = 0.7):
  - Pushes to introduce new concepts quickly
  - Forces transition from benefits → risks
  - Output: Broader topic coverage
```

### Parameter Selection Guide

| α Value | Effect | Use Case |
|---------|--------|----------|
| 0.0 | No penalty | Focused discussion |
| 0.3-0.5 | Gentle nudge | Balanced coverage |
| 0.6-1.0 | Topic exploration | Brainstorming |
| 1.0-2.0 | Aggressive diversity | Idea generation |
| > 2.0 | Excessive (avoid) | Disjointed text |

---

## Parameter 3: Repetition Penalty

**Purpose:** Unified penalty mechanism combining frequency and presence approaches (implementation varies by framework).

### Implementation Variations

**Hugging Face Transformers:**

$$
\text{score}(w_i) = \frac{\text{logit}(w_i)}{\alpha} \text{ if } w_i \text{ appeared, else } \text{logit}(w_i)
$$

**NVIDIA/Other Frameworks:**

$$
\text{logit}'(w_i) = \text{logit}(w_i) - \alpha \times \log(1 + \text{count}(w_i))
$$

**Key Point:** Always check your framework's documentation—"repetition penalty" implementations differ significantly.

### Code Example

```python
# Hugging Face Transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

output = model.generate(
    input_ids,
    max_length=100,
    temperature=0.8,
    repetition_penalty=1.2,  # > 1.0 discourages repetition
    top_p=0.9
)
```

### Recommended Values

| Value | Effect | Typical Use |
|-------|--------|-------------|
| 1.0 | No penalty (default) | Standard generation |
| 1.1-1.3 | Gentle discouragement | Most applications |
| 1.3-1.5 | Moderate penalty | Creative writing |
| 1.5-2.0 | Strong penalty | Brainstorming |
| > 2.0 | Excessive | Avoid |

---

## Parameter 4: Stop Sequences

**Purpose:** Automatically terminate generation when specific text patterns appear.

### How It Works

**Mechanism:**
1. Generate tokens normally
2. Check if output ends with any stop sequence
3. If match found, **stop generation immediately**
4. Return text up to (but excluding) the stop sequence

### Practical Applications

**Use Case 1: Structured Output**

```python
# Extract a list without extra commentary
response = generate(
    prompt="List three programming languages:\n1.",
    stop=["\n\n", "Now", "In conclusion"],  # Stop at paragraph break or conclusions
    max_tokens=100
)

# Output:
# "1. Python
#  2. JavaScript
#  3. Java"
# [Stops before continuing to explanations]
```

**Use Case 2: Dialogue Systems**

```python
# Prevent model from continuing past user's turn
response = generate(
    prompt="User: What's the weather?\nAssistant:",
    stop=["User:", "\n\n"],  # Stop when next user turn starts
    max_tokens=150
)

# Output:
# "The weather today is sunny with a high of 75°F."
# [Stops before inventing user's next question]
```

**Use Case 3: Code Generation**

```python
# Stop at end of function
response = generate(
    prompt="def calculate_fibonacci(n):",
    stop=["\n\ndef ", "\nclass ", "```"],  # Stop at next definition
    max_tokens=200
)
```

### Best Practices

**Effective Stop Sequences:**
- Structural markers: `"\n\n"`, `"---"`, `"###"`
- Role indicators: `"User:"`, `"Assistant:"`, `"System:"`
- Code boundaries: `"```"`, `"//END"`, `"\n\n#"`
- Natural endings: `"In conclusion"`, `"To summarize"`, `"Finally"`

**Common Mistakes:**
- Stop sequences too short: `"\n"` (stops after every line)
- Overlapping sequences: `["\n", "\n\n"]` (first triggers first)
- Rare sequences: `"xyzabc"` (never appears, useless)

---

## Parameter 5: Logit Bias

**Purpose:** Manually adjust the probability of specific tokens (boost or suppress).

### How It Works

**Mechanism:**

$$
\text{logit}'(w_i) = \text{logit}(w_i) + \text{bias}(w_i)
$$

Where:
- $\text{bias}(w_i)$: Manual adjustment for token $w_i$
- Positive bias: Increases probability
- Negative bias: Decreases probability
- Typical range: -100 to +100

### Practical Examples

**Use Case 1: Ban Specific Words**

```python
# Prevent model from using profanity
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a restaurant review"}],
    logit_bias={
        "1279": -100,  # Token ID for a specific unwanted word
        "2435": -100,  # Another banned token
    }
)
```

**Use Case 2: Encourage Specific Terminology**

```python
# Prefer "machine learning" over "AI"
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Explain the technology"}],
    logit_bias={
        "8003": 5,   # Boost "machine"
        "12345": 5,  # Boost "learning"
        "7620": -3,  # Slightly discourage "AI"
    }
)
```

**Use Case 3: Control Sentiment**

```python
# Encourage positive language
positive_tokens = {
    "4950": 3,   # "excellent"
    "1049": 3,   # "great"
    "4966": 3,   # "amazing"
}

response = generate(
    prompt="Review this product:",
    logit_bias=positive_tokens
)
```

### Finding Token IDs

```python
# OpenAI Tokenizer
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4")
token_id = encoding.encode("machine")[0]
print(f"Token ID for 'machine': {token_id}")

# Hugging Face
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
token_id = tokenizer.encode("machine", add_special_tokens=False)[0]
print(f"Token ID: {token_id}")
```

### Limitations

**Challenges:**
- **Multi-token words**: "machine learning" = 2 tokens (must bias both)
- **Subword tokens**: "running" might be "run" + "ning"
- **Context-dependency**: Biasing "bank" affects both financial and river contexts
- **Diminishing returns**: Extreme biases (±100) may create unnatural text

---

## Parameter 6: Length Controls

### Max Tokens

**Purpose:** Hard limit on generation length.

```python
response = generate(
    prompt="Explain quantum computing",
    max_tokens=100,  # Stop after 100 tokens (~75 words)
)
```

**Best Practices:**
- Set conservatively to avoid truncation mid-sentence
- Account for prompt length in total token budget
- Use stop sequences for natural endings

### Min Tokens (Framework-Specific)

**Purpose:** Force minimum output length.

```python
# Hugging Face
output = model.generate(
    input_ids,
    min_length=50,   # Generate at least 50 tokens
    max_length=150
)
```

**Use Cases:**
- Prevent overly brief responses
- Ensure completeness in summaries
- Force elaboration in explanations

---

## Combining Parameters: Best Practices

### The Parameter Interaction Matrix

| Temperature | Top-p | Frequency Penalty | Presence Penalty | Best For |
|-------------|-------|------------------|------------------|----------|
| 0.0 | N/A | 0.0 | 0.0 | Factual QA |
| 0.3 | 0.85 | 0.0 | 0.0 | Code generation |
| 0.5 | 0.90 | 0.3 | 0.0 | Professional writing |
| 0.7 | 0.90 | 0.5 | 0.3 | Marketing content |
| 0.9 | 0.92 | 0.7 | 0.5 | Creative writing |
| 1.0 | 0.95 | 0.5 | 0.7 | Brainstorming |

### Configuration Examples by Use Case

**1. Customer Support Bot**

```python
response = generate(
    prompt=user_query,
    temperature=0.4,        # Consistent, reliable
    top_p=0.85,             # Focused responses
    frequency_penalty=0.2,  # Slight variety
    presence_penalty=0.1,   # Stay on topic
    stop=["User:", "\n\n"], # Natural conversation breaks
    max_tokens=150
)
```

**Rationale:** Prioritize accuracy and consistency while avoiding robotic repetition.

---

**2. Blog Post Generation**

```python
response = generate(
    prompt="Write about renewable energy",
    temperature=0.7,        # Creative but coherent
    top_p=0.90,             # Balanced exploration
    frequency_penalty=0.5,  # Varied vocabulary
    presence_penalty=0.3,   # Cover multiple aspects
    max_tokens=500
)
```

**Rationale:** Encourage rich language and topic diversity while maintaining readability.

---

**3. Code Completion**

```python
response = generate(
    prompt="def sort_array(arr):",
    temperature=0.2,        # Deterministic, reliable
    top_p=0.85,             # High-confidence tokens
    frequency_penalty=0.0,  # Allow natural keyword repetition
    presence_penalty=0.0,   # Don't penalize standard patterns
    stop=["\n\ndef ", "\nclass "],  # Stop at next function
    max_tokens=150
)
```

**Rationale:** Code has natural repetition (loops, keywords); prioritize correctness over variety.

---

**4. Creative Fiction**

```python
response = generate(
    prompt="Once upon a time in a distant galaxy",
    temperature=1.0,        # High creativity
    top_p=0.95,             # Broad exploration
    frequency_penalty=0.6,  # Rich vocabulary
    presence_penalty=0.6,   # Topic diversity
    max_tokens=800
)
```

**Rationale:** Maximize creativity and diversity for engaging narratives.

---

**5. Product Description**

```python
response = generate(
    prompt="Describe wireless noise-canceling headphones",
    temperature=0.6,        # Moderately creative
    top_p=0.88,             # Focused but varied
    frequency_penalty=0.7,  # Avoid repetitive adjectives
    presence_penalty=0.2,   # Cover all features
    logit_bias={
        token_id_excellent: 2,  # Encourage positive terms
        token_id_amazing: 2,
    },
    max_tokens=200
)
```

**Rationale:** Professional tone with rich language and positive sentiment.

---

**6. Brainstorming Session**

```python
response = generate(
    prompt="Generate startup ideas for sustainable tech",
    temperature=0.9,        # Highly creative
    top_p=0.92,             # Broad exploration
    frequency_penalty=0.5,  # Varied phrasing
    presence_penalty=0.8,   # Force diverse ideas
    max_tokens=400
)
```

**Rationale:** Maximize idea diversity; presence penalty forces new concepts.

---

## Common Multi-Parameter Mistakes

### Mistake 1: Conflicting Parameters

**Problem:**
```python
temperature=0.0,         # Deterministic
frequency_penalty=1.5    # High penalty
```

**Issue:** Temperature=0.0 means greedy decoding (no sampling), so penalties have no effect.

**Solution:** Use temperature ≥ 0.3 when applying penalties.

---

### Mistake 2: Over-Penalization

**Problem:**
```python
frequency_penalty=1.5,
presence_penalty=1.5,
repetition_penalty=2.0
```

**Issue:** Multiple aggressive penalties create unnatural, disjointed text.

**Solution:** Use penalties moderately; choose one primary penalty mechanism.

---

### Mistake 3: Ignoring Context

**Problem:** Using high creativity parameters for factual tasks

```python
# Wrong for factual QA
temperature=1.0,
top_p=0.95,
frequency_penalty=0.8
```

**Solution:** Match parameters to task requirements (see configuration matrix above).

---

## Debugging Parameter Issues

### Issue 1: Repetitive Output

**Symptoms:** Same words/phrases repeated excessively

**Diagnosis:**
- Check: Is temperature > 0?
- Check: Are penalties enabled?

**Solution:**
```python
frequency_penalty=0.5-0.8  # Start moderate
presence_penalty=0.3-0.5   # If topic-level repetition
```

---

### Issue 2: Incoherent Output

**Symptoms:** Grammatically correct but meaningless text

**Diagnosis:**
- Check: Is temperature too high? (> 1.2)
- Check: Are penalties too aggressive? (> 1.5)

**Solution:**
```python
temperature=0.7-0.9  # Reduce creativity
frequency_penalty=0.3-0.5  # Lower penalties
```

---

### Issue 3: Truncated Responses

**Symptoms:** Answers cut off mid-sentence

**Diagnosis:**
- Check: `max_tokens` too low?
- Check: Stop sequences triggering early?

**Solution:**
```python
max_tokens=300  # Increase limit
stop=["\n\n"]   # Use natural boundaries only
```

---

## Key Takeaways

**Penalty parameters:**
- **Frequency penalty**: Discourages word repetition proportionally
- **Presence penalty**: Encourages topic diversity (binary)
- **Repetition penalty**: Framework-specific unified approach
- All penalties work best with temperature > 0.3

**Control mechanisms:**
- **Stop sequences**: Natural boundaries for generation
- **Logit bias**: Surgical token probability adjustments
- **Length controls**: Hard limits on output size

**Parameter combinations:**
- Match configuration to task type (factual vs creative)
- Avoid conflicting settings (deterministic + high penalties)
- Use penalties moderately (0.3-0.8 range for most tasks)
- Test iteratively with real prompts

**Best practices:**
- Start with baseline (temp=0.7, top_p=0.9, no penalties)
- Adjust one parameter at a time
- Document configurations that work for your use case
- Build a configuration library for common scenarios

**Common pitfalls:**
- Over-penalization creates unnatural text
- Extreme parameter values rarely help
- Penalties have no effect with temperature=0.0
- Always consider task requirements first

---

## Comprehensive Parameter Reference

**Quick Reference Table:**

| Parameter | Range | Default | Primary Effect |
|-----------|-------|---------|----------------|
| Temperature | 0.0-2.0 | 1.0 | Randomness control |
| Top-p | 0.0-1.0 | 1.0 | Token filtering |
| Top-k | 1-100+ | None | Fixed nucleus size |
| Frequency penalty | 0.0-2.0 | 0.0 | Word repetition |
| Presence penalty | 0.0-2.0 | 0.0 | Topic diversity |
| Repetition penalty | 1.0-2.0 | 1.0 | Unified penalty |
| Max tokens | 1-∞ | Model-specific | Length limit |
| Stop sequences | String array | None | Termination control |
| Logit bias | -100 to +100 | 0 | Token probability |

---

## Final Thoughts: The Art of Parameter Tuning

Mastering LLM inference parameters is both science and art:

**The Science:**
- Understand mathematical effects
- Follow established best practices
- Use systematic testing

**The Art:**
- Develop intuition through experimentation
- Recognize patterns in output quality
- Adapt to specific use cases

**Key Principle:** Parameters are tools, not magic. The best configuration depends on your specific task, domain, and quality requirements.

**Recommended Workflow:**

1. **Start simple**: Temperature + Top-p baseline
2. **Identify issues**: Repetition? Incoherence? Off-topic?
3. **Apply targeted fixes**: Frequency penalty for repetition, etc.
4. **Test systematically**: One parameter at a time
5. **Document successes**: Build your configuration library
6. **Iterate**: Refine based on real-world performance

With these tools and understanding, you now have complete control over LLM generation behavior.

---

**Series Navigation:**
- [Part 1: Temperature and Randomness Control]({% post_url 2025-05-10-LLM-Inference-Parameters-Part1-Temperature %})
- [Part 2A: Basic Decoding Strategies]({% post_url 2025-05-15-LLM-Inference-Parameters-Part2A-Basic-Strategies %})
- [Part 2B: Advanced Sampling Methods]({% post_url 2025-05-20-LLM-Inference-Parameters-Part2B-Advanced-Sampling %})
- **Part 3: Advanced Parameters and Practical Applications** (Current)

**References:**
- [OpenAI API Documentation - Parameter Reference](https://platform.openai.com/docs/api-reference/chat/create)
- [Anthropic Claude API - Generation Parameters](https://docs.anthropic.com/claude/reference/messages_post)
- Keskar et al. (2019). "CTRL: A Conditional Transformer Language Model for Controllable Generation." arXiv:1909.05858
- [Hugging Face - Generation Configuration](https://huggingface.co/docs/transformers/main_classes/text_generation)
- Holtzman et al. (2019). "The Curious Case of Neural Text Degeneration." ICLR 2020
- [Google AI - Best Practices for Text Generation](https://ai.google/responsibility/responsible-ai-practices/)
- Klein et al. (2017). "OpenNMT: Open-Source Toolkit for Neural Machine Translation." ACL 2017
