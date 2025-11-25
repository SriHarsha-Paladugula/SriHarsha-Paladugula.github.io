---
title : "Transformers from Scratch - Part 2: Architecture and Embeddings"
date : 2025-04-07 14:00:00 +0800
categories : ["Transformers", "LLM"]
tags :  ["Deep Learning", "NLP", "Transformers", "Embeddings", "Positional Encoding"]
---

# Transformers from Scratch - Part 2: Architecture and Embeddings

In [Part 1]({% post_url 2025-04-01-Transformers-Part1-RNN-and-Attention %}), we explored RNNs and understood why we need attention. Now, let's dive into the Transformer architecture and understand how it prepares input data for processing.

## Transformer Architecture Overview

The Transformer architecture introduced in the seminal paper "Attention Is All You Need" revolutionized how we process sequential data. Here's the complete architecture:

<div align="center">
  <img src="/assets/img/Tranformers_from_scratch/Transformers_Architecture.webp" alt="Transformers Architecture" />
</div>

The Transformer consists of two main macro-blocks:

1. **Encoder**: Processes the input sequence
2. **Decoder**: Generates the output sequence
3. **Linear + Softmax Layer**: Converts decoder output to predictions

### The High-Level Flow

```
Input Text → Embedding → Positional Encoding → Encoder → 
Encoder Output → Decoder (with target) → Linear → Softmax → Output
```

Each encoder and decoder consists of multiple identical layers (6 in the original paper), stacked on top of each other. This deep architecture allows the model to learn complex patterns and representations.

## The Encoder Block

Let's focus on the encoder first. Here's what it looks like:

<div align="center">
  <img src="/assets/img/Tranformers_from_scratch/Encoder.webp" alt="Encoder Architecture" />
</div>

The encoder has several key components:
- Input Embedding
- Positional Encoding
- Multi-Head Attention
- Layer Normalization
- Feed Forward Network
- Residual Connections

In this post, we'll cover the first two components that prepare the input data.

## Input Embeddings

<div align="center">
  <img src="/assets/img/Tranformers_from_scratch/Input_Embedding.webp" alt="Input Embedding" />
</div>

### What are Input Embeddings?

Input embeddings are a way to convert raw data (like words, sentences, or other types of input) into numerical representations that machine learning models can understand. They map each item in the input (e.g., a word) to a vector of numbers, capturing semantic meaning or relationships based on patterns learned from large datasets.

**Key Property**: Words with similar meanings have similar embeddings, allowing the model to recognize context and relationships between them.

For example:
- "king" and "queen" would have similar embeddings
- "cat" and "dog" would be closer than "cat" and "car"

### The Process: From Text to Embeddings

<div align="center">
  <img src="/assets/img/Tranformers_from_scratch/Input_Embedding_numbers.webp" alt="Input Embedding in real" />
</div>

#### Step 1: Tokenization

The first step in processing a sentence is tokenization. This involves breaking down the sentence into smaller pieces called tokens (which could be words, subwords, or characters, depending on the tokenizer).

**Example of tokenization**:

Let's take the sentence: "Your cat is a lovely cat"

```
"Your"   → Token ID 105
"cat"    → Token ID 6587
"is"     → Token ID 5475
"a"      → Token ID 3578
"lovely" → Token ID 65
"cat"    → Token ID 6587 (same as before)
```

The tokenized sequence becomes: `[105, 6587, 5475, 3578, 65, 6587]`

#### Step 2: Embedding Layer

After tokenization, each token ID is mapped to a vector of real numbers in the embedding space. These vectors are high-dimensional representations that capture semantic properties of the tokens.

**Dimensions in Practice**:
- In modern models like GPT, BERT, or Transformers, each token is represented as a vector with hundreds or thousands of dimensions
- The original Transformer paper uses **512 dimensions** (also called $d_{model}$)

**Example**:

```
Token ID 105   → [0.23, -0.45, 0.67, ..., 0.12]  (512 numbers)
Token ID 6587  → [0.89, 0.34, -0.23, ..., 0.45]  (512 numbers)
Token ID 5475  → [-0.12, 0.78, 0.34, ..., -0.56] (512 numbers)
```

Each of these 512-dimensional vectors encodes semantic information about the word.

### Why Embeddings?

1. **Semantic Representation**: Similar words get similar vectors
2. **Dimensionality Reduction**: A vocabulary of 50,000 words is represented in just 512 dimensions
3. **Learnable**: The embeddings are learned during training to be optimal for the task
4. **Dense Representation**: Unlike one-hot encoding, embeddings are dense vectors that capture relationships

## Positional Encoding

Now we have a crucial problem: **Transformers process all tokens in parallel**, unlike RNNs which process sequentially. This means the model has no inherent sense of word order!

### The Problem

Consider these two sentences:
- "The cat chased the dog"
- "The dog chased the cat"

Without positional information, the Transformer would treat both sentences identically since they contain the same words (same embeddings)! But the meaning is completely different.

### The Solution: Positional Encoding

Positional encoding is a technique used to inject information about the order of tokens in a sequence.

<div align="center">
  <img src="/assets/img/Tranformers_from_scratch/positional_encoding.webp" alt="position encoding" />
</div>

### How It Works

We add a position embedding vector of size 512 to our original word embedding. The values in the position encoding vector are calculated only once and reused for every sentence during training and inference.

**Formula**:

$$\text{Encoder Input} = \text{Word Embedding} + \text{Position Embedding}$$

**Key Insight**: For the same word, the word embedding is always the same, but the position embedding changes based on where the word appears in the sentence.

Example:
```
"cat" at position 1: Embedding[cat] + PositionalEncoding[1]
"cat" at position 6: Embedding[cat] + PositionalEncoding[6]
```

These produce different final representations even though the word is the same!

### Calculating Position Embeddings

<div align="center">
  <img src="/assets/img/Tranformers_from_scratch/position_embedding.webp" alt="position encoding" />
</div>

The position embeddings use sinusoidal functions:

**For even positions** (i = 0, 2, 4, ...):
$$PE_{(pos, i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

**For odd positions** (i = 1, 3, 5, ...):
$$PE_{(pos, i)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

Where:
- $pos$ = Position of the word in the sentence (0, 1, 2, ...)
- $i$ = Dimension index (0 to 511)
- $d_{model}$ = Model dimension (512)

### Why Trigonometric Functions?

Trigonometric functions have several beneficial properties:

1. **Continuous Pattern**: sin and cos naturally represent a pattern that the model can recognize as continuous

2. **Relative Position**: The model can learn to attend to relative positions easily. For any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$

3. **Bounded Values**: sin and cos always stay between -1 and 1, preventing the position encodings from dominating the word embeddings

4. **Unique Encoding**: Each position gets a unique encoding

5. **Extrapolation**: The model can potentially handle sequences longer than those seen during training

### The Complete Input

After both word embeddings and positional encodings, we have:

```
Input Shape: (sequence_length, 512)

For each token:
Final_Embedding[token] = Word_Embedding[token] + Positional_Encoding[position]
```

This final embedding contains both:
- **What** the word means (from word embedding)
- **Where** the word is located (from positional encoding)

## Summary: From Text to Representation

Let's trace the complete journey:

```
1. Input Text: "Hello world"
   
2. Tokenization: [152, 856]
   
3. Word Embeddings:
   Token 152 → [0.23, -0.45, ..., 0.12] (512-d)
   Token 856 → [0.89, 0.34, ..., 0.45]  (512-d)
   
4. Positional Encodings:
   Position 0 → [0.00, 1.00, ..., 0.01] (512-d)
   Position 1 → [0.84, 0.54, ..., 0.02] (512-d)
   
5. Final Input:
   Token 0: [0.23, 0.55, ..., 0.13] (embedding + position)
   Token 1: [1.73, 0.88, ..., 0.47] (embedding + position)
```

Now this representation is ready to be processed by the attention mechanism!

## What's Next?

In **Part 3**, we'll dive deep into the heart of the Transformer: **Multi-Head Attention**. We'll explore:

- What are Query, Key, and Value?
- Scaled dot-product attention mechanism
- Why we need multiple attention heads
- How attention allows parallel processing

The embeddings we've created in this post will be transformed by the attention mechanism to capture relationships between all words in the sequence.

## Key Takeaways

1. **Input Embeddings** convert tokens to dense 512-dimensional vectors capturing semantic meaning
2. **Tokenization** breaks text into manageable pieces (words, subwords, or characters)
3. **Positional Encoding** injects position information using sinusoidal functions
4. **Addition** of word embeddings and positional encodings creates the final input
5. **Trigonometric functions** provide unique, continuous, and bounded positional encodings
6. **Each position** gets the same positional encoding across all sentences

---

**Series Navigation:**
- [Part 1: From RNNs to Attention]({% post_url 2025-04-01-Transformers-Part1-RNN-and-Attention %})
- **Part 2: Architecture and Embeddings** (Current)
- [Part 3: Multi-Head Attention Deep Dive]({% post_url 2025-04-14-Transformers-Part3-Multi-Head-Attention %})
- [Part 4: Layer Norm and Feed-Forward Networks]({% post_url 2025-04-21-Transformers-Part4-Layer-Norm-FFN %})
- [Part 5: Decoder and Output Generation]({% post_url 2025-04-28-Transformers-Part5-Decoder-Output %})
- [Part 6: Training, Inference, and Applications]({% post_url 2025-05-05-Transformers-Part6-Training-Applications %})
