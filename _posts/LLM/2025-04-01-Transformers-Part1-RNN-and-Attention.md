---
title : "Transformers from Scratch - Part 1: From RNNs to Attention"
date : 2025-04-01 14:00:00 +0800
categories : ["Transformers", "LLM"]
tags :  ["Deep Learning", "NLP", "Transformers", "RNN", "Attention"]
---

Welcome to this comprehensive series on understanding Transformers from the ground up! In this first part, we'll explore the evolution from Recurrent Neural Networks (RNNs) to the attention mechanism that forms the foundation of Transformers.

## Why This Series?

Transformers have revolutionized natural language processing and beyond. To truly understand their power, we need to start from the beginning—understanding what came before and why we needed something better.

## RNN Understanding

Recurrent Neural Network (RNN) is a type of neural network designed to handle sequential data. Unlike traditional feedforward neural networks, RNNs have connections that form cycles, allowing information to be passed from one step of the sequence to the next. This structure enables RNNs to maintain a memory of previous inputs, making them suitable for tasks where context over time is important (like time-series forecasting, language modeling, etc.).

<div align="center">
  <img src="/assets/img/Tranformers_from_scratch/RNN_Image.webp" alt="RNN Architecture" />
</div>

### Basic Mechanism

1. **Inputs**: At each time step \( t \), an RNN receives an input vector \( x_t \).
2. **Hidden State**: The hidden state \( h_t \) is updated at each time step based on the current input \( x_t \) and the previous hidden state \( h_{t-1} \).
3. **Output**: The RNN generates an output at each time step, or a final output after processing the entire sequence.

#### Mathematical Formulation

- **Hidden State**:  
  ![Equation](https://latex.codecogs.com/svg.latex?h_t%20=%20\text{activation}(W_h%20\cdot%20h_{t-1}%20+%20W_x%20\cdot%20x_t))

- **Output**:  
  ![Equation](https://latex.codecogs.com/svg.latex?y_t%20=%20W_y%20\cdot%20h_t)

### Inputs and Outputs

- **Inputs**: RNNs take in a sequence of data. The input at each time step \( x_t \) can be a vector, representing things like a word in a sentence, a stock price in a time series, or a feature in a sequence of data.
  
- **Outputs**: 
  - For **sequence-to-sequence tasks** (e.g., machine translation), RNNs produce a sequence of outputs.
  - For **many-to-one tasks** (e.g., sentiment analysis), RNNs output a single value after processing the entire sequence.

### Advantages of RNNs

1. **Sequence Handling**: RNNs are designed to handle sequential data, making them ideal for tasks like speech recognition, time-series forecasting, and natural language processing.
  
2. **Context Retention**: The hidden state allows RNNs to retain memory of previous inputs, enabling them to learn dependencies over time.
  
3. **Flexible Input/Output Lengths**: RNNs can process sequences of varying lengths, which is useful in many NLP and time-series tasks.
  
4. **Parameter Sharing**: The weights in an RNN are shared across all time steps, making the model more efficient in terms of size compared to fully connected networks.

### Limitations of RNNs

Despite their advantages, RNNs have significant limitations that motivated the development of Transformers:

1. **Vanishing Gradient Problem**: During backpropagation, gradients can become very small, making it difficult to learn long-term dependencies. This limits the effectiveness of RNNs on long sequences.
  
2. **Exploding Gradient Problem**: On the other hand, gradients can also grow excessively large, causing unstable training.

3. **Difficulty with Long-Term Dependencies**: RNNs struggle to learn and retain information over long sequences due to the vanishing gradient problem.

4. **Slow Training**: Training RNNs is computationally expensive and slow, as each time step depends on the previous one, making them hard to parallelize.

5. **Limited Parallelism**: Since RNNs process data sequentially, it's challenging to parallelize computations effectively, which can hinder scalability.

## The Need for Attention

The limitations of RNNs, especially with long sequences, led researchers to develop a new mechanism: **Attention**.

### What is Attention?

Attention is a mechanism used in deep learning models, particularly in natural language processing (NLP) and computer vision, that allows the model to focus on specific parts of the input sequence when making predictions. Instead of processing the entire input equally, the attention mechanism helps the model determine which parts of the input are most important at each step.

The idea behind attention is inspired by how humans process information: when we read a sentence or observe an image, we don't give equal attention to every word or pixel. Instead, we focus on specific parts that are most relevant for understanding or making decisions. Attention in neural networks mimics this behavior by assigning different weights (or importance) to different elements in the input, based on the task at hand.

### Why We Need Attention

#### 1. Capturing Long-Term Dependencies

Attention helps overcome the limitations of traditional RNNs and LSTMs by allowing models to focus on relevant parts of the input, even from distant positions in the sequence. This enables better learning of long-term dependencies in tasks like machine translation or text generation.

**Example**: In the sentence "The cat, which had been sleeping on the warm windowsill all afternoon, suddenly jumped," attention allows the model to connect "cat" with "jumped" despite the long intervening clause.

#### 2. Improved Performance in Complex Tasks

Attention improves model performance by enabling the focus on important parts of the input sequence, which is particularly useful for tasks such as machine translation, text summarization, and image captioning.

#### 3. Parallelization

Attention mechanisms, especially in architectures like Transformers, enable parallel processing of input sequences. This significantly speeds up training and inference compared to sequential models like RNNs, leading to more scalable solutions.

**Key Insight**: Unlike RNNs that process tokens one at a time, attention can look at all tokens simultaneously, making computation much faster.

#### 4. Interpretability

Attention mechanisms provide insight into how the model makes predictions by highlighting the parts of the input it focuses on, which improves the interpretability of decisions, especially in complex tasks like machine translation.

You can visualize which words the model is "paying attention to" when making each prediction.

#### 5. Handling Variable-Length Sequences

Attention can efficiently handle input sequences of varying lengths by dynamically weighing the importance of different parts of the sequence, making it ideal for tasks with unpredictable input sizes, such as NLP.

#### 6. Flexibility Across Modalities

Attention is versatile and can be applied to different data modalities, such as text, images, and videos. In tasks like image captioning, attention helps focus on specific objects or regions in the image, improving the quality of generated descriptions.

## The Evolution Path

```
RNNs (Sequential Processing)
    ↓
LSTMs/GRUs (Better at long-term dependencies)
    ↓
Attention Mechanism (Focus on relevant parts)
    ↓
Transformers (Attention is all you need!)
```

## What's Next?

Now that we understand the limitations of RNNs and the motivation for attention, we're ready to dive into the Transformer architecture itself. In **Part 2**, we'll explore:

- The complete Transformer architecture overview
- Input embeddings and how they work
- Positional encoding and why it's crucial
- How Transformers maintain sequence order without recurrence

The attention mechanism we've introduced here will become the centerpiece of the Transformer, allowing it to overcome all the limitations of RNNs while achieving state-of-the-art performance.

## Key Takeaways

1. **RNNs** were groundbreaking for sequential data but suffer from vanishing gradients and slow training
2. **Long-term dependencies** are hard for RNNs to capture due to sequential processing
3. **Attention mechanism** allows models to focus on relevant parts of input regardless of distance
4. **Parallelization** becomes possible with attention, unlike sequential RNN processing
5. **Interpretability** improves as we can visualize what the model attends to

Stay tuned for Part 2, where we'll see how these concepts come together in the Transformer architecture!

---

**Series Navigation:**
- **Part 1: From RNNs to Attention** (Current)
- [Part 2: Architecture and Embeddings]({% post_url 2025-04-07-Transformers-Part2-Architecture-Embeddings %})
- [Part 3: Multi-Head Attention Deep Dive]({% post_url 2025-04-14-Transformers-Part3-Multi-Head-Attention %})
- [Part 4: Layer Norm and Feed-Forward Networks]({% post_url 2025-04-21-Transformers-Part4-Layer-Norm-FFN %})
- [Part 5: Decoder and Output Generation]({% post_url 2025-04-28-Transformers-Part5-Decoder-Output %})
- [Part 6: Training, Inference, and Applications]({% post_url 2025-05-05-Transformers-Part6-Training-Applications %})
