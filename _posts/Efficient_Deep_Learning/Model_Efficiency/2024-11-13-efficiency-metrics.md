---
title : Model Efficiency Metrics
date : 2024-11-13 13:00:00 +8000
categories : ["Model Efficiency"]
tags :  ["Model Efficiency", "metrics"]
description: How should we measure the efficiency of neural networks?
image: /assets/img/Efficiency_metrics.png
math: true
---

# Efficiency Metrics

When evaluating the efficiency of neural networks, we can assess it from three main perspectives: performance-related, memory-related efficiency and compute-related efficiency. These metrics help us understand how well the model uses computational resources and memory, which are critical factors in optimizing training time and inference speed, especially when working with large datasets or complex models.

## Performance-Related Efficieny Metrics
- ### Latency
    - Latency refers to the time delay between a request and its corresponding response. It can be influenced by several factors:
        - ***Network Latency :***   Time taken for data to travel across the network, influenced by bandwidth, routing, and other network characteristics.
        - ***Disk Latency :*** The time taken for the system to read/write data to/from storage.
        - ***Memory Latency :*** The time it takes to access data from memory (e.g., cache latency, RAM latency).
        - ***CPU Latency :*** Time taken by the CPU to process an instruction or data (e.g., pipeline delays).
    - **Latency is often seen as a measure of responsiveness, particularly in real-time or interactive applications**
    - **Measurement**: Measured in milliseconds (ms) or microseconds (µs).
    - **Importance**: Low latency is crucial for real-time applications (e.g., autonomous vehicles, real-time speech recognition), where quick responses are needed.
    - **Example**: Consider a real-time image classification task where an input image is fed into a neural network, and the goal is to classify the image into one of several categories.

        - Input: An image of size 224x224 pixels.
        - Network: A pre-trained convolutional neural network (CNN) like ResNet50.
        - If it takes 50 milliseconds (ms) for the network to process the image and produce a label, the latency of this system is 50 ms.
    
        - If a system needs to process 10 images per second in a real-time environment, the total latency per image must be less than 100 ms to ensure real-time performance.

- ### Throughput
    - **Definition :** Throughput is the number of tasks (or inputs) a system can process in a given time period.
    - **Measurement :** Measured in inferences per second (IPS) or samples per second.
    - **Importance :** High throughput is essential for batch processing or systems handling large volumes of data (e.g., recommendation engines).
    - **Example :**
        - To calculate throughput, we divide the batch size by the time taken to process it:
        <div style="font-size: 20px; font-style: italic; font-weight: bold; text-align: center;">
        $$
        \text{Throughput} = \frac{\text{Batch size}}{\text{Time per batch}} = \frac{100 \, \text{images}}{200 \, \text{ms}} = 500 \, \text{images per second (IPS)}
         $$
        </div>
        - This means that with a batch size of 100 images and a processing time of 200 milliseconds, the throughput is **500 images per second (IPS)**.

- ### Latency vs. Throughput: Key Differences

| **Metric**    | **Latency**                               | **Throughput**                              |
|---------------|-------------------------------------------|---------------------------------------------|
| **Definition**| Time to process a single input.          | Number of inputs processed per unit of time. |
| **Unit**      | **ms** or **µs**.                         | **IPS** or **samples per second**.          |
| **Focus**     | Measures **response time** for each request. | Measures **volume** of tasks handled per time period. |
| **Optimization**| Lower latency for real-time systems.   | Higher throughput for batch processing.     |
| **Example**   | Time to process one image.               | Number of images processed per second.      |

- ### Latency-Throughput Tradeoff:
    - In many machine learning and neural network systems, there's a tradeoff between latency and throughput. For example:
    
    - **Low latency** may require processing fewer items at once, which can decrease throughput. This is typical in real-time systems where each input needs a quick response.
    - **High throughput** often involves processing larger batches of data at once, which can lead to increased latency per individual request (since the system is processing multiple inputs at once). However, this can maximize resource utilization and speed when handling large volumes of data.
    - In practice, **batch size** plays a key role in this tradeoff. Larger batch sizes increase throughput (more items processed per second), but they often result in higher latency for each individual request, as the system waits to process all items in the batch before returning a result.

    - **Example of Latency-Throughput Tradeoff**
        - Real-Time Video Processing: If you want to classify each frame of a video with minimal delay, you may use a small batch size (e.g., 1 frame at a time) to ensure low latency, but this may reduce throughput, limiting the number of frames processed per second.
        - Image Classification in Bulk: If you're classifying a large number of images at once (e.g., in an image recognition task for a large dataset), you might process them in a batch (e.g., 256 images), which would increase throughput (processing many images per second), but each individual image would take longer to process due to the larger batch size.
- ### Conclusion
    - Latency is about how fast a single input is processed and is crucial for real-time or interactive applications.
    - Throughput is about how much data a system can process over a given period and is essential for large-scale or batch processing tasks.


## Memory-Related Efficiency Metrics

- ### No of Parameters
    - #### Definition
       - Parameters is the parameter (synapse/weight) count of the given neural network, i.e., the number of elements in the weight tensors
    
    - #### Notations
    
        | Notation                           | Parameter Name      |
        | :---------------------             | :-------------------|
        | ***n***                            | Batch Size          |
        | ***C<sub>i</sub>***                | Input Channels      |
        | ***C<sub>o</sub>***                | Output Channels     |
        | ***h<sub>i</sub>, h<sub>o</sub>*** | Input/Output Height |
        | ***w<sub>i</sub>, w<sub>o</sub>*** | Input/Output Width  |
        | ***k<sub>h</sub>, k<sub>w</sub>*** | Kernel Height/Width |
         | ***g***                           | Groups              |
 
   
    - #### No of parameters for different Layers

        | Layers                | Parameters (bias is ignored) |
        | :---------------------| :----------------------------|
        | Linear Layer          | ***C<sub>o</sub> ⋅ C<sub>i</sub>*** |
        | Convolution           | ***C<sub>o</sub> ⋅ C<sub>i</sub> ⋅ K<sub>h</sub> ⋅ K<sub>w</sub>***|
        | Grouped Convolution   | ***C<sub>0</sub> /g ⋅ C<sub>i</sub> /g ⋅ k<sub>h</sub> ⋅ k<sub>w</sub> ⋅ g*** |
        | Depthwise Convolution | ***C<sub>o</sub> ⋅ K<sub>h</sub> . K<sub>w</sub>***|

- ### Model Size
    - #### Definition
        - The total amount of memory required to store the model’s parameters (weights, biases, etc.).
        - Units: Bytes, Kilobytes (KB), Megabytes (MB), Gigabytes (GB).
        - In general, if the whole neural network uses the same data type (e.g., floating-point)
            <div style="font-size: 19px; font-style: italic; font-weight: bold; text-align: center;">
             $$
             \text{Model Size} = \text{Number of Parameters} \times \text{Size (Bit Width) of Each Parameter}
             $$
             </div>
             -  If all weights are stored with 32-bit numbers, total storage will be about
                 - Example: AlexNet has 61M parameters.
                     - 61M × 4 Bytes (32 bits) = 244 MB (244 × 106 Bytes)
                 - If all weights are stored with 8-bit numbers, total storage will be about
                     - 61M × 1 Byte (8 bits) = 61 MB

    - #### Why it's important:
        - A model with fewer parameters will require less memory, allowing it to run on machines with limited memory resources. This is crucial when deploying models to devices with memory constraints (e.g., mobile devices, embedded systems).

- ### Total/Peak activations
    - #### Total Activations
        - ##### Definition
            - Total activations refer to the overall memory needed to store all the intermediate outputs (or activations) as data moves through the network during training or inference. In essence, it’s the total amount of memory consumed by the model during its computations.
        - ##### Why it matters:
            - The more layers and neurons a model has, the more memory it requires to store activations.
            - Too much memory usage can slow down the model or even cause it to crash, especially on devices with limited resources.
            - Reducing total activations can help optimize memory use without sacrificing performance.
    
    - #### Peak Activations
        - ##### Definition
            - Peak activations represent the maximum amount of memory needed at any one point during the model’s run. This is especially important because it shows the "worst-case" scenario where the model is using the most memory at once, which could cause issues if the device doesn’t have enough memory.

       - ##### Why it matters:
           - If a model’s peak memory usage exceeds the available memory (like GPU RAM), it can lead to errors or slowdowns.
           - Optimizing for peak memory can prevent these problems and improve overall performance.
    
    - #### Why These Metrics Matter for Efficiency:
       - **Memory Usage**: Understanding these metrics helps you know how much memory the model needs, which is crucial when working with devices like smartphones or GPUs that have limited memory.
       - **Speed & Cost**: Reducing memory usage can make the model run faster and cheaper, especially on cloud platforms where memory costs money.
       - **Hardware Limitations**: Peak activations help ensure that the model doesn't overload the hardware, avoiding slowdowns or crashes.

## Compute-Related Efficiency Metrics
- ### **MAC**
    - #### Multiply-Accumulate operation (MAC)
        $$
            \mathbf{a \leftarrow a + b \cdot c}
        $$
    - #### Matrix-Vector Multiplication (MV)
        $$
            \mathbf{MACs = m \cdot n}
        $$
    - #### General Matrix-Matrix Multiplication (GEMM)
        $$
            \mathbf{MACs = m \cdot n \cdot k}
        $$
    - #### **No of MACs for different layers**

        | Layer                     | MACs (batch size =  1)                          |
        | :------------------------ | :-----------------------------------------------|
        | **Convolution**           | $c_o \cdot c_i$                               |
        | **Grouped Convolution**   | $c_i \cdot k_h \cdot k_w \cdot h_o \cdot w_o \cdot c_o$ |
        | **Grouped Convolution**   | $\frac{c_i}{g} \cdot k_h \cdot k_w \cdot h_o \cdot w_o \cdot c_o$ |
        | **Depthwise Convolution** | $k_h \cdot k_w \cdot h_o \cdot w_o \cdot c_o$ |

- ### **FLOP/FLOPS**

    - A multiply is a floating point operation
    - An add is a floating point operation
    - One multiply-accumulate operation (**MAC**)is two floating point operations (FLOP)
        - Example: AlexNet has 724M MACs, total number of floating point operations will be
            - 724M × 2 = 1.4G FLOPs

    - #### Floating Point Operation Per Second (FLOPS)
        <div style="font-size: 19px; font-style: italic; font-weight: bold; text-align: center;">
        $$
            \mathbf{FLOPS = \frac{FLOPs}{Sec}}
        $$
        </div>
