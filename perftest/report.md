# Performance Benchmark Report: NVIDIA GeForce RTX 5070 Ti

## Summary
This report summarizes the performance results for the NVIDIA GeForce RTX 5070 Ti, focusing on matrix multiplication, memory bandwidth, and deep learning throughput.

## System Information
- **GPU**: NVIDIA GeForce RTX 5070 Ti
- **Benchmark Date**: January 27, 2026

## Matrix Multiplication Results
Measured using 8192 x 8192 matrices.

| Precision | Latency (seconds/op) |
|-----------|----------------------|
| **FP32**  | 0.0645               |
| **BF16**  | 0.0247               |

### Analysis
The **BF16** performance is approximately **2.61x faster** than FP32. This demonstrates the efficiency of the modern Tensor Cores in handling BFloat16, which is the standard for modern AI workloads.

## Data Transfer Performance
- **Host-to-GPU Bandwidth**: 7.36 GB/s

## Deep Learning Throughput
- **Model**: ResNet50
- **Batch Size**: 64
- **Throughput**: **335.69 images/sec**

## RL Pipeline Performance
Measured using `quick_dev_pipline.py` with CartPole-v1.

- **Device**: CUDA
- **Training Speed**: **217 steps/sec**
- **Evaluation Reward**: 11.0 (after 2000 tiny-step training)

### Observations
The ResNet50 throughput of ~336 images/sec and RL speed of 217 SPS are consistent with high-end consumer-grade hardware for the 50-series architecture, providing solid performance for both Computer Vision and Reinforcement Learning tasks.

## LLM Representation Engineering
A new pipeline for model steering has been implemented in [repe_pipeline.py](repe_pipeline.py). 

- **Method**: 4-bit Quantization (bitsandbytes) + Forward Hooks.
- **Goal**: Extract "Refusal" vectors and steer model behavior without fine-tuning.
- **Capacity**: 16GB VRAM allows for Mistral-7B/Llama-3-8B manipulation with ~10GB headroom for activation vectors.
