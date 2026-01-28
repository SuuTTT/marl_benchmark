# Performance Report: RTX 5090 RL Instance

## Instance Specifications
- **GPU**: NVIDIA GeForce RTX 5090 (32GB VRAM, 108.1 TFLOPS)
- **CPU**: 12th Gen Intel Core i5-12400F (12 Virtual Cores)
- **RAM**: 64.1 GB
- **Motherboard**: B760M-HDVP (PCIe 5.0 @ 41.2 GB/s)
- **Disk**: WD_BLACK SN850X (5839 MB/s)

## Low-Level GPU Benchmarks
- **FP32 Matrix Multiplication**: 0.0175 seconds/op
- **BF16 Matrix Multiplication**: 0.0053 seconds/op
- **Host-to-GPU Bandwidth**: 14.59 GB/s

## RL Run Performance

### 1. CleanRL PPO (Single-Agent Baseline)
- **Environment**: `CartPole-v1`
- **Algorithm**: PPO (PyTorch)
- **Training Speed**: **813 SPS** (Steps Per Second)

### 2. BenchMARL MAPPO (Multi-Agent Benchmark)
- **Environment**: VMAS `Navigation` (Multi-Robot)
- **Algorithm**: MAPPO (On-policy)
- **Device**: CUDA
- **Iteration Time**: 5.93s / 6000 frames
- **Training Speed**: **~1,012 SPS**

## Observations
- **Architecture Efficiency**: The RTX 5090 demonstrates high matrix math throughput in BF16, which is beneficial for modern RL architectures.
- **SPS Comparison**: The multi-agent VMAS navigation benchmark achieves higher overall throughput (1,012 SPS) compared to the simple single-agent CartPole script (813 SPS) on this instance, likely due to better vectorization in the VMAS/TorchRL stack despite the complexity of MAPPO.
- **PCIe Bandwidth**: The measured host-to-GPU bandwidth (~14.6 GB/s) is solid, ensuring that data movement doesn't severely bottleneck the on-policy updates.
