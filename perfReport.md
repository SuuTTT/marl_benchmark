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

---

# Instance Performance Appendix: RTX 5060 Ti

## Instance Information
- **GPU**: 1x RTX 5060 Ti (Max CUDA 12.8, 22.8 TFLOPS, 15.9 GB VRAM)
- **CPU**: AMD EPYC 7502 (32-Core)
- **RAM**: 36.8 GB
- **Disk**: WD_BLACK SN850X HS 2000GB (5773.0 MB/s)
- **Motherboard**: ROMED8-2T (PCIe 4.0 x8, 13.4 GB/s)
- **Network**: 880.1 Mbps down / 804.1 Mbps up
- **Instance ID**: 30609757 | Host: 244103 | Machine ID: 39053

## Run Context
- **Issue Note**: BenchMARL performance was flagged as failure while running the pipeline in [perftest_output.txt](../perftest_output.txt).

## RL Run Performance (RTX 5060 Ti)

### 1. Quick Dev Pipeline (Single-Agent Smoke Test)
- **Environment**: `CartPole-v1`
- **Algorithm**: Simple Actor-Critic (smoke test)
- **Device**: CUDA
- **Training Speed**: **251 SPS**
- **Eval Episode Reward**: **22.0**

### 2. BenchMARL MAPPO (Multi-Agent Benchmark)
- **Environment**: VMAS `Navigation` (Multi-Robot)
- **Algorithm**: MAPPO (On-policy)
- **Device**: CUDA
- **Status**: **Failed** (bandwidth regression)
- **Notes**: Failure aligned with the bandwidth drop to 3.32 GB/s in this run.

## Comparative Metrics

| Metric | Your Previous Run (5060 Ti )| This Run 5060 Ti (running benchmarl) | Impact |
| --- | --- | --- | --- |
| FP32 Math | 0.0665 s | 0.0728 s | ~10% Slower (Not terrible) |
| BF16 Math | 0.0228 s | 0.0288 s | ~25% Slower |
| Bandwidth | 12.80 GB/s | 3.32 GB/s | **CRITICAL FAILURE** |

## Analysis
- The bandwidth regression is the dominant issue and aligns with the observed failure case under BenchMARL. This suggests a data-transfer bottleneck rather than a pure compute limitation.
- The FP32/BF16 slowdowns are relatively modest and would not alone explain the failure; the PCIe 4.0 x8 path and the measured bandwidth drop are the primary suspects.
