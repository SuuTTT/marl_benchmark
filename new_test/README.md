# GPU Limit Benchmark (Stress Test)

This directory contains tools to push any RTX GPU to its limit using Multi-Agent RL.

## Quick Start on New Instance

1. **Run Setup** (installs all dependencies):
   ```bash
   bash new_test/setup_all.sh
   ```

2. **Run Stress Test**:
   ```bash
   python3 new_test/stress_test.py
   ```

## What this tests:

### 1. Vectorization Scaling
It runs the benchmark at 512, 2048, and 8192 environments.
- **Goal**: Find where the SPS plateaus. High-end cards (5090) usually peak at 8192+, while lower-end cards (3060) may peak at 2048.

### 2. Compute Complexity (TFLOPS)
It switches between:
- **Small Model**: [128, 128] MLP (standard benchmark).
- **Large Model**: [1024, 1024, 1024] MLP.
- **Goal**: Stress the Tensor Cores. On a 5090, the Large model should show significantly higher TFLOPS utilization.

### 3. I/O & PCIE Bottleneck
By comparing SPS across different environment counts, you can see if the card is waiting on data (low scaling) or compute (model size drop).

## Results
Results are saved to a JSON file named after the GPU (e.g., `stress_results_NVIDIA_GeForce_RTX_5090.json`).
