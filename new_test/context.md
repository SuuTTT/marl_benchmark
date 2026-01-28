# New Test Suite: Hardware Stress Benchmarks

This directory contains advanced stress tests designed to push the RTX 5090 to its limits across RL, LLM, and low-level compute workloads.

## Main Scripts
- `full_benchmark.py`: A comprehensive suite measuring:
    - **TFLOPS**: FP32 and BF16 peak performance.
    - **PCIe Bandwidth**: Host-to-Device and Device-to-Host.
    - **RL Peak SPS**: MAPPO throughput at 8192 environments using VMAS.
    - **Transformer Throughput**: Training speed for large hidden dimensions.
- `stress_test.py`: Isolated scaling test for environment vectorization.

## Usage
To run the full hardware profile:
```bash
python new_test/full_benchmark.py
```

## Results
The results are saved as JSON files named `rtx_5090_<timestamp>.json` within this directory for historical tracking.
