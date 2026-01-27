# MARL Benchmark Repository

## Overview
This repository consolidates benchmarking tools and baselines for Single-Agent and Multi-Agent Reinforcement Learning.

## Contents
- **[cleanrl_run/](cleanrl_run/)**: CleanRL baselines (PPO) and execution logs.
- **[benchmarl_run/](benchmarl_run/)**: BenchMARL experiments, focusing on performance benchmarking (SPS) and SMACv2/VMAS tasks.
- **[perftest/](perftest/)**: Performance testing scripts and throughput analysis.

## Central Changelog
### 2026-01-27
- **CleanRL**: Set up PPO baseline on CartPole-v1.
- **BenchMARL**: 
    - Resolved `IqlConfig` and `VmasTask` import naming issues.
    - Integrated WandB logging with `WANDB_MODE=online`.
    - Implemented a 10M frame training benchmark on `vmas/navigation`.
    - Documented headless rendering fixes (OpenGL issues).
    - Identified SMACv2 binary dependency requirements.
- **Environment**: Verified CUDA availability on NVIDIA GeForce RTX 5070 Ti.

## Device Usage
Experiments currently default to CPU for initial stability, but GPU (cuda:0) is verified and available.
