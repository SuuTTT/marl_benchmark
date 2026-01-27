# BenchMARL Project Context

## Goal
1. Successfully run the default experiment from the [BenchMARL README](https://github.com/facebookresearch/BenchMARL).
2. Set up BenchMARL to log to Weights & Biases (WandB) for comparing running speeds across different instances.

## Solution
1. Create a dedicated directory (`benchmarl_run`).
2. Install dependencies (torch, torchvision, torchrl, benchmarl, wandb).
3. Verified existing WandB credentials in `/root/.netrc` (logged in as `sudingli21`).
4. Created the WandB project `benchmarl-instance-speed` via CLI initialization.
5. Developed `speed_test.py` to log detailed metrics to the new project.

## Encountered Issues
- **ImportErrors**: `IQLConfig` and `VmasConfig` were incorrect. Correct classes are `IqlConfig` and `VmasTask`.
- **Wandb Dependency**: BenchMARL defaults to wandb logging. Had to install `wandb` and set `export WANDB_MODE=disabled` initially, then switched to `online` for full tracking.
- **Rendering Exception**: VMAS tried to render periodic evaluations, failing due to missing OpenGL (GLU). Disabled evaluation via `experiment_config.evaluation = False`.
- **Logging issue**: Previous runs showed up in WandB but with no data. Fixed by ensuring `WANDB_MODE=online`, adding `reinit=True` in `wandb_extra_kwargs`, and ensuring the experiment doesn't crash before the first sync.
- **SMACv2 Availability**: Tried to run SMACv2 but it requires StarCraft II binaries (specifically `/root/StarCraftII`). Since these are missing, transitioned to a 10M frame training run on VMAS Navigation as a stable performance benchmark.

## Bugs
- Case sensitivity in configuration imports (`IqlConfig` vs `IQLConfig`).
- `checkpoint_interval` must be a multiple of `collected_frames_per_batch`.

## Lessons
- BenchMARL is highly modular and depends on `torchrl` and `vmas`.
- When running on headers or environments without displays, disable evaluation/rendering to avoid OpenGL errors.
- Use `WANDB_MODE=online` and check the experiment progress bar to ensure data is being pushed.
- SMACv2 requires a system-level StarCraft II installation; `pip install smacv2` only installs the Python wrapper.

## Device Utilization
- **Current Status**: Investigating **GPU (CUDA)** performance vs **CPU**.
- **Observation**: Initial benchmarks show MAPPO on CUDA (~51.6s/it) is significantly slower than IQL on CPU (~34.5s/it).
- **Hardware**: NVIDIA GeForce RTX 5070 Ti.

## Analysis: Why GPU can be slower in MARL Benchmarks
The observation that GPU execution is slower than CPU for this specific BenchMARL/VMAS setup is common in RL for several reasons:

1. **Kernel Launch Overhead**: Small MLP models (like the 128x128 used here) do not provide enough computational work to saturate a high-end GPU. The overhead of the CPU telling the GPU what to do (kernel launches) often exceeds the time saved by parallel execution.
2. **Data Transfer Latency**: If the simulator (VMAS) or the buffer management involves frequent movement of `TensorDict` objects between CPU and GPU memory, the PCI-e bus becomes a bottleneck.
3. **Synchronization & Latency**: On-policy algorithms like MAPPO require strict synchronization between sampling and training. GPUs have higher throughput but also higher latency; for small batches (6000 frames), the lower latency of the CPU cache wins.
4. **Simulator Vectorization Scale**: VMAS/TorchRL is optimized for *massive* vectorization. While 600 envs is a lot for a regular desktop, GPUs often only start showing benefits over optimized CPU backends when running **thousands** of parallel environments (e.g., 4096+). At 600, the overhead dominates.
5. **Python/Global Interpreter Lock (GIL)**: If the environment stepping is bound by Python logic despite being "vectorized," the GPU will spend most of its time idle waiting for the next instruction from the CPU.

## Launch Instructions
### Environment Requirements
- **Python**: 3.12
- **Virtual Environment**: `/workspace/marl_benchmark/benchmarl_run/venv`
- **Logging**: `wandb` (logged in as `sudingli21`)

### Running the Benchmarks
**CPU (IQL):**
```bash
source /workspace/marl_benchmark/benchmarl_run/venv/bin/activate
export WANDB_MODE=online
python3 -u /workspace/marl_benchmark/benchmarl_run/full_train.py > /workspace/marl_benchmark/benchmarl_run/full_train_execution_v2.log 2>&1 &
```

**GPU (MAPPO):**
```bash
source /workspace/marl_benchmark/benchmarl_run/venv/bin/activate
export WANDB_MODE=online
python3 -u /workspace/marl_benchmark/benchmarl_run/mappo_cuda_train.py > /workspace/marl_benchmark/benchmarl_run/mappo_cuda_execution.log 2>&1 &
```

## Changelog
### Important Files
- [benchmarl_run/context.md](benchmarl_run/context.md): Updated with CUDA analysis and dual-run instructions.
- [benchmarl_run/mappo_cuda_train.py](benchmarl_run/mappo_cuda_train.py): New script for GPU-accelerated MAPPO training.
- [benchmarl_run/full_train.py](benchmarl_run/full_train.py): Finalized script for 10M frame training on VMAS Navigation (CPU).
- [benchmarl_run/speed_test.py](benchmarl_run/speed_test.py): Initial performance script for standardized speed measurement.
- [benchmarl_run/run_default.py](benchmarl_run/run_default.py): Baseline script used for initial debugging.
- [benchmarl_run/smac_train.py](benchmarl_run/smac_train.py): Attempted SMACv2 config (Missing binaries).

### Active Runs
- **CPU IQL**: PID in [benchmarl_run/mappo_pid_v2.txt](benchmarl_run/mappo_pid_v2.txt).
- **GPU MAPPO**: PID in [benchmarl_run/mappo_cuda_pid.txt](benchmarl_run/mappo_cuda_pid.txt).

### Output & Logs
- [benchmarl_run/full_train_execution_v2.log](benchmarl_run/full_train_execution_v2.log): Active CPU training log.
- [benchmarl_run/mappo_cuda_execution.log](benchmarl_run/mappo_cuda_execution.log): Active GPU training log.
- [benchmarl_run/run.log](benchmarl_run/run.log): Log for initial trial runs.
- [benchmarl_run/install_check.log](benchmarl_run/install_check.log): Verification log for dependencies (`benchmarl`, `torchrl`).
