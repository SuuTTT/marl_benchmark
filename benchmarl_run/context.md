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
- **Current Status**: Running on **CPU**.
- **Reason**: The `sampling_device` and `train_device` were hardcoded to `"cpu"` in [benchmarl_run/full_train.py](benchmarl_run/full_train.py) and [benchmarl_run/speed_test.py](benchmarl_run/speed_test.py) to ensure maximum compatibility during initial environment setup and debugging of MARL dependencies.
- **Hardware Available**: Initial checks confirm an **NVIDIA GeForce RTX 5070 Ti** is available (`CUDA Available: True`).
- **Recommendation**: For actual performance benchmarking across instances, the scripts should be updated to use `cuda:0`.

## Launch Instructions
### Environment Requirements
- **Python**: 3.10+
- **Core ML**: `torch`, `torchvision`, `torchrl`
- **MARL Frameworks**: `benchmarl`, `vmas`
- **Logging**: `wandb` (logged in via `wandb login` or environment variables)
- **Headless Setup**: If running on a server without a GPU/Display, ensure `experiment_config.evaluation = False` or install `python3-opengl` and use `xvfb-run`.

### Running the Benchmark
To run the full 10M frame training benchmark with WandB logging:
```bash
export WANDB_MODE=online
python3 -u benchmarl_run/full_train.py
```

## Changelog
### Important Files
- [benchmarl_run/context.md](benchmarl_run/context.md): Project goals, issues, lessons, and launch docs.
- [benchmarl_run/full_train.py](benchmarl_run/full_train.py): Finalized script for 10M frame training on VMAS Navigation with WandB online logging.
- [benchmarl_run/speed_test.py](benchmarl_run/speed_test.py): Initial performance script for standardized speed measurement (100k frames).
- [benchmarl_run/run_default.py](benchmarl_run/run_default.py): Baseline script used for initial debugging of imports and rendering.
- [benchmarl_run/smac_train.py](benchmarl_run/smac_train.py): Attempted SMACv2 config (Requires external SC2 binaries).

### Output & Logs
- [benchmarl_run/full_train.log](benchmarl_run/full_train.log): Output log for the active 10M frame training run.
- [benchmarl_run/run.log](benchmarl_run/run.log): Log for initial trial runs.
- [benchmarl_run/install_check.log](benchmarl_run/install_check.log): Verification log for dependencies (`benchmarl`, `torchrl`).
