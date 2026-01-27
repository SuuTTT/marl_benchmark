# Development Context: Reinforcement Learning Pipeline

This document provides the context and instructions required to rerun the performance tests and development pipeline from scratch.

## 1. Environment Setup

### Prerequisites
- **OS**: Linux (tested on Ubuntu)
- **Python**: 3.10+
- **Hardware**: GPU with CUDA support is recommended for performance, but CPU is supported.

### Dependencies
The following packages are required:
```bash
pip install gymnasium torch tensorboard numpy
```

### Installation
If you encounter missing modules, install them using:
```bash
pip install gymnasium torch tensorboard
```

## 2. Running the Code

### Quick Development Pipeline
To run the smoke test and verify the full pipeline (training, checkpointing, evaluation):
```bash
cd /workspace
python3 perftest/quick_dev_pipline.py
```

### Expected Output
- **Device**: Should detect `cuda` if a GPU is available.
- **SPS (Steps Per Second)**: Should be around 250+ SPS on a standard GPU setup.
- **Models**: Checkpoints are saved in `models/` (relative to the execution root).
- **Logs**: Tensorboard logs are saved in `runs/`.

## 3. Known Issues & Troubleshooting

### ModuleNotFoundError: No module named 'gymnasium'
**Issue**: The environment may not have `gymnasium` installed by default.  
**Fix**: Run `pip install gymnasium`. Note that older tutorials might use `gym`, but this pipeline specifically requires `gymnasium`.

### Path Errors
**Issue**: Running the script from inside the `perftest/` directory instead of the workspace root might cause issues with relative imports or saving paths.  
**Fix**: Always run the script from the `/workspace` directory using the full path: `python3 perftest/quick_dev_pipline.py`.

### CUDA Initialization
**Issue**: If CUDA is not correctly configured, `torch` might fallback to CPU, significantly reducing SPS.  
**Fix**: Verify CUDA installation with:
```python
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Headless Environments
**Issue**: `gymnasium` evaluation with `render_mode="human"` will fail on servers without a display.  
**Fix**: The script is configured to use `render_mode="rgb_array"` which is headless-safe. Avoid changing this to "human" unless working on a local machine with a monitor.
