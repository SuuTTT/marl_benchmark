# CleanRL Project Context

## Goal
Set up a clean environment and run a CleanRL algorithm (e.g., PPO) to verify the installation and baseline performance.

## Solution
1. Create a dedicated directory (`cleanrl_run`).
2. Installed dependencies: `torch`, `gymnasium[classic_control]`, `tensorboard`, `tyro`.
3. Downloaded `ppo.py` from CleanRL repository.
4. Run a baseline CleanRL script.

## Encountered Issues
- Terminal feedback was inconsistent ("opened the alternate buffer"), necessitating background runs and log checking.
- `tyro` argument parsing error: `--track False` is invalid; use `--no-track` or omit it.
- Log buffering making it hard to see real-time progress; solved by `python3 -u`.

## Bugs
- Invalid argument usage for `--track`.

## Lessons
- Use `-u` with Python when redirecting output to logs for real-time visibility.
- Always check the help menu (`--help`) for libraries using `tyro` or `argparse` as defaults/toggles might differ from expectations.
- CleanRL output can be found in the `runs/` directory even if terminal output is sparse.

## Launch Instructions
### Environment Requirements
- **Python**: 3.8+
- **Core RL**: `torch`, `gymnasium[classic_control]`
- **Utilities**: `tensorboard`, `tyro`

### Running the Script
To run the PPO baseline:
```bash
python3 -u cleanrl_run/ppo.py --env-id CartPole-v1 --total-timesteps 50000 --no-track
```

## Changelog
### Important Files
- [cleanrl_run/context.md](cleanrl_run/context.md): Project goals, issues, and lessons.
- [cleanrl_run/ppo.py](cleanrl_run/ppo.py): PPO implementation downloaded from CleanRL.

### Output & Logs
- [cleanrl_run/run.log](cleanrl_run/run.log): Execution log for the baseline run.
- [runs/](runs/): Tensorboard event files and experiment metadata.
