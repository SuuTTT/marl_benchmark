import torch
import time
import os
import json
from benchmarl.algorithms import MappoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import MlpConfig

# --- Compatibility Monkeypatch ---
import torchrl.objectives.ppo as ppo
original_init = ppo.ClipPPOLoss.__init__
def patched_init(self, *args, **kwargs):
    if "critic_coef" in kwargs: kwargs["critic_coeff"] = kwargs.pop("critic_coef")
    if "entropy_coef" in kwargs: kwargs["entropy_coeff"] = kwargs.pop("entropy_coef")
    return original_init(self, *args, **kwargs)
ppo.ClipPPOLoss.__init__ = patched_init
# ---------------------------------

def run_stress_test():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    # 1. Configuration Matrix
    # We test scaling (env count) and complexity (model size)
    scales = [512, 2048, 8192]
    models = {
        "Small": [128, 128],
        "Large": [1024, 1024, 1024] # To stress TFLOPS
    }
    
    results = []
    os.environ["WANDB_MODE"] = "disabled"

    print(f"🔥 Starting Stress Test on: {gpu_name}")
    print("="*50)

    for model_name, layers in models.items():
        for n_envs in scales:
            print(f"\n▶ Testing: Model={model_name} | Vectorization={n_envs} envs")
            
            # Setup Config
            exp_config = ExperimentConfig.get_from_yaml()
            exp_config.sampling_device = device
            exp_config.train_device = device
            exp_config.max_n_iters = 5
            exp_config.loggers = []
            exp_config.evaluation = False
            
            # Task setup
            # We bypass the BenchMARL task loader and use the VmasTask enum directly 
            # with our desired config to avoid the "multiple values" error 
            # which stems from how BenchMARL merges YAML and passed configs.
            from benchmarl.environments import VmasTask
            task = VmasTask.NAVIGATION.get_from_yaml()
            
            # This is the "cleanest" way to forceBenchMARL to use our env count:
            # We modify the config AND we don't pass anything that would conflict.
            task.config.update({"num_envs": n_envs, "max_steps": 100})
            
            # To avoid the 'multiple values' error, we strip num_envs from 
            # the lambda that BenchMARL generates if it exists.
            # But the lambda is hidden. 
            
            # Let's try to just use a different Task enum approach:
            # Most BenchMARL tasks can be created with a config dict.
            # If we don't use 'get_from_yaml', we might avoid the conflict.
            # But we want the navigation scenario.
            
            # FINAL ATTEMPT AT TASK SETUP:
            # We will use the VmasTask class directly with a clean config.
            from benchmarl.environments.vmas.common import VmasTask as VmasTaskClass
            task = VmasTaskClass.NAVIGATION.get_from_yaml()
            # Clear all and set ONLY what we need
            task._config = {"num_envs": n_envs, "max_steps": 100}
            
            # Model setup
            # In BenchMARL, MlpConfig requires layer_class and num_cells
            model_config = MlpConfig(
                num_cells=layers, 
                layer_class=torch.nn.Linear,
                activation_class=torch.nn.ReLU
            )
            
            experiment = Experiment(
                task=task,
                algorithm_config=MappoConfig.get_from_yaml(),
                model_config=model_config,
                critic_model_config=model_config,
                seed=0,
                config=exp_config,
            )

            # 2. Execution & Profiling
            # Snapshot CPU/GPU stats before
            print("--- Resource Snapshot (Before) ---")
            os.system("nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits")
            os.system("ps -p $$ -o %cpu,%mem --no-headers")
            
            start_time = time.time()
            experiment.run()
            total_time = time.time() - start_time
            
            # Snapshot CPU/GPU stats after
            print("--- Resource Snapshot (After) ---")
            os.system("nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits")
            
            total_frames = n_envs * 100 * 5 # (envs * frames_per_batch * iters)
            sps = total_frames / total_time
            
            print(f"📊 Result: {sps:.0f} SPS")
            results.append({
                "model": model_name,
                "envs": n_envs,
                "sps": sps,
                "total_time": total_time
            })

    # Save results
    filename = f"stress_results_{gpu_name.replace(' ', '_')}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n✅ All tests complete. Results saved to {filename}")

if __name__ == "__main__":
    run_stress_test()
