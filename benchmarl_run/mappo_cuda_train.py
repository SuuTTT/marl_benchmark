from benchmarl.algorithms import MappoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import MlpConfig
import os
import torch

if __name__ == "__main__":
    # Force wandb to sync online
    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_ENTITY"] = "sudingli21"

    # Get the default experiment configuration
    experiment_config = ExperimentConfig.get_from_yaml()
    
    # Use CUDA for sampling and training
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    experiment_config.sampling_device = device
    experiment_config.train_device = device
    
    # MAPPO is on-policy
    experiment_config.on_policy = True
    
    # Standardize 1e7 training frames
    experiment_config.max_n_frames = 10_000_000
    
    # Configure WandB Logging
    experiment_config.loggers = ["wandb", "csv"]
    experiment_config.project_name = "benchmarl-instance-speed"
    experiment_config.wandb_extra_kwargs = {
        "entity": "sudingli21",
        "reinit": True,
        "name": f"mappo_navigation_{device.replace(':', '_')}"
    }
    
    # Checkpoints
    experiment_config.checkpoint_interval = 600_000
    experiment_config.checkpoint_at_end = True
    
    # Disable evaluation rendering
    experiment_config.evaluation = False 
    
    # Use MAPPO algorithm
    algorithm_config = MappoConfig.get_from_yaml()
    
    # Use VMAS Navigation
    print("Initializing VMAS task (Navigation) for 10M frames with MAPPO...")
    task = VmasTask.NAVIGATION.get_from_yaml()

    # Use a simple MLP model
    model_config = MlpConfig.get_from_yaml()

    # Create and run the experiment
    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=model_config,
        seed=0,
        config=experiment_config,
    )
    print("Starting MAPPO CUDA experiment... Check WandB page for real-time charts.")
    experiment.run()
