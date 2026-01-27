from benchmarl.algorithms import IqlConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import MlpConfig
import os

if __name__ == "__main__":
    # Force wandb to sync online and ensure it captures everything
    os.environ["WANDB_MODE"] = "online"
    # Optional: ensure we are using the right entity
    os.environ["WANDB_ENTITY"] = "sudingli21"

    # Get the default experiment configuration
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.sampling_device = "cpu"
    experiment_config.train_device = "cpu"
    
    # Standardize 1e7 training frames
    experiment_config.max_n_frames = 10_000_000
    experiment_config.on_policy = False
    
    # Configure WandB Logging Fixed
    experiment_config.loggers = ["wandb", "csv"]
    experiment_config.project_name = "benchmarl-instance-speed"
    experiment_config.wandb_extra_kwargs = {
        "entity": "sudingli21",
        "reinit": True # Allow multiple runs to re-init
    }
    
    # Checkpoints (must be multiple of 6000)
    experiment_config.checkpoint_interval = 600_000
    experiment_config.checkpoint_at_end = True
    
    # Disable evaluation rendering to avoid OpenGL errors in headless
    experiment_config.evaluation = False # If you don't have a display
    
    # Use IQL algorithm
    algorithm_config = IqlConfig.get_from_yaml()
    
    # Use VMAS Navigation (Stable substitute for SMAC in this env)
    # SMACv2 requires StarCraft II binaries which are not found in /root/StarCraftII
    print("Initializing VMAS task (Navigation) for 10M frames...")
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
    print("Starting experiment... Check WandB page for real-time charts.")
    experiment.run()
