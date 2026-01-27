from benchmarl.algorithms import IqlConfig
from benchmarl.environments import Smacv2Task
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import MlpConfig
import os

if __name__ == "__main__":
    # Force wandb to sync online
    os.environ["WANDB_MODE"] = "online"

    # Get the default experiment configuration
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.sampling_device = "cpu"
    experiment_config.train_device = "cpu"
    
    # 1e7 training frames
    experiment_config.max_n_frames = 10_000_000
    experiment_config.on_policy = False
    
    # Configure WandB
    experiment_config.loggers = ["wandb", "csv"] # Adding CSV to verify local logging
    experiment_config.project_name = "benchmarl-instance-speed"
    experiment_config.wandb_extra_kwargs = {
        "entity": "sudingli21",
        "settings": {"silent": False}
    }
    
    # Checkpoints
    experiment_config.checkpoint_interval = 600_000 # Must be multiple of collected_frames_per_batch (6k)
    experiment_config.checkpoint_at_end = True

    # Use IQL algorithm
    algorithm_config = IqlConfig.get_from_yaml()
    
    # Use SMACv2 environment
    # Let's try a standard task. 
    try:
        task = Smacv2Task.PROTOSS_5_VS_5.get_from_yaml()
    except Exception as e:
        print(f"Failed to load SMACv2 task: {e}")
        # Fallback to VMAS if SMACv2 is not available to test logging
        print("Falling back to VMAS for logging test...")
        from benchmarl.environments import VmasTask
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
    experiment.run()
