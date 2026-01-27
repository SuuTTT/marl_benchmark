from benchmarl.algorithms import IqlConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import MlpConfig

if __name__ == "__main__":
    # Get the default experiment configuration and customize
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.sampling_device = "cpu"
    experiment_config.train_device = "cpu"
    experiment_config.total_frames = 100_000 # Increased frames for better speed measurement
    experiment_config.max_n_iters = 50
    experiment_config.on_policy = False
    
    # Configure WandB
    experiment_config.loggers = ["wandb"]
    experiment_config.project_name = "benchmarl-instance-speed"
    experiment_config.wandb_extra_kwargs = {
        "entity": "sudingli21"
    }
    
    # Use IQL algorithm
    algorithm_config = IqlConfig.get_from_yaml()
    
    # Use VMAS environment (Navigation task)
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
