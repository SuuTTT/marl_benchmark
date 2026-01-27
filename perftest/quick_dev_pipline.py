import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from torch.utils.tensorboard import SummaryWriter

# ==========================================
# 1. THE "NEW ALGORITHM" (Replace this part!)
# ==========================================
class SimpleActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def get_action(self, x):
        probs = self.actor(x)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), self.critic(x)

# ==========================================
# 2. THE PIPELINE TEST
# ==========================================
def run_pipeline_test(experiment_name="dev_test"):
    # CONFIG
    env_id = "CartPole-v1"
    total_steps = 2000      # Tiny number for quick testing
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🚀 Starting Pipeline Test: {experiment_name}")
    print(f"--- Device: {device}")

    # SETUP: Environment & Logging
    run_name = f"{experiment_name}_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    env = gym.make(env_id)
    
    # SETUP: Agent & Optimizer
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    agent = SimpleActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

    # 3. TRAINING LOOP (The "Inner Loop")
    print("\n🔄 Step 1: Training Loop Smoke Test...")
    obs, _ = env.reset()
    start_time = time.time()
    
    for step in range(total_steps):
        # A. Collect Data
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        action, log_prob, value = agent.get_action(obs_tensor)
        
        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        # B. Mock Update (In real RL, you'd use a buffer here)
        # We just do a dummy backward pass to test gradients
        loss = -log_prob * reward + (value - reward)**2 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # C. Logging
        if step % 100 == 0:
            writer.add_scalar("charts/loss", loss.item(), step)
            print(f"    Step {step}/{total_steps} | Loss: {loss.item():.4f}")

        # Reset if done
        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs

    sps = int(total_steps / (time.time() - start_time))
    print(f"✅ Training Complete. Speed: {sps} Steps/Sec")

    # 4. CHECKPOINTING TEST
    print("\n💾 Step 2: Saving Model...")
    os.makedirs("models", exist_ok=True)
    save_path = f"models/{run_name}.pt"
    torch.save(agent.state_dict(), save_path)
    if os.path.exists(save_path):
        print(f"✅ Model saved to {save_path}")
    else:
        raise RuntimeError("❌ Model save failed!")

    # 5. LOADING & EVALUATION TEST
    print("\nRun Step 3: Loading & Evaluation...")
    eval_agent = SimpleActorCritic(obs_dim, act_dim).to(device)
    eval_agent.load_state_dict(torch.load(save_path))
    eval_agent.eval()

    eval_env = gym.make(env_id, render_mode="rgb_array") # Headless safe
    obs, _ = eval_env.reset()
    done = False
    total_reward = 0
    
    while not done:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            action, _, _ = eval_agent.get_action(obs_tensor)
        obs, reward, terminated, truncated, _ = eval_env.step(action.item())
        total_reward += reward
        done = terminated or truncated
    
    print(f"✅ Eval Episode Reward: {total_reward}")
    writer.close()
    print("\n🎉 PIPELINE TEST PASSED! You are ready to develop.")

if __name__ == "__main__":
    run_pipeline_test()