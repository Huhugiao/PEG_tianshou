import os, torch
import numpy as np
from tianshou.data import Batch

def test(env, log_path, policy, collector, device):
    np.random.seed(None)
    print(f"Loading TorchScript agent under {log_path}")
    
    # 加载TorchScript编译后的actor网络
    actor_path = os.path.join(log_path, "actor.pt")
    actor = torch.jit.load(actor_path, map_location=device)
    actor.eval()
    
    observation, info = env.reset()
    observation = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
    env.render()
    terminated, truncated = False, False
    total_reward = 0

    while not(terminated or truncated):
        with torch.no_grad():
            logits = actor(observation)
            action_logits = logits[0]
            action = action_logits.argmax(dim=1).item()
            
        observation, reward, terminated, truncated, info = env.step(action)
        observation = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        total_reward += reward
        env.render()

    input("Press Enter to exit...")