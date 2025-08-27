import gym
import numpy as np


# 创建环境
env = gym.make("NashProtecting-v0")
obs, info = env.reset()
print("初始观测：", obs)

for step in range(50):
    # action 无实际意义，由 NashTrackingEnv 内部替换
    obs, rew, done, trunc, info = env.step(0)
    u1, u2 = info['nash_u1'], info['nash_u2']
    print(f"Step {step:02d}: u1 = {np.round(u1,3)}, u2 = {np.round(u2,3)}, reward = {rew:.2f}")
    if done or trunc:
        break

env.close()