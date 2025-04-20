import os
import pickle
import pprint
import gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import PrioritizedVectorReplayBuffer, VectorReplayBuffer, Batch, Collector
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import OnpolicyTrainer
from tianshou.policy import PPOPolicy
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net, ActorCritic
import algo_config
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from policy import policy_maker
import policy_test

def train(active_policy="a"):

    # 构建训练环境和测试环境（传入共享模型）
    train_envs = SubprocVectorEnv([
        lambda: gym.make(algo_config.task) 
        for _ in range(algo_config.training_num)])
    test_envs = SubprocVectorEnv([
        lambda: gym.make(algo_config.task)
        for _ in range(algo_config.training_num)])

    # 设置随机数种子
    np.random.seed(algo_config.seed)
    torch.manual_seed(algo_config.seed)
    train_envs.seed(algo_config.seed)
    test_envs.seed(algo_config.seed)

    policy, optim_target, optim_tracker = policy_maker()
    policy.set_active_policy(active_policy)

    # buffer
    if algo_config.prioritized_replay:
        buf = PrioritizedVectorReplayBuffer(
            algo_config.buffer_size,
            buffer_num=len(train_envs),
            alpha=algo_config.alpha,
            beta=algo_config.beta,
            weight_norm=True,
        )
    else:
        buf = VectorReplayBuffer(algo_config.buffer_size, buffer_num=len(train_envs))

    # collector
    train_collector = Collector(policy, train_envs, buf, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    # 预采集
    train_collector.reset()
    train_collector.collect(n_step=algo_config.batch_size * algo_config.training_num)

    # 设置数据存储路径
    log_path = algo_config.logdir
    # log_path = os.path.join(
    #     algo_config.alt_logdir if algo_config.mission == 1 
    #     else algo_config.logdir)

    if not algo_config.watch_agent:
        if algo_config.using_tensorboard:
            writer = SummaryWriter(log_path)
            logger = TensorboardLogger(writer, save_interval=algo_config.save_interval)
        else:
            # Determine WandB parameters based on mission
            if algo_config.mission == 1:
                wb_name = algo_config.alt_wb_name
                run_id = algo_config.alt_run_id
            else:
                wb_name = algo_config.wb_name
                run_id = algo_config.run_id
                
            logger = WandbLogger(
                save_interval=algo_config.save_interval,
                name=wb_name,
                project=algo_config.wb_project,
                run_id=run_id,
            )
            logger.load(SummaryWriter(log_path))

    # 存储最优记录
    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    # 终止函数
    def stop_fn(mean_rewards):
        return mean_rewards >= algo_config.reward_threshold

    # 存储checkpoint
    def save_checkpoint_fn(epoch, env_step, gradient_step):
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        # Save both policies' state
        torch.save({
            "model": policy.state_dict(),
            "optim_target": optim_target.state_dict(),
            "optim_tracker": optim_tracker.state_dict()
        }, ckpt_path)

    # 观察一次智能体仿真（如果有设置）
    if algo_config.watch_agent:
        np.random.seed(None)
        print(f"Loading agent under {log_path}")
        ckpt_path = os.path.join(log_path, "policy.pth")
        policy.load_state_dict(torch.load(ckpt_path, map_location=algo_config.device))
        policy.eval()
        env = gym.make(algo_config.task)
        collector = Collector(policy, env, exploration_noise=True)
        collector.collect(n_episode=1, render=1/24)
        input("Press Enter to exit...")
        return

    # 从历史记录中恢复训练（如果有设置）
    if algo_config.resume:
        print(f"Loading agent under {log_path}")
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=algo_config.device)
            policy.load_state_dict(checkpoint["model"])
            
            # Load the appropriate optimizer based on active policy
            if policy.active_policy == "a":
                optim_target.load_state_dict(checkpoint["optim_target"])
                print("Successfully restored target policy and optimizer.")
            else:
                optim_tracker.load_state_dict(checkpoint["optim_tracker"])
                print("Successfully restored tracker policy and optimizer.")
        else:
            print("Fail to restore policy and optimizer.")
    

    result = OnpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector, 
            max_epoch=algo_config.epoch,
            step_per_epoch=algo_config.step_per_epoch,
            step_per_collect=algo_config.step_per_collect,
            repeat_per_collect=algo_config.repeat_per_collect,
            episode_per_test=algo_config.test_num,
            batch_size=algo_config.batch_size,
            update_per_step=algo_config.update_per_step,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            resume_from_log=algo_config.resume,
            save_checkpoint_fn=save_checkpoint_fn,
            ).run()
    
    def check_gradients(model, name="model"):
        has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.parameters() if p.requires_grad)
        requires_grad = any(p.requires_grad for p in model.parameters())
        print(f"{name} requires_grad={requires_grad}, has non-zero gradients={has_grad}")
    check_gradients(policy.policy_a, "tracker")
    check_gradients(policy.policy_b, "target")

def alt_train():
    initial_epoch = algo_config.epoch
    algo_config.resume = False  # Reset resume flag for new training
    
    for i in range(0, 5):
        if i == 0:
            algo_config.mission = 0
            active_policy = "a"
            algo_config.epoch = initial_epoch  # Set initial epoch value
        else:
            mission = 1 + ((i - 1) % 2)  # Alternates between 1 and 2
            algo_config.mission = mission
            
            # Increase epoch with each mission switch
            if mission == 2:
                algo_config.epoch += algo_config.epoch
            
            # Set active policy based on mission
            if mission == 1:
                active_policy = "b"  # Target policy
                print(f"Training target policy with epoch {algo_config.epoch}")
            else:
                active_policy = "a"  # Tracker policy
                print(f"Training tracker policy with epoch {algo_config.epoch}")
        
        # Pass the active policy to train
        train(active_policy=active_policy)
        
        algo_config.resume = True


if __name__ == "__main__":
    alt_train()


