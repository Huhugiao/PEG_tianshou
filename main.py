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
from trainer import MyOnpolicyTrainer
from tianshou.utils import TensorboardLogger, WandbLogger
import algo_config
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from policy import policy_maker
import policy_test

import wandb
wandb.tensorboard.patch(root_logdir=algo_config.logdir)

def train(active_policy="a", stage_idx: int = None):

    # 构建训练环境和测试环境（传入共享模型）
    train_envs = SubprocVectorEnv([
        lambda: gym.make(algo_config.task) 
        for _ in range(algo_config.training_num)])
    test_envs = SubprocVectorEnv([
        lambda: gym.make(algo_config.task)
        for _ in range(algo_config.training_num)])

    # 设置随机数种子
    np.random.seed(algo_config.seed)
    # torch.manual_seed(algo_config.seed)
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

    root_log = algo_config.logdir
    events_folder = "events"
    tb_log = os.path.join(root_log, events_folder)
    os.makedirs(tb_log, exist_ok=True)


    stage_policy_dir = os.path.join(root_log, "stage_policies")
    os.makedirs(stage_policy_dir, exist_ok=True)

    # 加载policy参数的通用函数
    def load_policy_state(policy, path):
        state = torch.load(path, map_location=algo_config.device)
        # 兼容直接保存policy.state_dict()或checkpoint["model"]
        if "policy_a" in state and "policy_b" in state:
            policy.load_state_dict(state)
        elif "model" in state and "policy_a" in state["model"] and "policy_b" in state["model"]:
            policy.load_state_dict(state["model"])
        else:
            raise RuntimeError("Unknown checkpoint format!")

    # watch_agent模式下加载
    if algo_config.watch_agent:
        np.random.seed(None)
        print(f"Loading agent under {root_log}")
        algo_config.mission = 1
        ckpt_path = os.path.join(root_log, "stage_policies", "policy17.pth")
        load_policy_state(policy, ckpt_path)
        policy.eval()
        env = gym.make(algo_config.task)
        collector = Collector(policy, env, exploration_noise=True)
        # Collect one episode with rendering
        result = collector.collect(n_episode=1, render=1/24)
        # 直接打印奖励信息
        print("Collect rewards:", result["rews"])
        input("Press Enter to exit...")
        return
    
    writer = SummaryWriter(tb_log)
    if algo_config.using_tensorboard:
        logger = TensorboardLogger(writer, save_interval=algo_config.save_interval)
    else:
        logger = WandbLogger(
            save_interval=algo_config.save_interval,
            name=f"{algo_config.wb_name}", 
            project=algo_config.wb_project,
            run_id=f"{algo_config.run_id}", 
        )
        logger.load(writer)

    # 存储最优记录
    def save_best_fn(policy):
        # 保存阶段模型
        if stage_idx is not None:
            stage_file = os.path.join(stage_policy_dir, f"policy{stage_idx}.pth")
            torch.save(policy.state_dict(), stage_file)
        # 保存整体模型
        best_file = os.path.join(root_log, "policy.pth")
        torch.save(policy.state_dict(), best_file)

        # 分别保存两个智能体的模型参数
        agents_dir = os.path.join(root_log, "agents")
        os.makedirs(agents_dir, exist_ok=True)
        tracker_file = os.path.join(agents_dir, f"tracker{stage_idx if stage_idx is not None else ''}.pth")
        target_file = os.path.join(agents_dir, f"target{stage_idx if stage_idx is not None else ''}.pth")
        torch.save(policy.policy_a.state_dict(), tracker_file)
        torch.save(policy.policy_b.state_dict(), target_file)

    # 终止函数
    def stop_fn(mean_reward: float, reward_std: float) -> bool:
        return ((mean_reward > 35) and (reward_std < 5)) or \
            ((mean_reward > 50) and (reward_std < 10))

    # 存储checkpoint
    def save_checkpoint_fn(epoch, env_step, gradient_step):
        ckpt_path = os.path.join(root_log, "checkpoint.pth")
        torch.save({
            "model": policy.state_dict(),
            # 如果你想恢复优化器，也可以加上
            "optim_target": optim_target.state_dict(),
            "optim_tracker": optim_tracker.state_dict()
        }, ckpt_path)
        return ckpt_path

    # 从历史记录中恢复训练（如果有设置）
    if algo_config.resume:
        print(f"Loading agent under {root_log}")
        ckpt_path = os.path.join(root_log, "checkpoint.pth")
        if os.path.exists(ckpt_path):
            load_policy_state(policy, ckpt_path)
            checkpoint = torch.load(ckpt_path, map_location=algo_config.device)
            optim_target.load_state_dict(checkpoint["optim_target"])
            optim_tracker.load_state_dict(checkpoint["optim_tracker"])
        else:
            print("Fail to restore policy and optimizer.")
    

    epoch, result = MyOnpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector, 
            max_epoch=algo_config.epoch,
            step_per_epoch=algo_config.step_per_epoch,
            episode_per_collect=64,
            repeat_per_collect=algo_config.repeat_per_collect,
            episode_per_test=32,
            batch_size=algo_config.batch_size,
            update_per_step=algo_config.update_per_step,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            resume_from_log=algo_config.resume,
            save_checkpoint_fn=save_checkpoint_fn,
            ).run()
    
    return epoch
    
    # def check_gradients(model, name="model"):
    #     has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.parameters() if p.requires_grad)
    #     requires_grad = any(p.requires_grad for p in model.parameters())
    #     print(f"{name} requires_grad={requires_grad}, has non-zero gradients={has_grad}")
    # check_gradients(policy.policy_a, "tracker")
    # check_gradients(policy.policy_b, "target")

def alt_train():
    initial_epoch = algo_config.epoch
    algo_config.resume = False  # Reset resume flag for new training
    
    for i in range(0, 30):
        if i == 0:
            algo_config.mission = 0
            active_policy = "a"
            algo_config.epoch = initial_epoch  # Set initial epoch value
        else:
            mission = 1 + ((i - 1) % 2)  # Alternates between 1 and 2
            algo_config.mission = mission
            
            # Set active policy based on mission
            if mission == 1:
                active_policy = "b"  # Target policy
                algo_config.epoch = epoch + initial_epoch
                print(f"Training target, epoch {algo_config.epoch}, stage {i}")
            else:
                active_policy = "a"  # Tracker policy
                algo_config.epoch = epoch + initial_epoch
                print(f"Training tracker, epoch {algo_config.epoch}, stage {i}")
    
        # 传入当前阶段 i，训练结束时将会保存为 policy_stage_i.pth
        epoch = train(active_policy=active_policy, stage_idx=i)
        algo_config.resume = True


if __name__ == "__main__":
    alt_train()


