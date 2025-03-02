import os
import pickle
import pprint
import gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import PrioritizedVectorReplayBuffer, VectorReplayBuffer, Batch, Collector
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import OnpolicyTrainer
from tianshou.policy import PPOPolicy
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net, ActorCritic
from typing import Any, Callable, Dict, List, Optional, Union, Sequence, Tuple
import algo_config
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from tsppo import Actor, Critic


def Train():
    env = gym.make(algo_config.task, cl_flag=algo_config.cl_flag,
            target_mode="Fix", obstacle_mode="Dynamic", training_stage=algo_config.training_stage)
    algo_config.state_shape = env.observation_space.shape
    algo_config.action_shape = env.action_space.n

    # 指定奖励上限
    if algo_config.reward_threshold is None:
        default_reward_threshold = {"TrackingEnv-v0": 500000}
        algo_config.reward_threshold = default_reward_threshold.get(algo_config.task, env.spec.reward_threshold)

    # 构建训练环境和测试环境
    train_envs = SubprocVectorEnv(
        [lambda: gym.make(algo_config.task, cl_flag=algo_config.cl_flag,
            target_mode="Fix", obstacle_mode="Dynamic", training_stage=algo_config.training_stage) for _ in
         range(algo_config.training_num)])
    test_envs = SubprocVectorEnv(
        [lambda: gym.make(algo_config.task, cl_flag=algo_config.cl_flag,
            target_mode="Fix", obstacle_mode="Dynamic", training_stage=algo_config.training_stage) for _ in
         range(algo_config.test_num)])

    # 设置随机数种子
    np.random.seed(algo_config.seed)
    torch.manual_seed(algo_config.seed)
    train_envs.seed(algo_config.seed)
    # test_envs.seed(algo_config.seed)

    # 构建网络
    net = Net(
        algo_config.state_shape[0]-algo_config.god_view_shape[0],
        algo_config.action_shape,
        hidden_sizes=algo_config.hidden_sizes,
        device=algo_config.device,
        )

    net_c = Net(
        algo_config.state_shape,
        algo_config.action_shape,
        hidden_sizes=algo_config.hidden_sizes,
        device=algo_config.device,
        )
    
    if algo_config.use_god_view != True:
        net_c = net

    # 建立Actor-Critic实例
    actor = Actor(
        net,
        algo_config.action_shape,
        device=algo_config.device,
        )
    critic = Critic(
        net_c,  
        device=algo_config.device,
        )
    actor_critic = ActorCritic(actor, critic).to(algo_config.device)

    # 正交初始化
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    # 优化器
    optim = torch.optim.Adam(actor_critic.parameters(), lr=algo_config.lr)
    dist = torch.distributions.Categorical
    # 策略
    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=algo_config.gamma,
        max_grad_norm=algo_config.max_grad_norm,
        eps_clip=algo_config.eps_clip,
        vf_coef=algo_config.vf_coef,
        ent_coef=algo_config.ent_coef,
        gae_lambda=algo_config.gae_lambda,
        reward_normalization=algo_config.reward_normalization,
        dual_clip=algo_config.dual_clip,
        value_clip=algo_config.value_clip,
        advantage_normalization=algo_config.norm_adv,
        recompute_advantage=algo_config.recompute_adv,
        action_space=env.action_space,
        deterministic_eval=False,
    ).to(algo_config.device)

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
    log_path = os.path.join(algo_config.logdir)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer, save_interval=algo_config.save_interval)

    # 存储最优记录
    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    # 终止函数
    def stop_fn(mean_rewards):
        return mean_rewards >= algo_config.reward_threshold

    # 存储checkpoint
    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        # Example: saving by epoch num
        # ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        torch.save(
            {
                "model": policy.state_dict(),
                "optim": optim.state_dict(),
            },
            ckpt_path,
        )
        buffer_path = os.path.join(log_path, "train_buffer.pkl")
        with open(buffer_path, "wb") as f:
            pickle.dump(train_collector.buffer, f)
        return ckpt_path

    # 设置训练eps
    def train_fn(epoch, env_step):
        if env_step <= 10000:
            policy.ent_coef = 0.01
        elif env_step <= 50000:
            policy.ent_coef = 0.005
        else:
            policy.ent_coef = 0.001

        # beta annealing, just a demo
        if algo_config.prioritized_replay:
            if env_step <= 10000:
                beta = algo_config.beta
            elif env_step <= 50000:
                beta = algo_config.beta - (env_step - 10000) / 40000 * (algo_config.beta - algo_config.beta_final)
            else:
                beta = algo_config.beta_final
            buf.set_beta(beta)

    # 从历史记录中恢复训练（如果有设置）
    if algo_config.resume:
        print(f"Loading agent under {log_path}")
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=algo_config.device)
            policy.load_state_dict(checkpoint["model"])
            policy.optim.load_state_dict(checkpoint["optim"])
            print("Successfully restore policy and optim.")
        else:
            print("Fail to restore policy and optim.")
        buffer_path = os.path.join(log_path, "train_buffer.pkl")
        if os.path.exists(buffer_path):
            with open(buffer_path, "rb") as f:
                train_collector.buffer = pickle.load(f)
            print("Successfully restore buffer.")
        else:
            print("Fail to restore buffer.")

    # 观察一次智能体仿真（如果有设置）
    if algo_config.watch_agent:
        print(f"Loading agent under {log_path}")
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=algo_config.device)
            policy.load_state_dict(checkpoint["model"])
            policy.optim.load_state_dict(checkpoint["optim"])
            print("Successfully restore policy and optim.")
        else:
            print("Fail to restore policy and optim.")
        policy.eval()
        collector = Collector(policy, env, exploration_noise=True)
        collector.collect(n_episode=1, render=1/10)
        return
        
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
            train_fn=train_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            resume_from_log=algo_config.resume,
            save_checkpoint_fn=save_checkpoint_fn,
            ).run()

if __name__ == "__main__":
    algo_config.resume = False
    for stage in range(1,6):
        epoch_nums = [100,200,300,400,500]
        algo_config.epoch = epoch_nums[stage-1]
        algo_config.training_stage = stage
        Train()
        algo_config.resume = True


