from tianshou.policy import DiscreteSACPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tsppo import Actor, Critic
from tianshou.utils.net.common import Net
from tianshou.env import SubprocVectorEnv
from tianshou.data import PrioritizedVectorReplayBuffer, VectorReplayBuffer, Batch, Collector
from tianshou.utils import WandbLogger
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
import algo_config
import gym
import datetime
import pickle
import os
import pprint
import sys
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def train_sac():
    env = gym.make(algo_config.task, cl_flag=algo_config.cl_flag,
            target_mode="Fix", obstacle_mode="Dynamic", training_stage=algo_config.training_stage)
    algo_config.state_shape = env.observation_space.shape
    algo_config.action_shape = env.action_space.n

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
    test_envs.seed(algo_config.seed)

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
    
    actor = Actor(
        net,
        algo_config.action_shape,
        device=algo_config.device,
        )
    actor = Actor(net, algo_config.action_shape, device=algo_config.device, softmax_output=False)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=algo_config.lr)
    critic1 = Critic(net, last_size=algo_config.action_shape, device=algo_config.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=algo_config.lr)
    critic2 = Critic(net, last_size=algo_config.action_shape, device=algo_config.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=algo_config.lr)

    # define policy
    # if algo_config.auto_alpha:
    #     target_entropy = 0.98 * np.log(np.prod(algo_config.action_shape))
    #     log_alpha = torch.zeros(1, requires_grad=True, device=algo_config.device)
    #     alpha_optim = torch.optim.Adam([log_alpha], lr=algo_config.lr)
    #     algo_config.alpha = (target_entropy, log_alpha, alpha_optim)

    policy = DiscreteSACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic1=critic1,
        critic1_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        action_space=env.action_space,
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
    # writer = SummaryWriter(log_path)
    # logger = TensorboardLogger(writer, save_interval=algo_config.save_interval)

    logger = WandbLogger(
    save_interval=algo_config.save_interval,
    name="your_experiment_name",  # 实验名称
    project="your_project_name",  # 项目名称
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
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        # Example: saving by epoch num
        # ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        torch.save(
            {
                "model": policy.state_dict(),
            },
            ckpt_path,
        )
        buffer_path = os.path.join(log_path, "train_buffer.pkl")
        with open(buffer_path, "wb") as f:
            pickle.dump(train_collector.buffer, f)
        return ckpt_path


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

    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=algo_config.epoch,
        step_per_epoch=algo_config.step_per_epoch,
        step_per_collect=algo_config.step_per_collect,
        episode_per_test=algo_config.test_num,
        batch_size=algo_config.batch_size,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        update_per_step=algo_config.update_per_step,
        test_in_train=False,
        save_checkpoint_fn=save_checkpoint_fn,
    ).run()

    pprint.pprint(result)


if __name__ == "__main__":
    # algo_config.resume = False
    # for stage in range(1,6):
    #     epoch_nums = [100,200,300,400,500]
    #     algo_config.epoch = epoch_nums[stage-1]
    #     algo_config.training_stage = stage
    #     Train()
    #     algo_config.resume = True
    train_sac()

