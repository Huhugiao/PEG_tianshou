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

def train():

    opponent_policy = None
    if algo_config.mission in [1, 2]:
        policy_path = (os.path.join(algo_config.logdir, "actor.pt")
                       if algo_config.mission == 1
                       else os.path.join(algo_config.alt_logdir, "actor.pt"))
        # 加载并冻结对手模型，加快推理速度，减少阻塞
        opponent_policy = torch.jit.load(policy_path, map_location=algo_config.device)
        opponent_policy.eval()
        opponent_policy = torch.jit.freeze(opponent_policy)

    # 构建训练环境和测试环境（传入共享模型）
    train_envs = SubprocVectorEnv([
        lambda: gym.make(algo_config.task, opponent_policy=opponent_policy) 
        for _ in range(algo_config.training_num)])
    test_envs = SubprocVectorEnv([
        lambda: gym.make(algo_config.task, opponent_policy=opponent_policy)
        for _ in range(algo_config.training_num)])

    # 设置随机数种子
    np.random.seed(algo_config.seed)
    torch.manual_seed(algo_config.seed)
    train_envs.seed(algo_config.seed)
    test_envs.seed(algo_config.seed)

    policy, optim = policy_maker()

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
    log_path = os.path.join(
        algo_config.alt_logdir if algo_config.mission == 1 
        else algo_config.logdir)

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
        # 确保示例输入在CPU上生成
        example_obs = torch.rand(
            1, 
            algo_config.base_obs_dim + algo_config.god_view_dim,
            device='cpu'  # 强制使用CPU生成示例输入
        )
        # 导出前将模型转到CPU
        traced_actor = torch.jit.trace(policy.actor.to('cpu'), example_obs, strict=False)
        # 保存前再次确认设备
        traced_actor = traced_actor.cpu()
        traced_actor.save(os.path.join(log_path, "actor.pt"))
        # 恢复原始模型设备
        policy.actor.to(algo_config.device)

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

    # 观察一次智能体仿真（如果有设置）\
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
    
    if 0:
        env = gym.make(algo_config.task)
        policy_test.test(env, log_path, policy, test_collector, algo_config.device)
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
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            resume_from_log=algo_config.resume,
            save_checkpoint_fn=save_checkpoint_fn,
            ).run()

def alt_train():
    for i in range(0, 20):
        if i == 0:
            algo_config.mission = 0
        else:
            algo_config.mission = 1 + ((i - 1) % 2)  # Alternates between 1 and 2 after the first iteration
        train()
        algo_config.resume = True

if __name__ == "__main__":
    alt_train()


