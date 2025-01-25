# main.py
import argparse
import pprint
import numpy as np
import torch
from tianshou.data import Collector, PrioritizedVectorReplayBuffer
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import ActorCritic, Recurrent
from tianshou.utils.net.discrete import Actor, Critic
from config import TrainerConfig
from net import CNN, RCNN
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Independent, Normal
def test_dqn(config: TrainerConfig):
    # 获取状态和动作空间
    env = config.make_env()
    # obs_shape = env.observation_space.shape or env.observation_space.n
    obs_shape1 = (3, config.param['obs_1'], config.param['obs_1']) # A 观测
    obs_shape2 = (3, config.param['obs_2'], config.param['obs_2']) # C 观测
    act_shape = env.action_space.shape or env.action_space.n

    # 创建环境
    train_envs, test_envs = config.create_env()
    
    # 设置种子 ShmemVectorEnv报警告
    config.set_seed(train_envs)
    config.set_seed(test_envs)

    # 构建网络和策略
    # A 网络
    net1 = RCNN(
        *obs_shape1,
        option = 1,
        obs_1 = config.param['obs_1'],
        obs_2 = config.param['obs_2'],
        device=config.param['device']).to(config.param['device'])

    # C 网络
    net2 = RCNN(
        *obs_shape1,
        option = 1,
        obs_1 = config.param['obs_1'],
        obs_2 = config.param['obs_2'],
        device=config.param['device']).to(config.param['device'])
    # ------------------------------------------------------------PPO-------------------------------------------------------------
    actor = Actor(
        net1, 
        act_shape,
        device=config.param['device']).to(config.param['device'])
    
    critic = Critic(
        net2,
        device=config.param['device']).to(config.param['device'])
    actor_critic = ActorCritic(actor, critic)
    # 正交初始化
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=config.param['lr'])
    dist = torch.distributions.Categorical
    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=config.param['gamma'],
        max_grad_norm=config.param['max_grad_norm'],
        eps_clip=config.param['eps_clip'],
        vf_coef=config.param['vf_coef'],
        ent_coef=config.param['ent_coef'],
        gae_lambda=config.param['gae_lambda'],
        reward_normalization=config.param['rew_norm'],
        dual_clip=config.param['dual_clip'],
        value_clip=config.param['value_clip'],
        advantage_normalization=config.param['norm_adv'],
        recompute_advantage=config.param['recompute_adv'],
        action_space=env.action_space,
        deterministic_eval=False,
    )

    # ------------------------------------------------------------PPO-------------------------------------------------------------
    # 构建并初始化收集器
    train_collector = Collector(
        policy, 
        train_envs, 
        PrioritizedVectorReplayBuffer(
            config.param['buffer_size'],
            buffer_num=len(train_envs),
            alpha=config.param['alpha'],
            beta=config.param['beta'],
            stack_num=config.param['stack_num']
        ))
    test_collector = Collector(policy, test_envs)
    
    # 获取日志记录器
    logger = config.logger 
    logger.load(SummaryWriter(config.log_path))

    # 运行训练器-off/on
    result = OnpolicyTrainer(
        policy,
        train_collector,
        test_collector,
        config.param['epoch'],
        config.param['step_per_epoch'],
        config.param['repeat_per_collect'],
        config.param['test_num'],
        config.param['batch_size'],
        step_per_collect=config.param['step_per_collect'],
        stop_fn=config.stop_fn,
        save_best_fn=config.save_best_fn,
        save_checkpoint_fn=lambda
        epoch, env_step, gradient_step: config.save_checkpoint_fn(policy, epoch, env_step, gradient_step),
        logger=config.logger,
        show_progress = True   
    )
    
    pprint.pprint(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='z_run/param_ppo.yaml')
    args = parser.parse_args()
    config = TrainerConfig(args.config)
    test_dqn(config)
