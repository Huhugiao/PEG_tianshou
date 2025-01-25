# main.py
import argparse
import pprint
import numpy as np
from tianshou.data import Collector, PrioritizedVectorReplayBuffer
from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net
from torch.optim import Adam
from config import TrainerConfig
from net import CNN, CNN_LSTM
from torch.utils.tensorboard import SummaryWriter

def test_dqn(config: TrainerConfig):
    # 获取状态和动作空间
    env = config.make_env()
    obs_shape = env.observation_space.shape or env.observation_space.n
    act_shape = env.action_space.shape or env.action_space.n

    # 创建环境
    train_envs, test_envs = config.create_env()

    # 设置种子 ShmemVectorEnv报警告
    config.set_seed(train_envs)
    config.set_seed(test_envs)

    # 构建网络和策略
    net = CNN(
        *obs_shape,
        action_shape = act_shape,
        device=config.param['device']).to(config.param['device'])
    optim = Adam(net.parameters(), lr=config.param['lr'])

    # ------------------------------------------------------------DQN-------------------------------------------------------------
    policy = DQNPolicy(
        net,
        optim,
        config.param['gamma'],
        config.param['n_step'],
        target_update_freq=config.param['target_update_freq'],
    )
        
    # 构建经验回放缓冲区-DQN
    buf = PrioritizedVectorReplayBuffer(
        config.param['buffer_size'],
        buffer_num=len(train_envs),
        alpha=config.param['alpha'],
        beta=config.param['beta'],
        stack_num=config.param['stack_num']
    )

    # 定义训练过程中的函数-DQN
    def train_fn_inner(epoch, env_step):
        if env_step <= 1e4:
            policy.set_eps(config.param['eps_train'])
        elif env_step <= 5e4:
            eps = config.param['eps_train'] - (env_step - 1e4) / 4e4 * (0.9 * config.param['eps_train'])
            policy.set_eps(eps)
        else:
            policy.set_eps(0.1 * config.param['eps_train'])
    
    # 定义测试过程中的函数-DQN
    def test_fn_inner(epoch, env_step):
        policy.set_eps(config.param['eps_test'])
    # -------------------------------------------------------------DQN-------------------------------------------------------------

    # 构建并初始化收集器
    train_collector = Collector(policy, train_envs, buf, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    train_collector.collect(n_step=config.param['batch_size'] * config.param['training_num'])

    # 获取日志记录器
    logger = config.logger 
    logger.load(SummaryWriter(config.log_path))

    # 运行训练器-off/on
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        config.param['epoch'],
        config.param['step_per_epoch'],
        config.param['step_per_collect'],
        config.param['test_num'],
        config.param['batch_size'],
        update_per_step=config.param['update_per_step'],
        train_fn=train_fn_inner,
        test_fn=test_fn_inner,
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
    parser.add_argument('--config', type=str, default='z_run/param_dqn.yaml')
    args = parser.parse_args()
    config = TrainerConfig(args.config)
    test_dqn(config)
