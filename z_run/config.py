# config.py
import yaml
import torch
import os
import datetime
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.env import ShmemVectorEnv
import gymnasium as gym
import numpy as np
from minigrid.wrappers import ImgObsWrapper, ViewSizeWrapper, FullyObsWrapper, GlobalLocalWrapper, NoDeath

    
class TrainerConfig:
    def __init__(self, param_path):
        self.param = self.load_config(param_path)
        self.current_time = datetime.datetime.now().strftime("%m%d_%H%M")
        self.log_name = os.path.join(self.param['algo_name'], self.current_time)
        self.log_path = os.path.join(self.param['logdir'], self.log_name)
        self.logger = self.create_logger()

    def make_env(self):
        env = gym.make(self.param['task']).unwrapped
        # env = FullyObsWrapper(env)
        # env = ImgObs4chwWrapper(env)
        env = GlobalLocalWrapper(env, agent_view_size = self.param['obs_1'])
        env = NoDeath(env, no_death_types=("ball",), death_cost=-0.5)
        return env
    
    def load_config(self, param_path):
        with open(param_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def create_env(self):
        train_envs = ShmemVectorEnv([lambda: self.make_env() for _ in range(self.param['training_num'])])
        test_envs = ShmemVectorEnv([lambda: self.make_env() for _ in range(self.param['test_num'])])
        return train_envs, test_envs
    
    def set_seed(self, envs):
        np.random.seed(self.param['seed'])
        torch.manual_seed(self.param['seed'])
        envs.seed(self.param['seed'])
    
    def create_logger(self):
        if self.param['logger'] == "wandb":
            logger = WandbLogger(
                save_interval=1,
                name=self.log_name.replace(os.path.sep, "_"),
                run_id=self.param.get('resume_id'),
                config=self.param,
                project=self.param['task'],
            )
        else:  # tensorboard
            writer = SummaryWriter(self.log_path)
            writer.add_text("args", str(self.param))
            logger = TensorboardLogger(writer)
        return logger
    
    def save_best_fn(self, policy):
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        torch.save(policy.state_dict(), os.path.join(self.log_path, f"Best_{self.current_time}.pth"))
    
    def save_checkpoint_fn(self, policy, epoch, env_step, gradient_step):
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        ckpt_path = os.path.join(self.log_path, f"checkpoint_{self.current_time}_{epoch}.pth")
        torch.save(policy.state_dict(), ckpt_path)
        return ckpt_path
    
    def stop_fn(self, mean_rewards):
        return mean_rewards >= self.param['reward_threshold']


