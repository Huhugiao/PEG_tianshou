import numpy as np
from typing import Any, Callable, Dict, Optional, Tuple, Union


from tianshou.trainer.onpolicy import OnpolicyTrainer
from tianshou.trainer.utils import test_episode
from collections import deque
from tianshou.trainer.utils import gather_info

class MyOnpolicyTrainer(OnpolicyTrainer):
    """
    Custom Onpolicy Trainer using a stop_fn that accepts both best_reward and best_reward_std.
    
    The stop_fn should have the signature:
        stop_fn(best_reward: float, best_reward_std: float) -> bool
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def test_step(self) -> Tuple[Dict[str, Any], bool]:
        """Perform one testing step and use custom stop_fn with reward standard deviation."""
        assert self.episode_per_test is not None
        assert self.test_collector is not None
        stop_fn_flag = False
        test_result = test_episode(
            self.policy, self.test_collector, self.test_fn, self.epoch,
            self.episode_per_test, self.logger, self.env_step, self.reward_metric
        )
        rew, rew_std = test_result["rew"], test_result["rew_std"]
        if self.best_epoch < 0 or self.best_reward < rew:
            self.best_epoch = self.epoch
            self.best_reward = float(rew)
            self.best_reward_std = rew_std
            if self.save_best_fn:
                self.save_best_fn(self.policy)
        if self.verbose:
            print(
                f"Epoch #{self.epoch}: test_reward: {rew:.6f} ± {rew_std:.6f},"
                f" best_reward: {self.best_reward:.6f} ± {self.best_reward_std:.6f} in #{self.best_epoch}"
            )
        test_stat = {}
        if not self.is_run:
            test_stat = {
                "test_reward": rew,
                "test_reward_std": rew_std,
                "best_reward": self.best_reward,
                "best_reward_std": self.best_reward_std,
                "best_epoch": self.best_epoch
            }
        # 使用新的 stop_fn 调用，传入 reward 与 reward_std
        if self.stop_fn and self.stop_fn(self.best_reward, self.best_reward_std):
            stop_fn_flag = True

        return test_stat, stop_fn_flag

    def train_step(self) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
        """Perform one training step with custom stop_fn that considers reward standard deviation."""
        assert self.episode_per_test is not None
        assert self.train_collector is not None
        stop_fn_flag = False
        if self.train_fn:
            self.train_fn(self.epoch, self.env_step)
        result = self.train_collector.collect(
            n_step=self.step_per_collect, n_episode=self.episode_per_collect
        )
        if result["n/ep"] > 0 and self.reward_metric:
            rew = self.reward_metric(result["rews"])
            result.update(rews=rew, rew=rew.mean(), rew_std=rew.std())
        self.env_step += int(result["n/st"])
        self.logger.log_train_data(result, self.env_step)
        self.last_rew = result["rew"] if result["n/ep"] > 0 else self.last_rew
        self.last_len = result["len"] if result["n/ep"] > 0 else self.last_len
        data = {
            "env_step": str(self.env_step),
            "rew": f"{self.last_rew:.2f}",
            "len": str(int(self.last_len)),
            "n/ep": str(int(result["n/ep"])),
            "n/st": str(int(result["n/st"])),
        }
        if result["n/ep"] > 0:
            if self.test_in_train and self.stop_fn and self.stop_fn(result["rew"], result["rew_std"]):
                assert self.test_collector is not None
                test_result = test_episode(
                    self.policy, self.test_collector, self.test_fn, self.epoch,
                    self.episode_per_test, self.logger, self.env_step, self.reward_metric
                )
                if self.stop_fn(test_result["rew"], test_result["rew_std"]):
                    stop_fn_flag = True
                    self.best_reward = test_result["rew"]
                    self.best_reward_std = test_result["rew_std"]
                else:
                    self.policy.train()
        return data, result, stop_fn_flag
    
    def run(self) -> Tuple[int, Dict[str, Union[float, str]]]:
        """Consume iterator and return the current epoch along with training info."""
        try:
            self.is_run = True
            deque(self, maxlen=0)  # Exhaust iterator.
            info = gather_info(
                self.start_time, self.train_collector, self.test_collector,
                self.best_reward, self.best_reward_std
            )
        finally:
            self.is_run = False

        return self.epoch, info