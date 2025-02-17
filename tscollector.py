import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import gym
import numpy as np
import torch

from tianshou.data import (
    Batch,
    CachedReplayBuffer,
    ReplayBuffer,
    ReplayBufferManager,
    VectorReplayBuffer,
    to_numpy,
)
from tianshou.env import BaseVectorEnv, DummyVectorEnv
from tianshou.policy import BasePolicy


class Collector(object):
    """Collector 使得策略可以在不同类型的环境中通过精确的步数或回合数与环境进行交互。

    :param policy: 一个 :class:`~tianshou.policy.BasePolicy` 类的实例。
    :param env: 一个 ``gym.Env`` 环境或一个 :class:`~tianshou.env.BaseVectorEnv` 类的实例。
    :param buffer: 一个 :class:`~tianshou.data.ReplayBuffer` 类的实例。
        如果设置为 None，则不会存储数据。默认为 None。
    :param function preprocess_fn: 一个函数，在数据被添加到缓冲区之前调用，详见 issue #42 和 :ref:`preprocess_fn`。默认为 None。
    :param bool exploration_noise: 确定动作是否需要被对应的策略的探索噪声修改。如果是，则自动调用 "policy.exploration_noise(act, batch)" 在动作中添加探索噪声。默认为 False。

    preprocess_fn 是一个函数，在数据被添加到缓冲区之前调用，以批量格式接收数据。
    当 Collector 重置环境时，它将只接收到 "obs" 和 "env_id"。在正常环境步骤中，
    它将接收到键 "obs_next", "rew", "terminated", "truncated", "info", "policy" 和 "env_id"。
    也可以使用键 "obs_next", "rew", "done", "info", "policy" 和 "env_id"。
    它返回一个字典或一个 :class:`~tianshou.data.Batch` 类型的对象，带有修改后的键和值。
    示例见 "test/base/test_collector.py"。

    .. note::

        请确保给定的环境有一个时间限制，如果使用 n_episode 收集选项。

    .. note::

        在 Tianshou 的早期版本中，传递给 `__init__` 的重放缓冲区会自动重置。
        目前的实现中不会这样做。
    """

    def __init__(
        self,
        policy: BasePolicy,
        env: Union[gym.Env, BaseVectorEnv],
        buffer: Optional[ReplayBuffer] = None,
        preprocess_fn: Optional[Callable[..., Batch]] = None,
        exploration_noise: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(env, gym.Env) and not hasattr(env, "__len__"):
            warnings.warn("检测到单个环境，将其包装为 DummyVectorEnv。")
            self.env = DummyVectorEnv([lambda: env])  # type: ignore
        else:
            self.env = env  # type: ignore
        self.env_num = len(self.env)
        self.exploration_noise = exploration_noise
        self._assign_buffer(buffer)
        self.policy = policy
        self.preprocess_fn = preprocess_fn
        self._action_space = self.env.action_space
        # 避免在 __init__ 之外创建属性
        self.reset(False)

    def _assign_buffer(self, buffer: Optional[ReplayBuffer]) -> None:
        """检查缓冲区是否符合约束。"""
        if buffer is None:
            buffer = VectorReplayBuffer(self.env_num, self.env_num)
        elif isinstance(buffer, ReplayBufferManager):
            assert buffer.buffer_num >= self.env_num
            if isinstance(buffer, CachedReplayBuffer):
                assert buffer.cached_buffer_num >= self.env_num
        else:  # ReplayBuffer 或 PrioritizedReplayBuffer
            assert buffer.maxsize > 0
            if self.env_num > 1:
                if type(buffer) == ReplayBuffer:
                    buffer_type = "ReplayBuffer"
                    vector_type = "VectorReplayBuffer"
                else:
                    buffer_type = "PrioritizedReplayBuffer"
                    vector_type = "PrioritizedVectorReplayBuffer"
                raise TypeError(
                    f"无法使用 {buffer_type}(size={buffer.maxsize}, ...) 来收集 "
                    f"{self.env_num} 个环境的数据，\n\t请使用 {vector_type}(total_size="
                    f"{buffer.maxsize}, buffer_num={self.env_num}, ...) 代替。"
                )
        self.buffer = buffer

    def reset(
        self,
        reset_buffer: bool = True,
        gym_reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """重置环境、统计数据、当前数据和可能的重放缓冲区。

        :param bool reset_buffer: 如果为真，则重置附加到 Collector 的重放缓冲区。
        :param gym_reset_kwargs: 额外的关键字参数传递给环境的 reset 函数。默认为 None。
        """
        # 使用空 Batch 来避免 self.data 支持切片时出现问题
        self.data = Batch(
            obs={},
            act={},
            rew={},
            terminated={},
            truncated={},
            done={},
            obs_next={},
            info={},
            policy={},
            god_view=np.zeros(7, dtype=np.float32),  # 新增 god_view 字段来存储 godview 信息
        )
        self.reset_env(gym_reset_kwargs)
        if reset_buffer:
            self.reset_buffer()
        self.reset_stat()

    def reset_stat(self) -> None:
        """重置统计变量。"""
        self.collect_step, self.collect_episode, self.collect_time = 0, 0, 0.0

    def reset_buffer(self, keep_statistics: bool = False) -> None:
        """重置数据缓冲区。"""
        self.buffer.reset(keep_statistics=keep_statistics)

    def reset_env(self, gym_reset_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """重置所有环境。"""
        gym_reset_kwargs = gym_reset_kwargs if gym_reset_kwargs else {}
        rval = self.env.reset(**gym_reset_kwargs)
        returns_info = isinstance(rval, (tuple, list)) and len(rval) == 2 and (
            isinstance(rval[1], dict) or isinstance(rval[1][0], dict)
        )
        if returns_info:
            obs, info = rval
            god_view = info.get('god_view_info', None) if 'god_view_info' in info else {}
            if self.preprocess_fn:
                processed_data = self.preprocess_fn(
                    obs=obs, info=info, env_id=np.arange(self.env_num)
                )
                obs = processed_data.get("obs", obs)
                info = processed_data.get("info", info)
                god_view = processed_data.get("god_view", god_view)
                if god_view.size == 0:
                    god_view = np.zeros(7, dtype=np.float32)
                print(f"god_view = {god_view}")  # 调试信息
            self.data.info = info
            self.data.god_view = god_view
        else:
            obs = rval
            if self.preprocess_fn:
                obs = self.preprocess_fn(obs=obs, env_id=np.arange(self.env_num
                                                                   )).get("obs", obs)
        self.data.obs = obs

    def _reset_state(self, id: Union[int, List[int]]) -> None:
        """重置隐藏状态：self.data.policy[id]."""
        if hasattr(self.data.policy, "hidden_state"):
            state = self.data.policy.hidden_state  # 这是一个引用
            if isinstance(state, torch.Tensor):
                state[id].zero_()
            elif isinstance(state, np.ndarray):
                state[id] = None if state.dtype == object else 0
            elif isinstance(state, Batch):
                state.empty_(id)

    def _reset_env_with_ids(
        self,
        local_ids: Union[List[int], np.ndarray],
        global_ids: Union[List[int], np.ndarray],
        gym_reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        gym_reset_kwargs = gym_reset_kwargs if gym_reset_kwargs else {}
        rval = self.env.reset(global_ids, **gym_reset_kwargs)
        returns_info = isinstance(rval, (tuple, list)) and len(rval) == 2 and (
            isinstance(rval[1], dict) or isinstance(rval[1][0], dict)
        )
        if returns_info:
            obs_reset, info = rval
            god_view = info.get('god_view_info', None) if 'god_view_info' in info else {}
            if god_view.size == 0:
                god_view = np.zeros(7, dtype=np.float32)
            print(f"god_view = {god_view}")  # 调试信息
            if self.preprocess_fn:
                processed_data = self.preprocess_fn(
                    obs=obs_reset, info=info, env_id=global_ids
                )
                obs_reset = processed_data.get("obs", obs_reset)
                info = processed_data.get("info", info)
                god_view = processed_data.get("god_view", god_view)
                
            self.data.info[local_ids] = info
            self.data.god_view[local_ids] = god_view
        else:
            obs_reset = rval
            if self.preprocess_fn:
                obs_reset = self.preprocess_fn(obs=obs_reset, env_id=global_ids
                                               ).get("obs", obs_reset)
        self.data.obs_next[local_ids] = obs_reset

    def collect(
        self,
        n_step: Optional[int] = None,
        n_episode: Optional[int] = None,
        random: bool = False,
        render: Optional[float] = None,
        no_grad: bool = True,
        gym_reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """收集指定数量的步数或回合。

        为了确保使用 n_episode 选项时采样的无偏结果，此函数将首先收集 ``n_episode - env_num`` 个回合，
        然后在最后 ``env_num`` 个回合中，它们将从每个环境中均匀收集。

        :param int n_step: 您希望收集的步数。
        :param int n_episode: 您希望收集的回合数。
        :param bool random: 是否使用随机策略收集数据。默认为 False。
        :param float render: 渲染连续帧之间的休眠时间。
            默认为 None（不渲染）。
        :param bool no_grad: 是否在 policy.forward() 中保留梯度。默认为 True（不保留梯度）。
        :param gym_reset_kwargs: 额外的关键字参数传递给环境的 reset 函数。
            默认为 None（额外的关键字参数）

        .. note::

            只能指定一个收集数量规范，即 ``n_step`` 或 ``n_episode``。

        :return: 包含以下键的字典

            * ``n/ep`` 收集的回合数。
            * ``n/st`` 收集的步数。
            * ``rews`` 收集的回合奖励数组。
            * ``lens`` 收集的回合长度数组。
            * ``idxs`` 收集的回合在缓冲区中的起始索引数组。
            * ``rew`` 平均回合奖励。
            * ``len`` 平均回合长度。
            * ``rew_std`` 回合奖励的标准误差。
            * ``len_std`` 回合长度的标准误差。
        """
        assert not self.env.is_async, "如果使用异步 venv，请使用 AsyncCollector。"
        if n_step is not None:
            assert n_episode is None, (
                f"在 Collector.collect() 中只能指定 n_step 或 n_episode 中的一个。"
                f"获取 n_step={n_step}, n_episode={n_episode}。"
            )
            assert n_step > 0
            if not n_step % self.env_num == 0:
                warnings.warn(
                    f"n_step={n_step} 不是 #env ({self.env_num}) 的倍数，"
                    "这可能会导致额外的转换存储到缓冲区。"
                )
            ready_env_ids = np.arange(self.env_num)
        elif n_episode is not None:
            assert n_episode > 0
            ready_env_ids = np.arange(min(self.env_num, n_episode))
            self.data = self.data[:min(self.env_num, n_episode)]
        else:
            raise TypeError(
                "请在 collector.collect() 中指定至少一个（n_step 或 n_episode）。"
            )

        start_time = time.time()

        step_count = 0
        episode_count = 0
        episode_rews = []
        episode_lens = []
        episode_start_indices = []

        while True:
            assert len(self.data) == len(ready_env_ids)
            # 恢复状态：如果最后一个状态是 None，将不会存储
            last_state = self.data.policy.pop("hidden_state", None)

            # 获取下一个动作
            if random:
                try:
                    act_sample = [
                        self._action_space[i].sample() for i in ready_env_ids
                    ]
                except TypeError:  # envpool 的动作空间不是每个环境的
                    act_sample = [self._action_space.sample() for _ in ready_env_ids]
                act_sample = self.policy.map_action_inverse(act_sample)  # type: ignore
                self.data.update(act=act_sample)
            else:
                if no_grad:
                    with torch.no_grad():  # 更快的速度
                        result = self.policy(self.data, last_state)
                else:
                    result = self.policy(self.data, last_state)
                # 更新 state / act / policy into self.data
                policy = result.get("policy", Batch())
                assert isinstance(policy, Batch)
                state = result.get("state", None)
                if state is not None:
                    policy.hidden_state = state  # 将状态保存到缓冲区
                act = to_numpy(result.act)
                if self.exploration_noise:
                    act = self.policy.exploration_noise(act, self.data)
                self.data.update(policy=policy, act=act)

            # 获取边界和重新映射的动作（不保存到缓冲区）
            action_remap = self.policy.map_action(self.data.act)
            # 在环境中执行步操作
            result = self.env.step(action_remap, ready_env_ids)  # type: ignore
            if len(result) == 5:
                obs_next, rew, terminated, truncated, info = result
                done = np.logical_or(terminated, truncated)
            elif len(result) == 4:
                obs_next, rew, done, info = result
                if isinstance(info, dict):
                    truncated = info["TimeLimit.truncated"]
                else:
                    truncated = np.array(
                        [
                            info_item.get("TimeLimit.truncated", False)
                            for info_item in info
                        ]
                    )
                terminated = np.logical_and(done, ~truncated)
            else:
                raise ValueError()

            print(f"collect: info keys = {info.keys()}")  # 调试信息
            god_view = info.get('god_view_info', None) if 'god_view_info' in info else {}
            print(f"god_view in tscollector before processing type: {type(god_view)}, shape: {god_view.shape if hasattr(god_view, 'shape') else 'None'}")  # 调试信息
            if god_view.size == 0:
                god_view = np.zeros(7, dtype=np.float32)
            print(f"god_view in tscollector after processing type: {type(god_view)}, shape: {god_view.shape if hasattr(god_view, 'shape') else 'None'}")  # 调试信息

            self.data.update(
                obs_next=obs_next,
                rew=rew,
                terminated=terminated,
                truncated=truncated,
                done=done,
                info=info,
                god_view=god_view,
            )
            if self.preprocess_fn:
                self.data.update(
                    self.preprocess_fn(
                        obs_next=self.data.obs_next,
                        rew=self.data.rew,
                        done=self.data.done,
                        info=self.data.info,
                        policy=self.data.policy,
                        env_id=ready_env_ids,
                    )
                )

            if render:
                self.env.render()
                if render > 0 and not np.isclose(render, 0):
                    time.sleep(render)

            # 将数据添加到缓冲区
            ptr, ep_rew, ep_len, ep_idx = self.buffer.add(
                self.data, buffer_ids=ready_env_ids
            )

            # 收集统计数据
            step_count += len(ready_env_ids)

            if np.any(done):
                env_ind_local = np.where(done)[0]
                env_ind_global = ready_env_ids[env_ind_local]
                episode_count += len(env_ind_local)
                episode_lens.append(ep_len[env_ind_local])
                episode_rews.append(ep_rew[env_ind_local])
                episode_start_indices.append(ep_idx[env_ind_local])
                # 现在我们复制 obs_next 到 obs，但由于可能有
                # 完成的回合，我们首先需要重置完成的环境。
                self._reset_env_with_ids(
                    env_ind_local, env_ind_global, gym_reset_kwargs
                )
                for i in env_ind_local:
                    self._reset_state(i)

                # 从 ready_env_ids 中移除多余的 env id
                # 以避免在选择环境时存在偏差
                if n_episode:
                    surplus_env_num = len(ready_env_ids) - (n_episode - episode_count)
                    if surplus_env_num > 0:
                        mask = np.ones_like(ready_env_ids, dtype=bool)
                        mask[env_ind_local[:surplus_env_num]] = False
                        ready_env_ids = ready_env_ids[mask]
                        self.data = self.data[mask]

            self.data.obs = self.data.obs_next

            if (n_step and step_count >= n_step) or \
                    (n_episode and episode_count >= n_episode):
                break

        # 生成统计数据
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)

        if n_episode:
            self.data = Batch(
                obs={},
                act={},
                rew={},
                terminated={},
                truncated={},
                done={},
                obs_next={},
                info={},
                policy={},
                god_view={},
            )
            self.reset_env()

        if episode_count > 0:
            rews, lens, idxs = list(
                map(
                    np.concatenate,
                    [episode_rews, episode_lens, episode_start_indices]
                )
            )
            rew_mean, rew_std = rews.mean(), rews.std()
            len_mean, len_std = lens.mean(), lens.std()
        else:
            rews, lens, idxs = np.array([]), np.array([], int), np.array([], int)
            rew_mean = rew_std = len_mean = len_std = 0

        return {
            "n/ep": episode_count,
            "n/st": step_count,
            "rews": rews,
            "lens": lens,
            "idxs": idxs,
            "rew": rew_mean,
            "len": len_mean,
            "rew_std": rew_std,
            "len_std": len_std,
        }
