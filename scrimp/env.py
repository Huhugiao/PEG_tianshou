import os
import sys
import math
import numpy as np
import gym
from typing import Optional, Union, Tuple
from gym import spaces

import env_lib, map_config
from alg_parameters import EnvParameters


class TrackingEnv(gym.Env):
    Metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 40
    }

    def __init__(self, mission=0, base_obs_dim=12, use_god_view=False, god_view_dim=4):
        super().__init__()
        self.mission = mission
        self.base_obs_dim = base_obs_dim
        self.use_god_view = use_god_view
        self.god_view_dim = god_view_dim

        # 配置
        self.mask_flag = getattr(map_config, 'mask_flag', False)
        self.width = map_config.width
        self.height = map_config.height
        self.pixel_size = map_config.pixel_size
        self.target_speed = map_config.target_speed
        self.tracker_speed = map_config.tracker_speed

        # 状态
        self.canvas = None
        self.tracker = None
        self.target = None
        self.base = None

        # 窗口
        self.window = None
        self.clock = None

        # 轨迹
        self.tracker_trajectory = []
        self.target_trajectory = []

        # 计数
        self.step_count = 0
        self.target_frame_count = 0

        # 历史
        self.prev_tracker_pos = None
        self.last_tracker_pos = None
        self.prev_target_pos = None
        self.last_target_pos = None

        # 观测空间
        obs_dim = self.base_obs_dim + (self.god_view_dim if self.use_god_view else 0)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        # 动作空间
        self.action_space = spaces.Discrete(48)

        self.current_obs = None

    def _get_obs_features(self):
        tracker_norm = [
            self.tracker['x'] / self.width * 2 - 1,
            self.tracker['y'] / self.height * 2 - 1,
        ]
        target_norm = [
            self.target['x'] / self.width * 2 - 1,
            self.target['y'] / self.height * 2 - 1,
        ]
        tracker_to_target = [
            (self.target['x'] - self.tracker['x']) / self.width,
            (self.target['y'] - self.tracker['y']) / self.height,
        ]
        tracker_to_base = [
            (self.base['x'] - self.tracker['x']) / self.width,
            (self.base['y'] - self.tracker['y']) / self.height,
        ]
        target_to_base = [
            (self.base['x'] - self.target['x']) / self.width,
            (self.base['y'] - self.target['y']) / self.height,
        ]
        feats = tracker_norm + target_norm + tracker_to_target + tracker_to_base + target_to_base
        feats += [float(self.tracker.get('theta', 0.0) / 360.0), float(self.target.get('theta', 0.0) / 360.0)]
        return np.array(feats, dtype=np.float32)

    def _parse_actions(self, action: Union[Tuple, list, np.ndarray, None], target_action: Optional[Tuple] = None):
        """
        支持以下输入：
          - action 为 ((angle_deg, speed_factor), (angle_deg, speed_factor))
          - action 为 (angle_deg, speed_factor) 且 target_action 单独提供或为 None
          - -1 表示该智能体使用rule
        返回: (tracker_action, target_action)，其中每个为 -1 或 (angle_deg, speed_factor)
        """
        def _is_pair(a):
            return isinstance(a, (tuple, list, np.ndarray)) and len(a) == 2 and np.isscalar(a[0]) and np.isscalar(a[1])

        if target_action is not None:
            ta = action
            tg = target_action
            if ta == -1 or tg == -1:
                return ta, tg
            if _is_pair(ta) or ta == -1:
                pass
            else:
                raise ValueError("tracker action must be (angle_deg, speed_factor) or -1")
            if _is_pair(tg) or tg == -1:
                pass
            else:
                raise ValueError("target action must be (angle_deg, speed_factor) or -1")
            return ta, tg

        if action is None:
            return -1, -1

        # action 为二元组：(tracker_act, target_act)
        if isinstance(action, (tuple, list, np.ndarray)) and len(action) == 2 and (isinstance(action[0], (tuple, list, np.ndarray)) or action[0] == -1 or isinstance(action[1], (tuple, list, np.ndarray)) or action[1] == -1):
            ta, tg = action[0], action[1]
            if ta != -1 and not _is_pair(ta):
                raise ValueError("tracker action must be (angle_deg, speed_factor) or -1")
            if tg != -1 and not _is_pair(tg):
                raise ValueError("target action must be (angle_deg, speed_factor) or -1")
            return ta, tg

        # 单个tracker动作对
        if _is_pair(action) or action == -1:
            return action, -1

        raise ValueError("Unsupported action format for TrackingEnv.step")


    def step(self, action: Union[Tuple, list, np.ndarray] = None, target_action: Optional[Tuple] = None):
        self.step_count += 1

        tracker_action, target_action = self._parse_actions(action, target_action)

        # Move tracker
        if tracker_action == -1:
            self.tracker = env_lib.tracker_nav(self.tracker, self.target, self.tracker_speed)
        else:
            self.tracker = env_lib.agent_move(self.tracker, tracker_action, self.tracker_speed)

        # Move target
        if target_action == -1:
            self.target = env_lib.target_nav(self.target, self.tracker, self.base, self.target_speed, self.target_frame_count)
        else:
            self.target = env_lib.agent_move(self.target, target_action, self.target_speed)

        # 记录轨迹
        self.tracker_trajectory.append((self.tracker['x'] + self.pixel_size / 2.0,
                                        self.tracker['y'] + self.pixel_size / 2.0))
        self.target_trajectory.append((self.target['x'] + self.pixel_size / 2.0,
                                       self.target['y'] + self.pixel_size / 2.0))
        max_len = getattr(map_config, 'trail_max_len', 600)
        if len(self.tracker_trajectory) > max_len:
            self.tracker_trajectory = self.tracker_trajectory[-max_len:]
        if len(self.target_trajectory) > max_len:
            self.target_trajectory = self.target_trajectory[-max_len:]

        # 历史位置
        if self.last_tracker_pos is not None:
            self.prev_tracker_pos = self.last_tracker_pos.copy()
        self.last_tracker_pos = self.tracker.copy()
        if self.last_target_pos is not None:
            self.prev_target_pos = self.last_target_pos.copy()
        self.last_target_pos = self.target.copy()

        reward, terminated, truncated, info = env_lib.reward_calculate(self.tracker, self.target, self.base, mission=self.mission)
        self.current_obs = self._get_obs_features()

        # 长度截断
        if self.step_count >= EnvParameters.EPISODE_LEN:
            truncated = True

        return self.current_obs, float(reward), bool(terminated), bool(truncated), info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.step_count = 0
        self.target_frame_count = 0
        self.tracker_trajectory = []
        self.target_trajectory = []

        self.base = {
            'x': float(self.width / 2.0 - self.pixel_size / 2.0),
            'y': float(self.height / 2.0 - self.pixel_size / 2.0)
        }
        self.tracker = {
            'x': float(np.random.randint(int(self.width * 0.48), int(self.width * 0.52))),
            'y': float(np.random.randint(int(self.height * 0.48), int(self.height * 0.52))),
            'theta': 0.0
        }
        # 目标随机边界
        boundary = int(np.random.randint(0, 4))
        margin = 5
        if boundary == 0:  # top
            x = float(np.random.randint(0, self.width - self.pixel_size))
            y = float(margin)
        elif boundary == 1:  # bottom
            x = float(np.random.randint(0, self.width - self.pixel_size))
            y = float(self.height - self.pixel_size - margin)
        elif boundary == 2:  # left
            x = float(margin)
            y = float(np.random.randint(0, self.height - self.pixel_size))
        else:  # right
            x = float(self.width - self.pixel_size - margin)
            y = float(np.random.randint(0, self.height - self.pixel_size))
        self.target = {'x': x, 'y': y, 'theta': 180.0}

        self.tracker_trajectory.append((self.tracker['x'] + self.pixel_size / 2.0,
                                        self.tracker['y'] + self.pixel_size / 2.0))
        self.target_trajectory.append((self.target['x'] + self.pixel_size / 2.0,
                                       self.target['y'] + self.pixel_size / 2.0))

        self.prev_tracker_pos = self.tracker.copy()
        self.last_tracker_pos = self.tracker.copy()
        self.prev_target_pos = self.target.copy()
        self.last_target_pos = self.target.copy()

        self.current_obs = self._get_obs_features()
        return self.current_obs, {}

    def render(self, mode='human'):
        canvas = env_lib.get_canvas(self.target, self.tracker, self.base,
                                  self.tracker_trajectory, self.target_trajectory)
        if mode == 'rgb_array':
            return canvas
        elif mode == 'human':
            try:
                import pygame
                if self.window is None:
                    pygame.init()
                    self.window = pygame.display.set_mode((self.width, self.height))
                    self.clock = pygame.time.Clock()
                s = pygame.surfarray.make_surface(np.transpose(canvas, (1, 0, 2)))
                self.window.blit(s, (0, 0))
                import pygame as pg
                pg.display.flip()
                self.clock.tick(self.Metadata['render_fps'])
            except Exception:
                pass

    def close(self):
        try:
            import pygame
            pygame.quit()
        except Exception:
            pass


if __name__ == '__main__':
    os.environ.setdefault('SCRIMP_RENDER_MODE', 'quality')
    env = TrackingEnv(mission=0)
    obs, _ = env.reset()
    rng = np.random.RandomState(0)
    for _ in range(300):
        # 随机动作对：(角度[-45,45]，速度因子[0,1])
        rand_pair = (float(rng.uniform(-45, 45)), float(rng.uniform(0, 1)))
        obs, r, ter, tru, info = env.step((rand_pair, -1))
        if ter or tru:
            env.reset()
    env.close()
    sys.exit(0)