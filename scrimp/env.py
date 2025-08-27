import gym
from gym import spaces
import numpy as np
import pygame
import sys
import time
from gym import spaces
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import task_config, utils
from typing import Optional

class TrackingEnv(gym.Env):
    Metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 40
    }

    def __init__(self, mission=0, base_obs_dim=12, use_god_view=False, god_view_dim=4):
        """
        Initialize the tracking environment.
        """
        self.mission = mission
        self.base_obs_dim = base_obs_dim
        self.use_god_view = use_god_view
        self.god_view_dim = god_view_dim
        
        self.canvas = None
        self.tracker = None
        self.target = None
        self.base = None
        self.nav_point = None
        self.blocking_space = None
        self.mask_flag = task_config.mask_flag
        # 设置地图尺寸和像素大小
        self.width = task_config.width
        self.height = task_config.height
        self.pixel_size = task_config.pixel_size
        self.target_speed = task_config.target_speed
        self.tracker_speed = task_config.tracker_speed 
        # 初始化窗口和时钟（用于渲染和计时）
        self.window = None
        self.clock = None
        # 初始化智能体和目标的轨迹
        self.tracker_trajectory = []
        self.target_trajectory = []
        # 初始化step
        self.step_count = 0
        self.target_frame_count = 0
        # 前两步位置，用于特权信息
        self.prev_tracker_pos = None  # 前前步位置
        self.last_tracker_pos = None  # 前一步位置
        self.prev_target_pos = None
        self.last_target_pos = None

        # 新的观测空间定义
        obs_dim = self.base_obs_dim
        if self.use_god_view:
            obs_dim += self.god_view_dim
        self.observation_space = spaces.Box(
            low=-1, high=1, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # 定义动作空间
        self.action_space = spaces.Discrete(48)

        # 添加LSTM需要的时序缓存（在训练时处理，环境不维护）
        self.current_obs = None


    def _get_obs_features(self):
        """生成当前时刻的观测特征（基础+特权）"""
        # 位置归一化（相对于地图尺寸）
        tracker_norm = [
            self.tracker['x'] / self.width * 2 - 1, 
            self.tracker['y'] / self.height * 2 - 1
        ]
        target_norm = [
            self.target['x'] / self.width * 2 - 1,
            self.target['y'] / self.height * 2 - 1
        ]
        
        # 入侵者-智能体
        tracker_to_target = [
            (self.target['x'] - self.tracker['x']) / self.width,
            (self.target['y'] - self.tracker['y']) / self.height]
        # 智能体-基地
        tracker_to_base = [
            (self.base['x'] - self.tracker['x']) / self.width,
            (self.base['y'] - self.tracker['y']) / self.height]
        # 入侵者-基地
        target_to_base = [
            (self.base['x'] - self.target['x']) / self.width,
            (self.base['y'] - self.target['y']) / self.height]

        base_features = tracker_norm + target_norm + \
                        tracker_to_target + tracker_to_base + \
                        target_to_base
        
        # 加入角度信息
        tracker_angle = (self.tracker.get('theta', 0) / 360.0)
        target_angle = (self.target.get('theta', 0) / 360.0)
        base_features += [tracker_angle, target_angle]
        
        return np.array(base_features, dtype=np.float32)


    def step(self, tracker_action: int, target_action: int):
        """Handle actions based on mission type"""
        self.step_count += 1
        old_tracker = self.tracker.copy()
        old_target = self.target.copy()

        # Process actions based on mission type
        if self.mission == 0:  # Train tracker (target uses rules)
            # Tracker使用策略动作
            if tracker_action != -1:
                self.tracker = utils.agent_move(old_tracker, tracker_action, self.tracker_speed)
            else:
                # 规则追踪：直接朝target移动
                tracker_to_target = [self.target['x'] - self.tracker['x'], self.target['y'] - self.tracker['y']]
                distance = np.sqrt(tracker_to_target[0]**2 + tracker_to_target[1]**2)
                if distance > 1e-5:  # 避免除零
                    angle = np.arctan2(tracker_to_target[1], tracker_to_target[0])
                    self.tracker['x'] += self.tracker_speed * np.cos(angle)
                    self.tracker['y'] += self.tracker_speed * np.sin(angle)
                    # 边界约束
                    self.tracker['x'] = np.clip(self.tracker['x'], 0, self.width - self.pixel_size)
                    self.tracker['y'] = np.clip(self.tracker['y'], 0, self.height - self.pixel_size)
                    self.tracker['theta'] = np.degrees(angle) % 360
            
            # Target使用规则导航
            self.target, _, _, self.target_frame_count, target_angle = utils.target_nav(
                self.target, self.tracker, self.base,
                self.target_speed, self.target_frame_count
            )
            self.target['theta'] = target_angle
            
        elif self.mission == 1:  # Train target (tracker uses rules)
            # Tracker使用规则追踪
            if tracker_action == -1:
                tracker_to_target = [self.target['x'] - self.tracker['x'], self.target['y'] - self.tracker['y']]
                distance = np.sqrt(tracker_to_target[0]**2 + tracker_to_target[1]**2)
                if distance > 1e-5:  # 避免除零
                    angle = np.arctan2(tracker_to_target[1], tracker_to_target[0])
                    self.tracker['x'] += self.tracker_speed * np.cos(angle)
                    self.tracker['y'] += self.tracker_speed * np.sin(angle)
                    # 边界约束
                    self.tracker['x'] = np.clip(self.tracker['x'], 0, self.width - self.pixel_size)
                    self.tracker['y'] = np.clip(self.tracker['y'], 0, self.height - self.pixel_size)
                    self.tracker['theta'] = np.degrees(angle) % 360
            else:
                # Tracker使用策略动作
                self.tracker = utils.agent_move(old_tracker, tracker_action, self.tracker_speed)
            
            # Target处理
            if target_action != -1:
                # Target使用策略动作
                self.target = utils.agent_move(old_target, target_action, self.target_speed)
            else:
                # Target使用规则导航
                self.target, _, _, self.target_frame_count, target_angle = utils.target_nav(
                    self.target, self.tracker, self.base,
                    self.target_speed, self.target_frame_count
                )
                self.target['theta'] = target_angle
                
        elif self.mission == 2:  # Train tracker with target policy
            # Tracker处理
            if tracker_action == -1:
                # 规则追踪
                tracker_to_target = [self.target['x'] - self.tracker['x'], self.target['y'] - self.tracker['y']]
                distance = np.sqrt(tracker_to_target[0]**2 + tracker_to_target[1]**2)
                if distance > 1e-5:
                    angle = np.arctan2(tracker_to_target[1], tracker_to_target[0])
                    self.tracker['x'] += self.tracker_speed * np.cos(angle)
                    self.tracker['y'] += self.tracker_speed * np.sin(angle)
                    self.tracker['x'] = np.clip(self.tracker['x'], 0, self.width - self.pixel_size)
                    self.tracker['y'] = np.clip(self.tracker['y'], 0, self.height - self.pixel_size)
                    self.tracker['theta'] = np.degrees(angle) % 360
            else:
                # 策略追踪
                self.tracker = utils.agent_move(old_tracker, tracker_action, self.tracker_speed)
            
            # Target处理
            if target_action != -1:
                # Target使用策略动作
                self.target = utils.agent_move(old_target, target_action, self.target_speed)
            else:
                # Target使用规则导航
                self.target, _, _, self.target_frame_count, target_angle = utils.target_nav(
                    self.target, self.tracker, self.base,
                    self.target_speed, self.target_frame_count
                )
                self.target['theta'] = target_angle
                
        elif self.mission == 3:  # Expert rule mode
            # Tracker处理
            if tracker_action == -1:
                # 规则追踪
                tracker_to_target = [self.target['x'] - self.tracker['x'], self.target['y'] - self.tracker['y']]
                distance = np.sqrt(tracker_to_target[0]**2 + tracker_to_target[1]**2)
                if distance > 1e-5:
                    angle = np.arctan2(tracker_to_target[1], tracker_to_target[0])
                    self.tracker['x'] += self.tracker_speed * np.cos(angle)
                    self.tracker['y'] += self.tracker_speed * np.sin(angle)
                    self.tracker['x'] = np.clip(self.tracker['x'], 0, self.width - self.pixel_size)
                    self.tracker['y'] = np.clip(self.tracker['y'], 0, self.height - self.pixel_size)
                    self.tracker['theta'] = np.degrees(angle) % 360
            else:
                # 策略追踪
                self.tracker = utils.agent_move(old_tracker, tracker_action, self.tracker_speed)
            
            # Target处理 - 专家规则或策略
            if target_action != -1:
                # Target使用专家规则动作（0-47）或策略动作
                self.target = utils.agent_move(old_target, target_action, self.target_speed)
            else:
                # Target使用基础规则导航
                self.target, _, _, self.target_frame_count, target_angle = utils.target_nav(
                    self.target, self.tracker, self.base,
                    self.target_speed, self.target_frame_count
                )
                self.target['theta'] = target_angle
                
        elif self.mission == -1:  # 自定义mission：两个都是规则
            # Tracker使用规则追踪
            if tracker_action == -1:
                tracker_to_target = [self.target['x'] - self.tracker['x'], self.target['y'] - self.tracker['y']]
                distance = np.sqrt(tracker_to_target[0]**2 + tracker_to_target[1]**2)
                if distance > 1e-5:
                    angle = np.arctan2(tracker_to_target[1], tracker_to_target[0])
                    self.tracker['x'] += self.tracker_speed * np.cos(angle)
                    self.tracker['y'] += self.tracker_speed * np.sin(angle)
                    self.tracker['x'] = np.clip(self.tracker['x'], 0, self.width - self.pixel_size)
                    self.tracker['y'] = np.clip(self.tracker['y'], 0, self.height - self.pixel_size)
                    self.tracker['theta'] = np.degrees(angle) % 360
            else:
                self.tracker = utils.agent_move(old_tracker, tracker_action, self.tracker_speed)
                
            # Target使用规则导航
            if target_action == -1:
                self.target, _, _, self.target_frame_count, target_angle = utils.target_nav(
                    self.target, self.tracker, self.base,
                    self.target_speed, self.target_frame_count
                )
                self.target['theta'] = target_angle
            else:
                # 如果是专家规则返回的动作（0-47）
                self.target = utils.agent_move(old_target, target_action, self.target_speed)
                
        else:
            # 默认：都使用策略
            self.tracker = utils.agent_move(old_tracker, tracker_action, self.tracker_speed)
            self.target = utils.agent_move(old_target, target_action, self.target_speed)

        # 记录轨迹
        self.tracker_trajectory.append((
            self.tracker['x'] + self.pixel_size//2,
            self.tracker['y'] + self.pixel_size//2
        ))
        self.target_trajectory.append((
            self.target['x'] + self.pixel_size//2,
            self.target['y'] + self.pixel_size//2
        ))

        # 更新位置历史
        self.prev_tracker_pos = self.last_tracker_pos.copy()
        self.last_tracker_pos = self.tracker.copy()
        self.prev_target_pos = self.last_target_pos.copy()
        self.last_target_pos = self.target.copy()

        # 计算奖励
        reward, terminated, truncated, info = utils.reward_calculate(
            self.tracker, self.target, self.base, mission=self.mission)

        # 更新观测
        self.current_obs = self._get_obs_features()
        self.canvas = utils.get_canvas(
            self.target, self.tracker, self.base,
            self.tracker_trajectory, self.target_trajectory
        )

        return self.current_obs, reward, terminated, truncated, info
    
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None, ):
        # 地图设置
        self.tracker_trajectory = []
        self.target_trajectory = []
        self.base = {'x':250, 'y':250}
        self.tracker = {
        'x': np.random.randint(240, 260), 
        'y': np.random.randint(240, 260),
        'theta': 0  # 初始化角度
        }

        boundary = np.random.randint(0, 4) #随机选择一个边界
        self.target = {
            'x': np.random.randint(0, self.width) if boundary < 2 else (boundary - 2) * (self.width - 1),
            'y': boundary * (self.height - 1) if boundary < 2 else np.random.randint(0, self.height)}

        # 记录初始位置
        self.tracker_trajectory.append(
            (self.tracker['x'] + self.pixel_size // 2, self.tracker['y'] + self.pixel_size // 2))
        self.target_trajectory.append(
            (self.target['x'] + self.pixel_size // 2, self.target['y'] + self.pixel_size // 2))

        self.prev_tracker_pos = self.tracker.copy()
        self.last_tracker_pos = self.tracker.copy()
        self.prev_target_pos = self.target.copy()
        self.last_target_pos = self.target.copy()

        # 生成初始观测
        self.current_obs = self._get_obs_features()
        # 更新渲染画面
        self.canvas = utils.get_canvas(self.target, self.tracker, self.base,
                                       self.tracker_trajectory, self.target_trajectory)

        info = {}
        
        return self.current_obs, info

    def render(self, mode='human'):
        # 如果是首次渲染并且模式为'human'，则初始化pygame窗口和时钟
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.width, self.height))  # 初始化一张画布
            pygame.display.set_caption("Protecting base from invader")

        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()
        # 获取当前环境状态的渲染画面
        self.canvas = utils.get_canvas(self.target, self.tracker, self.base,
                                       self.tracker_trajectory, self.target_trajectory)
        # 根据渲染模式进行显示或存储
        if mode == "human":
            # 将绘制的canvas内容复制到窗口中
            self.window.blit(self.canvas, self.canvas.get_rect())
            pygame.event.pump()  # 处理事件队列
            pygame.display.update()  # 更新显示

            # 控制渲染的帧率
            self.clock.tick(self.Metadata["render_fps"])
        else:  # 如果模式不是'human'，则返回RGB数组
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2))

    def close(self):
        """
        关闭仿真环境并释放相关资源。

        无返回值
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

if __name__ == '__main__':
    env = gym.make('Protecting-v0')
    total_steps = 300

    for episode in range(1):  # 设置测试回合数
        # 重置环境
        observation = env.reset()
        env.render()
        terminated, truncated = False, False
        # 初始化累计奖励
        total_reward = 0

        for step in range(total_steps):
            action = env.action_space.sample()  # 这里仅作为示例，使用随机动作
            env.render()
            time.sleep(0.1)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                env.render()
                time.sleep(1)
                print(
                    f"Episode {episode + 1} finished after {step + 1} steps with total reward {total_reward} because of {info}")
                break

    env.close()
    sys.exit()


