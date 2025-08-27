import numpy as np
import gym
from scipy.linalg import solve_continuous_are
from .. import task_config, algo_config, utils
from gym.envs.user.protecting import TrackingEnv

class NashTrackEnv(TrackingEnv):
    """
    在 TrackingEnv 基础上增加两个智能体的连续时间 LQ 差分博弈 Nash 解。
    """

    def __init__(self):
        super().__init__()
        # 构造线性化相对动力学：dx/dt = A x + B1 u1 + B2 u2 (4维状态，2维控制)
        self.A = np.array([[0,0,1,0],
                           [0,0,0,1],
                           [0,0,0,0],
                           [0,0,0,0]], dtype=np.float64)
        self.B1 = np.array([[0,0],
                            [0,0],
                            [1,0],
                            [0,1]], dtype=np.float64)
        self.B2 = self.B1.copy()

        # 二次型代价权重
        self.Q1 = np.eye(4) * 1.0;   self.R1 = np.eye(2) * 0.1
        self.Q2 = np.eye(4) * 1.0;   self.R2 = np.eye(2) * 0.1

        # 静态 ARE 求解得到 Riccati 矩阵
        P1 = solve_continuous_are(self.A, self.B1, self.Q1, self.R1)
        P2 = solve_continuous_are(self.A, self.B2, self.Q2, self.R2)
        # 生成 Nash 反馈增益
        self.K1 = np.linalg.inv(self.R1) @ self.B1.T @ P1
        self.K2 = np.linalg.inv(self.R2) @ self.B2.T @ P2

    def step(self, action):
        # 调用父类 step 更新 tracker、target
        _, _, terminated, truncated, _ = super().step(action)

        # 计算相对状态 x_rel = [dx, dy, dvx, dvy]
        dx  = self.target['x'] - self.tracker['x']
        dy  = self.target['y'] - self.tracker['y']
        dvx = self.target.get('vx',0) - self.tracker.get('vx',0)
        dvy = self.target.get('vy',0) - self.tracker.get('vy',0)
        x_rel = np.array([dx, dy, dvx, dvy], dtype=np.float64)

        # Nash 均衡反馈律
        u1 = - self.K1 @ x_rel
        u2 = - self.K2 @ x_rel

        # 映射到离散动作
        tracker_action = self._cont_to_disc(u1)
        target_action  = self._cont_to_disc(u2)

        # 覆盖原有动作执行
        self.tracker = utils.agent_move(self.tracker.copy(),
                                        tracker_action,
                                        self.tracker_speed)
        if algo_config.mission != 0:
            self.target = utils.agent_move(self.target.copy(),
                                           target_action,
                                           self.target_speed)

        # 更新轨迹
        self.tracker_trajectory.append((self.tracker['x'], self.tracker['y']))
        self.target_trajectory.append((self.target['x'],  self.target['y']))

        # 重新计算奖励
        rew, terminated, truncated, info = utils.reward_calculate(
            self.tracker, self.target, self.base)
        # 将 Nash 控制量放入 info
        info['nash_u1'] = u1.copy()
        info['nash_u2'] = u2.copy()

        # 更新观测与画布
        new_obs = self._get_obs_features()
        self.canvas = utils.get_canvas(self.target,
                                       self.tracker,
                                       self.base,
                                       self.tracker_trajectory,
                                       self.target_trajectory)

        return new_obs, rew, terminated, truncated, info

    def _cont_to_disc(self, u):
        """
        将连续加速度向量 quantize 到 8 方向动作(0..7)
        """
        angle = np.arctan2(u[1], u[0])
        idx   = int(np.round((angle / (2*np.pi)) * 8)) % 8
        return idx