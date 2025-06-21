import gym
import numpy as np
from gym import spaces
import task_config, algo_config

def solve_nash_gain(R1, R2, T):
    """
    根据二方 LQ Nash 解的闭式公式计算反馈增益（简化 2D 版）。
    R1, R2: (2×2) 控制权重矩阵；T: time-to-go 标量
    返回 K1, K2: (2×4) 反馈增益
    """
    # 状态维度 4: [dx, dy, dvx, dvy]
    # 系统矩阵
    A = np.block([[np.zeros((2,2)), np.eye(2)],
                  [np.zeros((2,2)), np.zeros((2,2))]])
    B = np.vstack([np.zeros((2,2)), np.eye(2)])
    # 合成参数
    gamma = np.linalg.inv(np.linalg.inv(R1) + np.linalg.inv(R2))
    # P 矩阵简化填写为块对角
    p11 = 3*gamma/(3*gamma + T**3)
    p14 = p11 * T
    p44 = p11 * T**2
    P = np.block([[p11*np.eye(2), p14*np.eye(2)],
                  [p14*np.eye(2), p44*np.eye(2)]])
    K1 = np.linalg.inv(R1) @ B.T @ P   # min 方
    K2 = np.linalg.inv(R2) @ B.T @ P   # max 方（应用时取相反号）
    return K1, K2

class NashTrackEnv(gym.Env):
    """
    使用 LQ Nash 反馈律的示例环境。
    状态: 两代理相对位置+速度 (4,)
    动作: None (由 Nash 增益自动生成)
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, T=5.0):
        # 控制权重，可从配置读取
        R1 = np.diag(task_config.R_tracker)  # 追踪者权重
        R2 = np.diag(task_config.R_target)   # 入侵者权重
        self.K1, self.K2 = solve_nash_gain(R1, R2, T)
        # 状态空间
        high = np.ones(4)*np.inf
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        # 动作由内置增益直接计算，无外部动作
        self.action_space = spaces.Discrete(1)
        self.state = None

    def reset(self, *, seed=None, options=None):
        # 初始化相对状态 [dx,dy,dvx,dvy]
        dx = np.random.uniform(-100,100)
        dy = np.random.uniform(-100,100)
        dvx = np.random.randn()*5
        dvy = np.random.randn()*5
        self.state = np.array([dx,dy,dvx,dvy], dtype=np.float32)
        return self.state, {}

    def step(self, action=None):
        y = self.state
        # 计算 Nash 控制
        u1 = - self.K1.dot(y)
        u2 = + self.K2.dot(y)
        # 简化动力学： x' = v; v' = u
        A = np.block([[np.zeros((2,2)), np.eye(2)],
                      [np.zeros((2,2)), np.zeros((2,2))]])
        B = np.vstack([np.zeros((2,2)), np.eye(2)])
        # 合并 u = u1 - u2 为零和场景
        u = u1 - u2
        dt = algo_config.dt
        # 状态更新
        self.state = self.state + dt*(A.dot(y) + B.dot(u))
        # 奖励: 距离惩罚
        dist = np.linalg.norm(self.state[:2])
        reward = -dist
        done = False
        info = {}
        return self.state, reward, done, False, info

    def render(self, mode='human'):
        # 可视化留空
        pass

    def close(self):
        pass