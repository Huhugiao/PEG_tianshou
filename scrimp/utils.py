import numpy as np
import pygame
import math
import task_config

def reward_calculate(tracker, target, base, mission=0):
    """
    Calculate zero-sum rewards with normalized distance penalties.
    
    Args:
        tracker: Tracker agent position dictionary
        target: Target agent position dictionary
        base: Base position dictionary
        mission: Mission type (0: train tracker, 1: train target, 2: train tracker with target policy)
    """
    reward = 0
    terminated = False
    truncated = False
    info = {}

    # Calculate raw distances
    current_target_distance = np.sqrt((tracker['x'] - target['x'])**2 + (tracker['y'] - target['y'])**2)
    current_base_distance = np.sqrt((target['x'] - base['x'])**2 + (target['y'] - base['y'])**2)

    # Normalize distances using map dimensions
    max_distance = np.sqrt(task_config.width**2 + task_config.height**2)
    norm_target = current_target_distance / max_distance  # 0-1 (far-near)
    norm_base = current_base_distance / max_distance  # 0-1 (far-near)

    # Termination conditions
    if current_base_distance <= task_config.pixel_size:  # Target reaches base
        reward = -50  # Tracker penalty
        terminated = True
        info['reason'] = 'Attacker reached the base'
    elif current_target_distance <= task_config.pixel_size:  # Tracker catches target
        reward = 50  # Tracker reward
        truncated = True
        info['reason'] = 'Defender intercepted the attacker'
    else:
        # Distance-based components (normalized 0-1)
        track_reward = 0.6 * (1 - norm_target)  # Closer to target -> higher reward
        base_penalty = 1 * (1 - norm_base)    # Closer to base -> higher penalty
        reward = track_reward - base_penalty
    
    if mission == 1:
        reward = -reward
        
    reward -= 0.2
    return reward, terminated, truncated, info

def get_canvas(target, tracker, base, tracker_trajectory, target_trajectory):
    # 加载并调整智能体和目标的图像尺寸
    tracker_img = task_config.tracker_img
    target_img = task_config.target_img
    base_img = task_config.base_img
    tracker_img = pygame.transform.scale(tracker_img, (20, 20))
    target_img = pygame.transform.scale(target_img, (20, 20))
    base_img = pygame.transform.scale(base_img, (25, 25))
    # 创建一个新的pygame surface作为背景
    canvas = pygame.Surface((task_config.width, task_config.height))  # 创建一个新的pygame surface作为背景
    canvas.fill((255, 255, 255))  # 设置背景为白色

    # 绘制轨迹
    for i in range(len(tracker_trajectory) - 1):
        pygame.draw.line(canvas, (0, 0, 255), tracker_trajectory[i], tracker_trajectory[i + 1], 1)
    for i in range(len(target_trajectory) - 1):
        pygame.draw.line(canvas, (255, 0, 0), target_trajectory[i], target_trajectory[i + 1], 1)

    # 在指定位置绘制智能体和目标的图像
    canvas.blit(tracker_img,
                (tracker['x'] - tracker_img.get_rect().bottom / 2 + task_config.pixel_size // 2,
                 tracker['y'] - tracker_img.get_rect().right / 2 + task_config.pixel_size // 2))
    canvas.blit(target_img,
                (target['x'] - target_img.get_rect().bottom / 2 + task_config.pixel_size // 2,
                 target['y'] - target_img.get_rect().right / 2 + task_config.pixel_size // 2))
    canvas.blit(base_img,
                (base['x'] - base_img.get_rect().bottom / 2 + task_config.pixel_size // 2,
                 base['y'] - base_img.get_rect().right / 2 + task_config.pixel_size // 2))
    return canvas


def agent_move(agent, action, moving_size):
    """
    根据组合动作更新智能体的位置。
    
    参数:
    - agent: 包含智能体位置和朝向信息的字典
    - action: 整数组合动作，取值范围0~47，其中：
              angle_index = action // 3 对应相对角度（单位：度），从 -45 到 45 离散取值
              speed_index = action % 3 对应速度选择：
                  0 -> 静止（0）
                  1 -> 半速（moving_size/2）
                  2 -> 全速（moving_size）
    - moving_size: 全速时的移动步长
    
    返回:
    - 更新后的智能体字典
    """
    # 解码动作: 重新调整为先选角度后选速度
    angle_index = action // 3
    speed_index = action % 3

    # 根据速度索引确定实际速度
    if speed_index == 0:
        speed = 0
    elif speed_index == 1:
        speed = moving_size / 2
    else:
        speed = moving_size

    # 离散角度列表：16个从 -45 到 45 度的值
    angle_offsets = np.linspace(-45, 45, 16)
    angle_offset = angle_offsets[angle_index]

    # 计算新的朝向（当前朝向加上角度偏移），确保结果在 0-360 度之间
    current_angle = agent.get('theta', 0)
    new_angle = (current_angle + angle_offset) % 360
    agent['theta'] = new_angle

    # 根据新的朝向和速度更新位置
    new_x = agent['x'] + speed * np.cos(np.deg2rad(new_angle))
    new_y = agent['y'] + speed * np.sin(np.deg2rad(new_angle))
    
    agent['x'] = np.clip(new_x, 0, task_config.width - task_config.pixel_size)
    agent['y'] = np.clip(new_y, 0, task_config.height - task_config.pixel_size)
    
    return agent

# def agent_move(agent, action, moving_size):
#     """
#     根据组合动作更新智能体的位置（全速前进）。
    
#     参数:
#     - agent: 包含智能体位置和朝向信息的字典
#     - action: 整数，取值范围0~35，对应-70到70度的角偏差（全速运动）
#     - moving_size: 全速移动时的步长
    
#     返回:
#     - 更新后的智能体字典
#     """
#     # 离散角度列表：36个从 -70 到 70 度的值
#     angle_offsets = np.linspace(-70, 70, 48)
#     angle_offset = angle_offsets[action]
    
#     # 计算新的朝向（当前朝向加上角偏差），确保结果在 0-360 度之间
#     current_angle = agent.get('theta', 0)
#     new_angle = (current_angle + angle_offset) % 360
#     agent['theta'] = new_angle
    
#     # 全速运动
#     new_x = agent['x'] + moving_size * np.cos(np.deg2rad(new_angle))
#     new_y = agent['y'] + moving_size * np.sin(np.deg2rad(new_angle))
    
#     agent['x'] = np.clip(new_x, 0, task_config.width - task_config.pixel_size)
#     agent['y'] = np.clip(new_y, 0, task_config.height - task_config.pixel_size)
    
#     return agent


def target_nav(target, tracker, base, moving_size, frame_count):
    """
    改进的连续逃逸导航策略，基于距离动态调整躲避强度，调整横向偏移频率并保持恒定速度。

    参数:
    - target: 当前目标的位置字典，包含{'x': float, 'y': float}，可选'theta'
    - tracker: 当前追踪者的位置字典
    - base: 基地位置字典
    - moving_size: 移动步长
    - frame_count: 帧计数器（用于周期机动）

    返回:
    - new_pos: 新位置字典（增加'theta'字段，表示新的朝向）
    - distance_to_tracker: 与追踪者的距离
    - distance_to_base: 与基地的距离
    - new_frame_count: 更新后的帧计数器
    - new_theta: 目标新的朝向（角度，单位：度）
    """
    # 计算相对向量和距离
    tracker_vec = np.array([tracker['x'] - target['x'], tracker['y'] - target['y']])
    base_vec = np.array([base['x'] - target['x'], base['y'] - target['y']])
    
    distance_to_tracker = np.linalg.norm(tracker_vec)
    distance_to_base = np.linalg.norm(base_vec)
    
    # 基础方向归一化
    if distance_to_base > 0:
        base_dir = base_vec / distance_to_base
    else:
        base_dir = np.zeros(2)

    # 动态躲避参数计算
    safe_radius = 150  # 最大影响范围
    min_scale = 0.2    # 最小躲避强度
    distance_scale = min(safe_radius / max(distance_to_tracker, 1e-5), 1)  # 距离影响系数
    
    # 生成横向偏移方向（始终垂直于追踪方向）
    if np.linalg.norm(tracker_vec) > 1e-5:
        tracker_dir = tracker_vec / np.linalg.norm(tracker_vec)
        lateral_dir = np.array([-tracker_dir[1], tracker_dir[0]])  # 逆时针垂直方向
        lateral_dir *= np.sign(math.sin(frame_count * 0.1))
    else:
        lateral_dir = np.zeros(2)

    # 方向合成：基础方向 + 动态横向躲避
    move_dir = (base_dir + lateral_dir * distance_scale * 2.5)
    
    if distance_to_tracker < 50:
        move_dir += lateral_dir * 1.5
    
    # 归一化方向并设置恒定速度
    norm = np.linalg.norm(move_dir)
    if norm > 1e-5:
        move_dir = move_dir / norm
    else:
        move_dir = base_dir

    new_pos = target.copy()
    new_pos['x'] += move_dir[0] * moving_size
    new_pos['y'] += move_dir[1] * moving_size

    # 边界约束
    new_pos['x'] = np.clip(new_pos['x'], 0, task_config.width - task_config.pixel_size)
    new_pos['y'] = np.clip(new_pos['y'], 0, task_config.height - task_config.pixel_size)

    # 根据移动方向计算新的朝向，转换为角度（确保在 0-360 范围内）
    if norm > 1e-5:
        new_theta = math.degrees(math.atan2(move_dir[1], move_dir[0])) % 360
    else:
        new_theta = target.get('theta', 0)

    new_pos['theta'] = new_theta

    return new_pos, distance_to_tracker, distance_to_base, frame_count + 1, new_theta

