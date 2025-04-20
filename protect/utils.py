import numpy as np
import pygame
import math
import algo_config
import task_config
import os, torch
from tianshou.data import Batch

def reward_calculate(tracker, target, base):
    """
    Calculate zero-sum rewards with normalized distance penalties.
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
        reward = -20  # Tracker penalty
        terminated = True
        info['reason'] = 'Attacker reached the base'
    elif current_target_distance <= task_config.pixel_size:  # Tracker catches target
        reward = 20  # Tracker reward
        terminated = True
        info['reason'] = 'Defender intercepted the attacker'
    else:
        # Distance-based components (normalized 0-1)
        track_reward = 0.7 * (1 - norm_target)  # Closer to target -> higher reward
        base_penalty = 0.7 * (1 - norm_base)    # Closer to base -> higher penalty
        reward = track_reward - base_penalty
        reward += 0.3
    if algo_config.mission == 1:
        reward = -reward
    
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
    # 动作对应的角度变化（24个方向，每个方向间隔15度）
    angle_changes = np.array([i * 15 for i in range(24)])  # 0度到345度，间隔15度
    alpha = angle_changes[action]
    new_angle = alpha % 360  # 更新智能体的全局朝向角度
    
    # 根据角度和距离计算新位置
    d = moving_size
    new_pos = {
        'x': agent['x'] + d * np.cos(np.deg2rad(new_angle)),
        'y': agent['y'] + d * np.sin(np.deg2rad(new_angle))
    }
    
    # 添加边界约束
    new_pos['x'] = np.clip(new_pos['x'], 0, task_config.width - task_config.pixel_size)
    new_pos['y'] = np.clip(new_pos['y'], 0, task_config.height - task_config.pixel_size)
    return new_pos



def target_nav(target, tracker, base, moving_size, frame_count):
    """
    改进的连续逃逸导航策略，基于距离动态调整躲避强度，调整横向偏移频率并保持恒定速度。

    参数:
    - target: 当前目标的位置，字典 {'x': float, 'y': float}
    - tracker: 当前追踪者的位置，字典 {'x': float, 'y': float}
    - base: 基地位置，字典 {'x': float, 'y': float}
    - moving_size: 移动步长
    - frame_count: 帧计数器（用于周期机动）

    返回:
    - new_target_pos: 新位置字典
    - distance_to_tracker: 与追踪者的距离
    - distance_to_base: 与基地的距离
    - frame_count: 更新后的帧计数器
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
        # 增加切换间隔（每31帧切换方向）
        lateral_dir *= np.sign(math.sin(frame_count * 0.1))  # 0.1对应约62.8帧周期
    else:
        lateral_dir = np.zeros(2)

    # 方向合成：基础方向 + 动态横向躲避
    move_dir = (base_dir + lateral_dir * distance_scale * 2.5)
    
    # 当距离非常近时增强躲避
    if distance_to_tracker < 50:
        move_dir += lateral_dir * 1.5
    
    # 方向归一化和恒定速度
    norm = np.linalg.norm(move_dir)
    if norm > 1e-5:
        move_dir = move_dir / norm
    else:
        move_dir = base_dir  # 退化为向基地移动

    # 应用移动（保持恒定速度）
    new_pos = target.copy()
    new_pos['x'] += move_dir[0] * moving_size  # 移除动态速度调整
    new_pos['y'] += move_dir[1] * moving_size

    # 边界约束
    new_pos['x'] = np.clip(new_pos['x'], 0, task_config.width - task_config.pixel_size)
    new_pos['y'] = np.clip(new_pos['y'], 0, task_config.height - task_config.pixel_size)

    return new_pos, distance_to_tracker, distance_to_base, frame_count + 1

