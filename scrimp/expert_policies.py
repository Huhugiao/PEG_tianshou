import numpy as np
import math
import map_config

def _clip_pair(angle_delta_deg: float, speed_factor: float):
    """
    将角度增量和速度因子限制在有效范围内
    角度范围: [-map_config.max_turn_deg, map_config.max_turn_deg]
    速度因子范围: [0, 1]
    """
    limit = float(getattr(map_config, 'max_turn_deg', 10.0))
    angle = float(np.clip(angle_delta_deg, -limit, limit))
    speed = float(np.clip(speed_factor, 0.0, 1.0))
    return angle, speed

def _wrap_deg(delta):
    """
    将角度差值包装到 [-180, 180] 范围内
    Args:
        delta: 角度差值（度）
    Returns:
        包装后的角度差值
    """
    while delta > 180.0:
        delta -= 360.0
    while delta < -180.0:
        delta += 360.0
    return delta

def get_expert_tracker_action_pair(observation):
    """
    Tracker专家策略：基于距离判断的策略
    - 计算beta = D线速度/A线速度
    - 若d < beta*R，D的方向角为垂直DA并偏向T
    - 否则直接冲向A
    """
    try:
        v_d_to_t = np.array([float(observation[6]), float(observation[7])], dtype=np.float64)  # D->T (tracker到base)
        v_d_to_a = np.array([float(observation[4]), float(observation[5])], dtype=np.float64)  # D->A (tracker到target)
        v_at = np.array([float(observation[8]), float(observation[9])], dtype=np.float64)      # A->T (target到base)
    except Exception:
        v_d_to_t = np.array([0.0, 0.0], dtype=np.float64)
        v_d_to_a = np.array([0.0, 0.0], dtype=np.float64)
        v_at = np.array([0.0, 0.0], dtype=np.float64)

    curr_D_deg = float(observation[10] * 360.0) if len(observation) > 10 else 0.0
    eps = 1e-8
    
    d = float(np.linalg.norm(v_d_to_t))  # d: tracker到base的距离
    r = float(np.linalg.norm(v_d_to_a))  # r: tracker到target的距离
    R = float(np.linalg.norm(v_at))      # R: target到base的距离

    # 获取速度参数计算beta
    try:
        vD = float(getattr(map_config, 'tracker_speed'))
        vA = float(max(getattr(map_config, 'target_speed'), 1e-6))
        beta = vD / vA
    except:
        beta = 1.0  # 默认值

    if r < eps:
        return _clip_pair(0.0, 1.0)

    # 距离判断：如果d < beta*R，使用垂直DA偏向T策略；否则直接冲向A
    if d < beta * R:
        # D的方向角为垂直DA并偏向T
        u_da = v_d_to_a / r  # DA方向单位向量
        
        # 计算垂直于DA的两个方向
        u_perp1 = np.array([-u_da[1], u_da[0]], dtype=np.float64)  # 逆时针90度
        u_perp2 = -u_perp1  # 顺时针90度
        
        # 选择偏向T的垂直方向
        if R > eps:
            u_at = v_at / R  # AT方向单位向量
            # 选择与AT方向夹角更小的垂直方向
            if float(np.dot(u_perp1, u_at)) >= float(np.dot(u_perp2, u_at)):
                desired_dir = u_perp1
            else:
                desired_dir = u_perp2
        else:
            # 如果target就在base处，随便选一个垂直方向
            desired_dir = u_perp1
        
        desired_angle_deg = float(np.degrees(np.arctan2(desired_dir[1], desired_dir[0])))
        
    else:
        # 直接冲向A
        u_da = v_d_to_a / r
        desired_angle_deg = float(np.degrees(np.arctan2(u_da[1], u_da[0])))

    # 计算角度差并裁剪
    rel_delta = _wrap_deg(desired_angle_deg - curr_D_deg)
    return _clip_pair(rel_delta, 1.0)

def get_expert_target_action_pair(observation):
    """
    专家策略(Target): 基于距离判断的策略
    - 计算beta = D线速度/A线速度
    - 若d < beta*R，使用原有的合成方向策略
    - 否则直接奔向基地T
    返回: (relative_angle_deg in [-45,45], speed_factor in [0,1])
    observation 索引约定:
      [4], [5]: tracker->target (d->a)
      [6], [7]: tracker->base (d->t)
      [8], [9]: target->base (a->t)
      [11]: target朝向/360
    """
    eps = 1e-8
    v_d_to_a = np.array([float(observation[4]), float(observation[5])], dtype=np.float64)  # tracker->target
    v_d_to_t = np.array([float(observation[6]), float(observation[7])], dtype=np.float64)  # tracker->base
    v_at = np.array([float(observation[8]), float(observation[9])], dtype=np.float64)      # target->base
    v_ad = -v_d_to_a  # target->tracker

    d = float(np.linalg.norm(v_d_to_t))  # d: tracker到base的距离
    r = float(np.linalg.norm(v_ad))      # r: target到tracker的距离
    R = float(np.linalg.norm(v_at))      # R: target到base的距离

    current_angle_deg = float(observation[11]) * 360.0

    # 获取速度参数计算beta
    try:
        vD = float(getattr(map_config, 'tracker_speed'))
        vA = float(max(getattr(map_config, 'target_speed'), 1e-6))
        beta = vD / vA
    except:
        beta = 1.0  # 默认值

    # 距离判断：如果d < beta*R，使用原策略；否则直接奔向基地
    if d < beta * R:
        # 使用原有的合成方向策略
        if r < eps and R < eps:
            return _clip_pair(0.0, 1.0)

        u_ad = v_ad / (r + eps)
        u_at = v_at / (R + eps)

        # 几何量（与原实现一致）
        dot = float(np.clip(np.dot(u_ad, u_at), -1.0, 1.0))
        cross_z = float(u_ad[0] * u_at[1] - u_ad[1] * u_at[0])
        sin_alpha = float(abs(cross_z))
        d_len_sq = r * r + R * R - 2.0 * r * R * dot
        d_len = float(np.sqrt(max(d_len_sq, 0.0)))

        if d_len < eps:
            desired_dir = u_at
        else:
            # 原公式计算的"角参数"
            sin_theta = (R - r * dot) / d_len
            cos_theta = (r * sin_alpha) / d_len
            sin_theta = float(np.clip(sin_theta, -1.0, 1.0))
            cos_theta = float(np.clip(cos_theta, -1.0, 1.0))

            # 关键修正：以 u_at 为基向量合成，不再围绕 u_ad 旋转
            # 侧向方向取"相对 u_ad 的垂直方向，并选取与 u_at 更同向的一侧"
            u_perp1 = np.array([-u_ad[1], u_ad[0]], dtype=np.float64)
            u_perp2 = -u_perp1
            u_perp = u_perp1 if float(np.dot(u_perp1, u_at)) >= float(np.dot(u_perp2, u_at)) else u_perp2

            desired_dir = cos_theta * u_ad + sin_theta * u_perp

        # 归一化并转角
        norm_des = float(np.linalg.norm(desired_dir))
        if norm_des < eps:
            desired_dir = u_at
            norm_des = float(np.linalg.norm(desired_dir))
            if norm_des < eps:
                desired_angle_deg = current_angle_deg
            else:
                desired_dir /= norm_des
                desired_angle_deg = float(np.degrees(np.arctan2(desired_dir[1], desired_dir[0])))
        else:
            desired_dir /= norm_des
            desired_angle_deg = float(np.degrees(np.arctan2(desired_dir[1], desired_dir[0])))

    else:
        # 直接奔向基地T
        if R < eps:
            return _clip_pair(0.0, 1.0)
        
        u_at = v_at / R
        desired_angle_deg = float(np.degrees(np.arctan2(u_at[1], u_at[0])))

    # 计算角度差并裁剪
    angle_diff = desired_angle_deg - current_angle_deg
    angle_diff = _wrap_deg(angle_diff)

    return _clip_pair(angle_diff, 1.0)