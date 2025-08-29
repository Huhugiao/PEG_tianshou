import numpy as np
import math
import map_config

def _clip_pair(angle_delta_deg: float, speed_factor: float):
    angle = float(np.clip(angle_delta_deg, -45.0, 45.0))
    speed = float(np.clip(speed_factor, 0.0, 1.0))
    return angle, speed

def _wrap_deg(delta):
    while delta > 180.0:
        delta -= 360.0
    while delta < -180.0:
        delta += 360.0
    return delta

def get_expert_tracker_action_pair(observation):
    """
    Tracker专家策略（微分博弈解）：
    - 设 AD 为从A(目标)指向D(追踪者)的向量；
    - A速度与AD的夹角为 gamma；
    - D最大速/A最大速为 alpha；
    - 则 D速度与AD的夹角 theta = arcsin(alpha * sin(gamma))；
    - D朝向 = AD方向 + theta；
    - 返回相对转角(裁切[-45,45])与满速(1.0)。

    实现细节：
    - 使用已实现的 A 专家策略推断其下一步速度方向，从而估计 gamma；
    - 若向量退化或角度不可用，回退为“追向A”的策略。
    """
    # 观测取值
    try:
        v_d_to_a = np.array([float(observation[4]), float(observation[5])], dtype=np.float64)  # D->A 方向（已归一化）
    except Exception:
        v_d_to_a = np.array([0.0, 0.0], dtype=np.float64)

    # 当前朝向（绝对角度，度）
    curr_D_deg = float(observation[10] * 360.0) if len(observation) > 10 else 0.0
    curr_A_deg = float(observation[11] * 360.0) if len(observation) > 11 else 0.0

    # 速度比 alpha
    vD = float(getattr(map_config, 'tracker_speed', 6.0))
    vA = float(max(getattr(map_config, 'target_speed', 8.0), 1e-6))
    alpha = float(np.clip(vD / vA, 0.0, 10.0))  # 允许>1但后续会用sin值裁界

    # AD 方向（A->D）
    v_ad = -v_d_to_a
    norm_ad = float(np.linalg.norm(v_ad))
    if norm_ad < 1e-8:
        # 退化：无有效AD方向，回退为“朝向A”
        desired_deg = curr_D_deg  # 不转向
        rel_delta = 0.0
        return _clip_pair(rel_delta, 1.0)

    u_ad = v_ad / norm_ad
    ad_deg = float(np.degrees(np.arctan2(u_ad[1], u_ad[0])))

    # 用 A 专家策略估计 A 的下一步绝对朝向
    # A 专家返回 (relative_angle_deg, speed_factor)
    a_rel_deg, _ = get_expert_target_action_pair(observation)
    a_next_deg = curr_A_deg + float(a_rel_deg)
    a_next_deg = (a_next_deg + 360.0) % 360.0  # 归一到[0,360)

    # 计算 gamma（A速度与AD的夹角，弧度，取[-pi,pi]）
    gamma_deg = _wrap_deg(a_next_deg - ad_deg)
    gamma_rad = math.radians(gamma_deg)

    # theta = arcsin(alpha * sin(gamma))（弧度）
    arg = float(np.clip(alpha * math.sin(gamma_rad), -1.0, 1.0))
    theta_rad = math.asin(arg)
    theta_deg = math.degrees(theta_rad)

    # D 期望绝对朝向
    desired_deg = ad_deg + theta_deg
    # 转为相对转角并裁切
    rel_delta = _wrap_deg(desired_deg - curr_D_deg)
    return _clip_pair(rel_delta, 1.0)

def get_expert_target_action_pair(observation):
    """
    专家策略(Target): 近似微分对策方向，兼顾朝基地与规避追踪者。
    返回: (relative_angle_deg in [-45,45], speed_factor in [0,1])
    observation 索引约定:
      [4], [5]: tracker->target (d->a)
      [8], [9]: target->base (a->t)
      [11]: target朝向/360
    """
    eps = 1e-8
    v_d_to_a = np.array([observation[4], observation[5]], dtype=np.float64)
    v_at = np.array([observation[8], observation[9]], dtype=np.float64)
    v_ad = -v_d_to_a

    r = float(np.linalg.norm(v_ad))
    R = float(np.linalg.norm(v_at))

    if r < eps and R < eps:
        current_angle_deg = float(observation[11]) * 360.0
        return _clip_pair(0.0, 1.0)

    u_ad = v_ad / (r + eps)
    u_at = v_at / (R + eps)

    dot = float(np.clip(np.dot(u_ad, u_at), -1.0, 1.0))
    cross_z = float(u_ad[0] * u_at[1] - u_ad[1] * u_at[0])
    sin_alpha = float(abs(cross_z))

    d_len_sq = r * r + R * R - 2.0 * r * R * dot
    d_len = float(np.sqrt(max(d_len_sq, 0.0)))

    if d_len < eps:
        desired_dir = u_at
    else:
        sin_theta = (R - r * dot) / d_len
        cos_theta = (r * sin_alpha) / d_len
        sin_theta = float(np.clip(sin_theta, -1.0, 1.0))
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        sign = 1.0 if cross_z >= 0.0 else -1.0
        u_perp = np.array([-u_ad[1], u_ad[0]], dtype=np.float64)
        desired_dir = cos_theta * u_ad + sign * sin_theta * u_perp

    norm_des = float(np.linalg.norm(desired_dir))
    if norm_des < eps:
        desired_dir = u_at
        norm_des = float(np.linalg.norm(desired_dir))
        if norm_des < eps:
            desired_angle_deg = float(observation[11]) * 360.0
        else:
            desired_dir /= norm_des
            desired_angle_deg = float(np.degrees(np.arctan2(desired_dir[1], desired_dir[0])))
    else:
        desired_dir /= norm_des
        desired_angle_deg = float(np.degrees(np.arctan2(desired_dir[1], desired_dir[0])))

    current_angle_deg = float(observation[11]) * 360.0
    angle_diff = desired_angle_deg - current_angle_deg
    while angle_diff > 180.0:
        angle_diff -= 360.0
    while angle_diff < -180.0:
        angle_diff += 360.0

    return _clip_pair(angle_diff, 1.0)