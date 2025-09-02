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
    Tracker专家策略（微分博弈解，基于 DA 视线）
    """
    try:
        v_d_to_a = np.array([float(observation[4]), float(observation[5])], dtype=np.float64)  # D->A（未必单位）
    except Exception:
        v_d_to_a = np.array([0.0, 0.0], dtype=np.float64)

    curr_D_deg = float(observation[10] * 360.0) if len(observation) > 10 else 0.0
    curr_A_deg = float(observation[11] * 360.0) if len(observation) > 11 else 0.0

    vD = float(getattr(map_config, 'tracker_speed'))
    vA = float(max(getattr(map_config, 'target_speed'), 1e-6))
    alpha = 1
    # alpha = float(np.clip(vD / vA, 0.0, 10.0))

    norm_da = float(np.linalg.norm(v_d_to_a))
    if norm_da < 1e-8:
        return _clip_pair(0.0, 1.0)

    u_da = v_d_to_a / norm_da
    da_deg = float(np.degrees(np.arctan2(u_da[1], u_da[0])))

    a_rel_deg, _ = get_expert_target_action_pair(observation)
    a_next_deg = (curr_A_deg + float(a_rel_deg)) % 360.0

    gamma_deg = _wrap_deg(a_next_deg - da_deg)
    gamma_rad = math.radians(gamma_deg)

    arg = float(np.clip(alpha * math.sin(gamma_rad), -1.0, 1.0))
    theta_deg = math.degrees(math.asin(arg))

    desired_deg = da_deg + theta_deg
    rel_delta = _wrap_deg(desired_deg - curr_D_deg)
    return _clip_pair(rel_delta, 1.0)

def get_expert_target_action_pair(observation):
    """
    专家策略(Target): 基于“朝基地方向”为基向量的合成方向（修正版，不加额外限制）
    返回: (relative_angle_deg in [-45,45], speed_factor in [0,1])
    observation 索引约定:
      [4], [5]: tracker->target (d->a)
      [8], [9]: target->base (a->t)
      [11]: target朝向/360
    """
    eps = 1e-8
    v_d_to_a = np.array([float(observation[4]), float(observation[5])], dtype=np.float64)
    v_at = np.array([float(observation[8]), float(observation[9])], dtype=np.float64)
    v_ad = -v_d_to_a

    r = float(np.linalg.norm(v_ad))
    R = float(np.linalg.norm(v_at))

    if r < eps and R < eps:
        current_angle_deg = float(observation[11]) * 360.0
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
        # 原公式计算的“角参数”
        sin_theta = (R - r * dot) / d_len
        cos_theta = (r * sin_alpha) / d_len
        sin_theta = float(np.clip(sin_theta, -1.0, 1.0))
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))

        # 关键修正：以 u_at 为基向量合成，不再围绕 u_ad 旋转
        # 侧向方向取“相对 u_ad 的垂直方向，并选取与 u_at 更同向的一侧”
        u_perp1 = np.array([-u_ad[1], u_ad[0]], dtype=np.float64)
        u_perp2 = -u_perp1
        u_perp = u_perp1 if float(np.dot(u_perp1, u_at)) >= float(np.dot(u_perp2, u_at)) else u_perp2

        desired_dir = cos_theta * u_at + sin_theta * u_perp

    # 归一化并转角
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
    angle_diff = _wrap_deg(angle_diff)

    return _clip_pair(angle_diff, 1.0)