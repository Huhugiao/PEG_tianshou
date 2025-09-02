import math
import numpy as np

try:
    import pygame
    import pygame.gfxdraw
except Exception:
    pygame = None  # 渲染降级

import map_config


def reward_calculate(tracker, target, base, mission=0):
    reward = 0.0
    terminated = False
    truncated = False
    info = {}

    dist_tt = float(np.hypot(tracker['x'] - target['x'], tracker['y'] - target['y']))
    dist_tb = float(np.hypot(target['x'] - base['x'], target['y'] - base['y']))

    capture_radius = getattr(map_config, 'capture_radius', map_config.pixel_size)
    base_radius = getattr(map_config, 'base_radius', map_config.pixel_size)

    max_d = float(np.hypot(map_config.width, map_config.height))
    norm_t = dist_tt / max_d
    norm_b = dist_tb / max_d

    if dist_tb <= base_radius:
        reward = -getattr(map_config, 'success_reward', 1.0)
        terminated = True
        info['reason'] = 'target_reached_base'
    elif dist_tt <= capture_radius:
        reward = getattr(map_config, 'success_reward', 1.0)
        truncated = True
        info['reason'] = 'tracker_caught_target'
    else:
        reward = 0.8 * (1.0 - norm_t) - 1.0 * (1.0 - norm_b)
        reward -= 0.01

    if mission == 1:
        reward = -reward

    return float(reward), terminated, truncated, info


def _to_hi_res(pt):
    ss = getattr(map_config, 'ssaa', 1)
    return int(round(pt[0] * ss)), int(round(pt[1] * ss))


def _draw_grid(surface):
    if pygame is None:
        return
    if not getattr(map_config, 'draw_grid', True):
        return
    ss = getattr(map_config, 'ssaa', 1)
    step = int(map_config.grid_step * ss)
    color = map_config.grid_color
    w, h = surface.get_size()
    for x in range(0, w, step):
        pygame.draw.line(surface, color, (x, 0), (x, h), 1)
    for y in range(0, h, step):
        pygame.draw.line(surface, color, (0, y), (w, y), 1)


def _draw_base(surface, base_center):
    if pygame is None:
        return
    ss = getattr(map_config, 'ssaa', 1)
    cx, cy = _to_hi_res(base_center)
    r_out = int(map_config.base_radius_draw * ss)
    r_in = max(1, int((map_config.base_radius_draw - 3) * ss))
    if getattr(map_config, 'enable_aa', True):
        pygame.gfxdraw.filled_circle(surface, cx, cy, r_out, map_config.base_color_outer)
        pygame.gfxdraw.aacircle(surface, cx, cy, r_out, map_config.base_color_outer)
        pygame.gfxdraw.filled_circle(surface, cx, cy, r_in, map_config.base_color_inner)
        pygame.gfxdraw.aacircle(surface, cx, cy, r_in, map_config.base_color_inner)
    else:
        pygame.draw.circle(surface, map_config.base_color_outer, (cx, cy), r_out)
        pygame.draw.circle(surface, map_config.base_color_inner, (cx, cy), r_in)


def _triangle_points(center, angle_deg, base, height):
    cx, cy = center
    theta = math.radians(angle_deg)
    tip = (cx + height * math.cos(theta), cy + height * math.sin(theta))
    lt = theta + math.radians(150)
    rt = theta - math.radians(150)
    left = (cx + base * math.cos(lt), cy + base * math.sin(lt))
    right = (cx + base * math.cos(rt), cy + base * math.sin(rt))
    return [tip, left, right]


def _draw_agent(surface, agent, color):
    if pygame is None:
        return
    ss = getattr(map_config, 'ssaa', 1)
    px = agent['x'] + map_config.pixel_size / 2.0
    py = agent['y'] + map_config.pixel_size / 2.0
    tri = _triangle_points((px * ss, py * ss), float(agent.get('theta', 0.0)),
                           base=6 * ss, height=10 * ss)
    if getattr(map_config, 'enable_aa', True):
        pygame.gfxdraw.filled_trigon(surface, int(tri[0][0]), int(tri[0][1]),
                                     int(tri[1][0]), int(tri[1][1]),
                                     int(tri[2][0]), int(tri[2][1]),
                                     color)
        pygame.gfxdraw.aatrigon(surface, int(tri[0][0]), int(tri[0][1]),
                                int(tri[1][0]), int(tri[1][1]),
                                int(tri[2][0]), int(tri[2][1]),
                                color)
    else:
        cx, cy = _to_hi_res((px, py))
        pygame.draw.circle(surface, color, (cx, cy), int(map_config.agent_radius * ss))


def _draw_trail(surface, traj, rgba, width_px):
    if pygame is None:
        return
    if len(traj) < 2:
        return
    ss = getattr(map_config, 'ssaa', 1)
    pts = [_to_hi_res(p) for p in traj[-getattr(map_config, 'trail_max_len', 600):]]
    pygame.draw.lines(surface, rgba, False, pts, max(int(width_px * ss), 1))


def get_canvas(target, tracker, base, tracker_trajectory, target_trajectory):
    w, h = map_config.width, map_config.height
    ss = getattr(map_config, 'ssaa', 1)
    if pygame is None:
        # 降级：返回空画布
        return np.zeros((h, w, 3), dtype=np.uint8)

    surface = pygame.Surface((w * ss, h * ss), flags=pygame.SRCALPHA)
    surface.fill(map_config.background_color)

    _draw_grid(surface)
    _draw_trail(surface, tracker_trajectory, map_config.trail_color_tracker, map_config.trail_width)
    _draw_trail(surface, target_trajectory, map_config.trail_color_target, map_config.trail_width)

    base_center = (base['x'] + map_config.pixel_size / 2.0, base['y'] + map_config.pixel_size / 2.0)
    _draw_base(surface, base_center)
    _draw_agent(surface, tracker, map_config.tracker_color)
    _draw_agent(surface, target, map_config.target_color)

    if ss > 1:
        canvas = pygame.transform.smoothscale(surface, (w, h))
        canvas = pygame.surfarray.array3d(canvas).swapaxes(0, 1)
    else:
        canvas = pygame.surfarray.array3d(surface).swapaxes(0, 1)
    return canvas

def agent_move(agent, action, moving_size, role=None):
    """
    action: (angle_delta_deg, speed_factor)
      - angle_delta_deg clipped per-agent to [-max_turn_deg, max_turn_deg]
      - speed_factor clipped to [0, 1], actual speed = speed_factor * moving_size
    role: 'tracker' or 'target' (optional). 若不传，则根据 moving_size 推断。
    """
    # 兼容输入为list/tuple/np.ndarray
    if isinstance(action, (list, tuple, np.ndarray)) and len(action) == 2:
        angle_delta, speed_factor = float(action[0]), float(action[1])
    else:
        raise ValueError("agent_move expects action=(angle_delta_deg, speed_factor)")

    # 推断角色
    if role is None:
        try:
            if abs(float(moving_size) - float(getattr(map_config, 'tracker_speed'))) < 1e-6:
                role = 'tracker'
            elif abs(float(moving_size) - float(getattr(map_config, 'target_speed'))) < 1e-6:
                role = 'target'
        except Exception:
            role = None

    # 角速度限幅（按角色）
    default_turn = float(getattr(map_config, 'max_turn_deg', 45.0))
    if role == 'tracker':
        max_turn = float(getattr(map_config, 'tracker_max_turn_deg', default_turn))
    elif role == 'target':
        max_turn = float(getattr(map_config, 'target_max_turn_deg', default_turn))
    else:
        max_turn = default_turn

    # 裁切
    angle_delta = float(np.clip(angle_delta, -max_turn, max_turn))
    speed_factor = float(np.clip(speed_factor, 0.0, 1.0))
    speed = float(speed_factor * float(moving_size))

    current_angle = float(agent.get('theta', 0.0))
    new_angle = (current_angle + angle_delta) % 360.0
    agent['theta'] = float(new_angle)

    agent['x'] = float(np.clip(agent['x'] + speed * math.cos(math.radians(new_angle)),
                               0, map_config.width - map_config.pixel_size))
    agent['y'] = float(np.clip(agent['y'] + speed * math.sin(math.radians(new_angle)),
                               0, map_config.height - map_config.pixel_size))
    return agent


def target_nav(target, tracker, base, moving_size, frame_count):
    tracker_vec = np.array([tracker['x'] - target['x'], tracker['y'] - target['y']], dtype=float)
    base_vec = np.array([base['x'] - target['x'], base['y'] - target['y']], dtype=float)

    d_t = float(np.linalg.norm(tracker_vec))
    d_b = float(np.linalg.norm(base_vec))

    base_dir = base_vec / d_b if d_b > 1e-6 else np.zeros(2, dtype=float)
    safe_radius = 150.0
    scale = min(safe_radius / max(d_t, 1e-6), 1.0)

    if d_t > 1e-6:
        perp = np.array([-tracker_vec[1], tracker_vec[0]], dtype=float) / d_t
    else:
        perp = np.array([0.0, 0.0], dtype=float)

    move_dir = base_dir + perp * scale * 2.5
    if d_t < 50.0:
        move_dir = perp

    norm = float(np.linalg.norm(move_dir))
    if norm > 1e-6:
        move_dir = move_dir / norm
    else:
        move_dir = np.zeros(2, dtype=float)

    new_pos = target.copy()
    new_pos['x'] = float(np.clip(new_pos['x'] + move_dir[0] * moving_size, 0, map_config.width - map_config.pixel_size))
    new_pos['y'] = float(np.clip(new_pos['y'] + move_dir[1] * moving_size, 0, map_config.height - map_config.pixel_size))
    new_theta = float((math.degrees(math.atan2(move_dir[1], move_dir[0])) % 360.0)) if norm > 1e-6 else float(target.get('theta', 0.0))
    new_pos['theta'] = new_theta
    return new_pos


def tracker_nav(tracker, target, moving_size):
    vec = np.array([target['x'] - tracker['x'], target['y'] - tracker['y']], dtype=float)
    d = float(np.linalg.norm(vec))
    if d > 1e-6:
        dir = vec / d
    else:
        dir = np.zeros(2, dtype=float)
    bias = 0.10
    dir = (1.0 - bias) * dir + bias * np.array([1.0, 0.0], dtype=float)
    n = float(np.linalg.norm(dir))
    dir = dir / n if n > 1e-6 else dir

    nx = float(np.clip(tracker['x'] + dir[0] * moving_size, 0, map_config.width - map_config.pixel_size))
    ny = float(np.clip(tracker['y'] + dir[1] * moving_size, 0, map_config.height - map_config.pixel_size))
    theta = float((math.degrees(math.atan2(dir[1], dir[0])) % 360.0)) if n > 1e-6 else float(tracker.get('theta', 0.0))
    new_pos = dict(tracker)
    new_pos['x'] = nx
    new_pos['y'] = ny
    new_pos['theta'] = theta
    return new_pos