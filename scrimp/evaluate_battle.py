import os
import numpy as np
import torch
import ray
import argparse
import pandas as pd
import time
from multiprocessing import Pool, cpu_count
import datetime  # 修改这里

from env import TrackingEnv
from model import Model
from alg_parameters import *
from util import make_gif


class BattleConfig:
    def __init__(self, 
                 tracker_type="rule", 
                 target_type="rule",
                 tracker_model_path=None,
                 target_model_path=None,
                 episodes=100,
                 save_gif_freq=10,
                 output_dir="./battle_results",
                 gif_dir="./battle_gifs",
                 seed=1234):
        
        self.tracker_type = tracker_type  # "rule" 或 "policy"
        self.target_type = target_type    # "rule", "expert_rule", 或 "policy"
        self.tracker_model_path = tracker_model_path
        self.target_model_path = target_model_path
        self.episodes = episodes
        self.save_gif_freq = save_gif_freq
        self.output_dir = output_dir
        self.gif_dir = gif_dir
        self.seed = seed
        
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(gif_dir):
            os.makedirs(gif_dir)
            
        # 根据对战类型自动确定mission
        if tracker_type == "rule" and target_type == "rule":
            self.mission = -1  # 自定义mission：两个都是规则
        elif tracker_type == "policy" and target_type == "rule":
            self.mission = 2   # tracker是policy，target是rule
        elif tracker_type == "rule" and target_type == "policy":
            self.mission = 1   # tracker是rule，target是policy
        elif tracker_type == "rule" and target_type == "expert_rule":
            self.mission = 3   # tracker是rule，target使用专家规则
        elif tracker_type == "policy" and target_type == "expert_rule":
            self.mission = 3   # tracker是policy，target使用专家规则
        else:
            self.mission = 2   # 都是policy

def get_expert_target_action(observation):
    """Expert action for target agent with improved lateral evasion strategy"""
    # Extract observation components
    tracker_to_target_x = observation[4]
    tracker_to_target_y = observation[5]
    target_to_base_x = observation[8]
    target_to_base_y = observation[9]
    current_angle = observation[11] * 360  # target_angle in degrees
    
    # Calculate distances
    distance_to_tracker = np.sqrt(tracker_to_target_x**2 + tracker_to_target_y**2)
    distance_to_base = np.sqrt(target_to_base_x**2 + target_to_base_y**2)
    
    # Calculate base and escape directions
    base_angle = np.arctan2(target_to_base_y, target_to_base_x)
    escape_angle = np.arctan2(-tracker_to_target_y, -tracker_to_target_x)
    
    # Calculate lateral escape vectors (perpendicular to tracker direction)
    lateral_angle_right = escape_angle + np.pi/2  # 90 degrees clockwise
    lateral_angle_left = escape_angle - np.pi/2   # 90 degrees counterclockwise
    
    # Choose the lateral direction that's closer to the base
    base_right_diff = np.abs(np.cos(lateral_angle_right - base_angle))
    base_left_diff = np.abs(np.cos(lateral_angle_left - base_angle))
    lateral_angle = lateral_angle_right if base_right_diff > base_left_diff else lateral_angle_left
    
    # Dynamic weighting based on tracker distance and position
    danger_threshold = 30 # Distance threshold for evasive action
    if distance_to_tracker < danger_threshold:
        # Tracker is close - prioritize evasion
        base_weight = 0.2
        escape_weight = 0.0
        lateral_weight = 0.8  # Strong lateral movement when being pursued
    else:
        # Safe distance - balance between base and lateral movement
        base_weight = 0.03
        escape_weight = 0.0
        lateral_weight = 0.97  # Strong lateral movement when being pursued
    
    # Combine directions with dynamic weights
    mixed_angle = (base_angle * base_weight) + (escape_angle * escape_weight) + (lateral_angle * lateral_weight)
    
    # Convert to degrees and normalize to 0-360
    desired_angle = (np.degrees(mixed_angle) + 360) % 360
    
    # Calculate relative turning angle needed
    relative_angle = ((desired_angle - current_angle + 180) % 360) - 180
    
    # Map to closest available relative angle in action space (-45 to 45)
    relative_angle = np.clip(relative_angle, -45, 45)
    
    # Map to direction index (0-15) for the action space
    direction_index = int((relative_angle + 45) / 90 * 16)
    direction_index = np.clip(direction_index, 0, 15)
    
    # Always use highest speed
    speed_level = 2  # always use fastest speed
    
    # Combine direction and speed for final action
    expert_action = direction_index * 3 + speed_level
    
    return expert_action

def run_battle_episode(config, episode_idx):
    """运行单个对战回合"""
    # 设置设备
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # 创建环境
    env = TrackingEnv(mission=config.mission)
    
    # 初始化模型
    tracker_model = None
    target_model = None
    
    # 加载tracker模型
    if config.tracker_type == "policy":
        tracker_model = Model(device, global_model=False)
        model_dict = torch.load(config.tracker_model_path, map_location=device)
        tracker_model.network.load_state_dict(model_dict['model'])
    
    # 加载target模型
    if config.target_type == "policy":
        target_model = Model(device, global_model=False)
        model_dict = torch.load(config.target_model_path, map_location=device)
        target_model.network.load_state_dict(model_dict['model'])
    
    # 重置环境
    obs, _ = env.reset()
    done = False
    episode_step = 0
    episode_reward = 0  # 从tracker的角度看的奖励
    tracker_caught_target = False
    target_reached_base = False
    
    # 决定是否保存GIF
    save_gif = (episode_idx % config.save_gif_freq == 0)
    episode_frames = [] if save_gif else None
    
    # 初始化隐藏状态
    tracker_hidden = None
    target_hidden = None
    
    if save_gif:
        try:
            frame = env.render(mode='rgb_array')
            if frame is not None:
                episode_frames.append(frame)
        except Exception as e:
            print(f"Error capturing initial frame: {e}")
    
    # 开始回合
    while not done and episode_step < EnvParameters.EPISODE_LEN:
        # 获取tracker动作
        if config.tracker_type == "policy":
            tracker_action, tracker_hidden, _, _ = tracker_model.evaluate(obs, tracker_hidden, greedy=True)
        else:
            # For rule-based tracker, we need to provide a valid action or let environment handle it
            # Check if environment expects specific action format
            tracker_action = -1  # Use -1 to indicate rule-based action
        
        # 获取target动作
        if config.target_type == "policy":
            target_action, target_hidden, _, _ = target_model.evaluate(obs, target_hidden, greedy=True)
        elif config.target_type == "expert_rule":
            # 使用Runner中的专家规则
            target_action = get_expert_target_action(obs)
        else:
            # For rule-based target, use -1 to indicate rule-based action
            target_action = -1
        
        # 执行动作
        try:
            obs, reward, terminated, truncated, info = env.step(tracker_action, target_action)
            done = terminated or truncated
            # 记录结果
            episode_reward += reward
            
            # 直接通过terminated和truncated判断胜负
            if terminated:
                target_reached_base = True  # Target reaches base is handled by terminated
            if truncated:
                tracker_caught_target = True  # Tracker catches target is handled by truncated
            
        except Exception as e:
            print(f"Error in step: {e}")
            break
        
        episode_step += 1
        
        # 保存GIF帧
        if save_gif:
            try:
                frame = env.render(mode='rgb_array')
                if frame is not None:
                    episode_frames.append(frame)
            except Exception as e:
                print(f"Error capturing frame: {e}")
    
    # 保存GIF
    if save_gif and episode_frames:
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # 修改这里
            gif_path = f"{config.gif_dir}/battle_{config.tracker_type}_vs_{config.target_type}_{episode_idx}_{timestamp}.gif"
            make_gif(episode_frames, gif_path)
        except Exception as e:
            print(f"Error saving GIF: {e}")
    
    # 关闭环境
    env.close()
    
    # 返回结果
    result = {
        "episode_id": episode_idx,
        "steps": episode_step,
        "reward": episode_reward,
        "tracker_caught_target": tracker_caught_target,
        "target_reached_base": target_reached_base,
        "tracker_type": config.tracker_type,
        "target_type": config.target_type
    }
    
    return result

def run_battle(config):
    """运行完整的对战测试"""
    print(f"开始对战评估: {config.tracker_type} tracker vs {config.target_type} target")
    print(f"总回合数: {config.episodes}, 每{config.save_gif_freq}回合保存一次GIF")
    
    # 设置随机种子
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # 使用进程池并行运行对战
    num_processes = min(cpu_count(), 8)  # 限制最大进程数
    print(f"使用{num_processes}个并行进程运行")
    
    start_time = time.time()
    
    # 准备参数
    args = [(config, i) for i in range(config.episodes)]
    
    # 使用进程池并行执行
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(run_battle_episode, args)
    
    # 处理结果
    df = pd.DataFrame(results)
    
    # 计算统计数据
    stats = {
        "total_episodes": len(df),
        "avg_steps": df["steps"].mean(),
        "avg_reward": df["reward"].mean(),
        "tracker_win_rate": df["tracker_caught_target"].mean(),
        "target_win_rate": df["target_reached_base"].mean(),
        "draw_rate": 1 - df["tracker_caught_target"].mean() - df["target_reached_base"].mean(),
        "tracker_type": config.tracker_type,
        "target_type": config.target_type
    }
    
    # 保存详细结果
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # 修改这里
    results_path = f"{config.output_dir}/battle_{config.tracker_type}_vs_{config.target_type}_{timestamp}.csv"
    df.to_csv(results_path, index=False)
    
    # 保存统计结果
    stats_path = f"{config.output_dir}/stats_{config.tracker_type}_vs_{config.target_type}_{timestamp}.csv"
    pd.DataFrame([stats]).to_csv(stats_path, index=False)
    
    # 打印统计信息
    print("\n对战评估结果:")
    print(f"{'Tracker类型':<15}: {config.tracker_type}")
    print(f"{'Target类型':<15}: {config.target_type}")
    print(f"{'总回合数':<15}: {stats['total_episodes']}")
    print(f"{'平均步数':<15}: {stats['avg_steps']:.2f}")
    print(f"{'平均奖励':<15}: {stats['avg_reward']:.2f}")
    print(f"{'Tracker胜率':<15}: {stats['tracker_win_rate']*100:.2f}%")
    print(f"{'Target胜率':<15}: {stats['target_win_rate']*100:.2f}%")
    print(f"{'平局率':<15}: {stats['draw_rate']*100:.2f}%")
    print(f"详细结果已保存到: {results_path}")
    print(f"统计结果已保存到: {stats_path}")
    
    print(f"\n评估完成，用时: {time.time() - start_time:.2f}秒")
    
    return stats, df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Agent Battle Evaluation')
    parser.add_argument('--tracker', type=str, default='rule', choices=['rule', 'policy'],
                        help='Tracker agent type (rule or policy)')
    parser.add_argument('--target', type=str, default='expert_rule', choices=['rule', 'expert_rule', 'policy'],
                        help='Target agent type (rule, expert_rule, or policy)')
    parser.add_argument('--tracker_model', type=str, default=None,
                        help='Path to tracker model checkpoint (required if tracker=policy)')
    parser.add_argument('--target_model', type=str, default=None,
                        help='Path to target model checkpoint (required if target=policy)')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of battle episodes')
    parser.add_argument('--save_gif_freq', type=int, default=10,
                        help='Save GIF every N episodes (0 to disable)')
    parser.add_argument('--output_dir', type=str, default='./scrimp_battle/battle_results',
                        help='Directory to save battle results')
    parser.add_argument('--gif_dir', type=str, default='./scrimp_battle/battle_gifs',
                        help='Directory to save battle GIFs')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed')
                        
    args = parser.parse_args()
    
    # 验证参数
    if args.tracker == 'policy' and args.tracker_model is None:
        parser.error("--tracker_model is required when tracker is 'policy'")
    if args.target == 'policy' and args.target_model is None:
        parser.error("--target_model is required when target is 'policy'")
    
    # 创建配置并运行评估
    config = BattleConfig(
        tracker_type=args.tracker,
        target_type=args.target,
        tracker_model_path=args.tracker_model,
        target_model_path=args.target_model,
        episodes=args.episodes,
        save_gif_freq=args.save_gif_freq,
        output_dir=args.output_dir,
        gif_dir=args.gif_dir,
        seed=args.seed
    )
    
    run_battle(config)