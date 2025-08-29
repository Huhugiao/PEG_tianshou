import os
import numpy as np
import torch
import argparse
import pandas as pd
import time
from multiprocessing import Pool, cpu_count
import datetime
import map_config

from env import TrackingEnv
from model import Model
from alg_parameters import *
from util import make_gif
from expert_policies import get_expert_target_action_pair, get_expert_tracker_action_pair


class BattleConfig:
    def __init__(self, 
                 tracker_type="expert_rule", 
                 target_type="expert_rule",
                 tracker_model_path=None,
                 target_model_path=None,
                 episodes=100,
                 save_gif_freq=10,
                 output_dir="./battle_results",
                 gif_dir="./battle_gifs",
                 seed=1234):
        
        self.tracker_type = tracker_type  # "rule", "expert_rule", 或 "policy"
        self.target_type = target_type    # "rule", "expert_rule", 或 "policy"
        self.tracker_model_path = tracker_model_path
        self.target_model_path = target_model_path
        self.episodes = episodes
        self.save_gif_freq = save_gif_freq
        self.output_dir = output_dir
        self.gif_dir = gif_dir
        self.seed = seed
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(gif_dir):
            os.makedirs(gif_dir)
            
        # mission推断：任一侧为专家规则 -> 3；其余维持原逻辑
        if tracker_type == "rule" and target_type == "rule":
            self.mission = -1
        elif "expert_rule" in {tracker_type, target_type}:
            self.mission = 3
        elif tracker_type == "policy" and target_type == "rule":
            self.mission = 2
        elif tracker_type == "rule" and target_type == "policy":
            self.mission = 1
        else:
            self.mission = 2


def run_battle_episode(config, episode_idx):
    """运行单个对战回合"""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    env = TrackingEnv(mission=config.mission)
    
    tracker_model = None
    target_model = None
    
    if config.tracker_type == "policy":
        tracker_model = Model(device, global_model=False)
        model_dict = torch.load(config.tracker_model_path, map_location=device)
        tracker_model.network.load_state_dict(model_dict['model'])
    
    if config.target_type == "policy":
        target_model = Model(device, global_model=False)
        model_dict = torch.load(config.target_model_path, map_location=device)
        target_model.network.load_state_dict(model_dict['model'])
    
    obs, _ = env.reset()
    done = False
    episode_step = 0
    episode_reward = 0
    tracker_caught_target = False
    target_reached_base = False
    
    save_gif = (config.save_gif_freq > 0 and episode_idx % config.save_gif_freq == 0)
    episode_frames = [] if save_gif else None
    
    tracker_hidden = None
    target_hidden = None
    
    if save_gif:
        try:
            frame = env.render(mode='rgb_array')
            if frame is not None:
                episode_frames.append(frame)
        except Exception as e:
            print(f"Error capturing initial frame: {e}")
    
    while not done and episode_step < EnvParameters.EPISODE_LEN:
        # tracker
        if config.tracker_type == "policy":
            tracker_action, tracker_hidden, _, _, _ = tracker_model.evaluate(obs, tracker_hidden, greedy=True)
        elif config.tracker_type == "expert_rule":
            tracker_action = get_expert_tracker_action_pair(obs)
        else:
            tracker_action = -1  # 规则
        
        # target
        if config.target_type == "policy":
            target_action, target_hidden, _, _, _ = target_model.evaluate(obs, target_hidden, greedy=True)
        elif config.target_type == "expert_rule":
            target_action = get_expert_target_action_pair(obs)
        else:
            target_action = -1
        
        try:
            obs, reward, terminated, truncated, info = env.step(tracker_action, target_action)
            done = terminated or truncated
            episode_reward += reward
            if terminated:
                target_reached_base = True
            if truncated:
                tracker_caught_target = True
        except Exception as e:
            print(f"Error in step: {e}")
            break
        
        episode_step += 1
        
        if save_gif:
            try:
                frame = env.render(mode='rgb_array')
                if frame is not None:
                    episode_frames.append(frame)
            except Exception as e:
                print(f"Error capturing frame: {e}")
    
    if save_gif and episode_frames:
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            gif_path = f"{config.gif_dir}/battle_{config.tracker_type}_vs_{config.target_type}_{episode_idx}_{timestamp}.gif"
            make_gif(episode_frames, gif_path)
        except Exception as e:
            print(f"Error saving GIF: {e}")
    
    env.close()
    
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
    
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    num_processes = min(cpu_count(), 8)
    print(f"使用{num_processes}个并行进程运行")
    
    start_time = time.time()
    args = [(config, i) for i in range(config.episodes)]
    
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(run_battle_episode, args)
    
    df = pd.DataFrame(results)
    
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
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"{config.output_dir}/battle_{config.tracker_type}_vs_{config.target_type}_{timestamp}.csv"
    df.to_csv(results_path, index=False)
     
    stats_path = f"{config.output_dir}/stats_{config.tracker_type}_vs_{config.target_type}_{timestamp}.csv"
    pd.DataFrame([stats]).to_csv(stats_path, index=False)
    
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
    parser.add_argument('--tracker', type=str, default='expert_rule', choices=['rule', 'expert_rule', 'policy'],
                        help='Tracker agent type (rule, expert_rule, or policy)')
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
    
    if args.tracker == 'policy' and args.tracker_model is None:
        parser.error("--tracker_model is required when tracker is 'policy'")
    if args.target == 'policy' and args.target_model is None:
        parser.error("--target_model is required when target is 'policy'")
    
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