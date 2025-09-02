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
                 seed=1234,
                 state_space="vector"):
        self.tracker_type = tracker_type
        self.target_type = target_type
        self.tracker_model_path = tracker_model_path
        self.target_model_path = target_model_path
        self.episodes = episodes
        self.save_gif_freq = save_gif_freq
        self.output_dir = output_dir
        self.seed = seed
        self.state_space = str(state_space)

        os.makedirs(output_dir, exist_ok=True)

        # mission推断
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

        # 运行目录（run_battle 中创建）
        self.run_dir = None
        self.run_timestamp = None


def run_battle_episode(config, episode_idx):
    """运行单个对战回合（速度与角速度均由 map_config 提供）"""
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
    episode_reward = 0.0
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
            tracker_action = -1  # env 内置 rule
        
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
            episode_reward += float(reward)
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
            ep_name = f"ep_{episode_idx:04d}.gif"
            gif_path = os.path.join(config.run_dir, ep_name)
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

    # 每次评估单独目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"battle_{config.tracker_type}_vs_{config.target_type}_{timestamp}"
    config.run_timestamp = timestamp
    config.run_dir = os.path.join(config.output_dir, run_name)
    os.makedirs(config.run_dir, exist_ok=True)
    print(f"结果目录: {config.run_dir}")
    
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    num_processes = min(cpu_count(), 8)
    print(f"使用{num_processes}个并行进程运行")
    
    start_time = time.time()
    args = [(config, i) for i in range(config.episodes)]
    
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(run_battle_episode, args)
    
    df = pd.DataFrame(results)

    # 统计
    avg_steps = float(df["steps"].mean()) if len(df) > 0 else 0.0
    last100 = df.tail(100)
    avg_steps_100 = float(last100["steps"].mean()) if len(last100) > 0 else 0.0

    # 从 map_config 读取速度/角速度（evaluate 不再设置）
    tracker_speed = float(getattr(map_config, "tracker_speed", 4.8))
    target_speed = float(getattr(map_config, "target_speed", 8.0))
    tracker_turn_deg = float(getattr(map_config, "tracker_max_turn_deg", getattr(map_config, "max_turn_deg", 45.0)))
    target_turn_deg = float(getattr(map_config, "target_max_turn_deg", getattr(map_config, "max_turn_deg", 45.0)))
    
    stats = {
        "total_episodes": int(len(df)),
        "avg_steps": avg_steps,
        "avg_steps_last_100": avg_steps_100,
        "avg_reward": float(df["reward"].mean()) if len(df) > 0 else 0.0,
        "tracker_win_rate": float(df["tracker_caught_target"].mean()) if len(df) > 0 else 0.0,
        "target_win_rate": float(df["target_reached_base"].mean()) if len(df) > 0 else 0.0,
        "draw_rate": (1.0 - float(df["tracker_caught_target"].mean()) - float(df["target_reached_base"].mean())) if len(df) > 0 else 0.0,
        "tracker_type": config.tracker_type,
        "target_type": config.target_type,
        "tracker_speed": tracker_speed,
        "target_speed": target_speed,
        "tracker_turn_deg": tracker_turn_deg,
        "target_turn_deg": target_turn_deg,
        "state_space": config.state_space
    }
    
    # CSV：主数据 + 末尾附 summary 与 params 两行
    results_path = os.path.join(config.run_dir, "results.csv")
    df.to_csv(results_path, index=False)
    summary_row = {
        "episode_id": "summary",
        "steps": avg_steps,
        "reward": float(df["reward"].mean()) if len(df) > 0 else 0.0,
        "tracker_caught_target": float(df["tracker_caught_target"].mean()) if len(df) > 0 else 0.0,
        "target_reached_base": float(df["target_reached_base"].mean()) if len(df) > 0 else 0.0,
        "tracker_type": config.tracker_type,
        "target_type": config.target_type,
        "avg_steps_100": avg_steps_100
    }
    params_row = {
        "episode_id": "params",
        "tracker_speed": tracker_speed,
        "target_speed": target_speed,
        "tracker_turn_deg": tracker_turn_deg,
        "target_turn_deg": target_turn_deg,
        "state_space": config.state_space
    }
    pd.DataFrame([summary_row, params_row]).to_csv(results_path, mode="a", header=False, index=False)
     
    stats_path = os.path.join(config.run_dir, "stats.csv")
    pd.DataFrame([stats]).to_csv(stats_path, index=False)
    
    print("\n对战评估结果:")
    print(f"{'Tracker类型':<20}: {config.tracker_type}")
    print(f"{'Target类型':<20}: {config.target_type}")
    print(f"{'总回合数':<20}: {stats['total_episodes']}")
    print(f"{'平均步数':<20}: {stats['avg_steps']:.2f}")
    print(f"{'最近100局平均步数':<20}: {stats['avg_steps_last_100']:.2f}")
    print(f"{'平均奖励':<20}: {stats['avg_reward']:.2f}")
    print(f"{'Tracker胜率':<20}: {stats['tracker_win_rate']*100:.2f}%")
    print(f"{'Target胜率':<20}: {stats['target_win_rate']*100:.2f}%")
    print(f"{'平局率':<20}: {stats['draw_rate']*100:.2f}%")
    print(f"{'Tracker线/角速度':<20}: {tracker_speed} / {tracker_turn_deg} deg")
    print(f"{'Target线/角速度':<20}: {target_speed} / {target_turn_deg} deg")
    print(f"{'状态空间':<20}: {config.state_space}")
    print(f"已保存到: {config.run_dir}（包含 GIF 与 CSV）")
    
    print(f"\n评估完成，用时: {time.time() - start_time:.2f}秒")
    return stats, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Agent Battle Evaluation')
    parser.add_argument('--tracker', type=str, default='expert_rule', choices=['rule', 'expert_rule', 'policy'])
    parser.add_argument('--target', type=str, default='expert_rule', choices=['rule', 'expert_rule', 'policy'])
    parser.add_argument('--tracker_model', type=str, default=None)
    parser.add_argument('--target_model', type=str, default=None)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--save_gif_freq', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default='./scrimp_battle/battle_results')
    parser.add_argument('--seed', type=int, default=1234)
    # 仅记录标签（速度与角速度请在 map_config 中配置）
    parser.add_argument('--state_space', type=str, default='vector',
                        help='Label to record in CSV, e.g. vector/god_view')

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
        seed=args.seed,
        state_space=args.state_space
    )
    run_battle(config)