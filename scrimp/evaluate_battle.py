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

        # 简化 mission 配置：统一使用 0（与 driver.py 保持一致）
        # mission=0: 不翻转奖励，适用于评估场景
        self.mission = 0

        self.run_dir = None
        self.run_timestamp = None

    def _continuous_to_discrete(self, angle_deg, speed_factor):
        """
        (保留但不再用于 env 交互)
        旧逻辑: 将连续动作粗糙量化为 (angle_idx, speed_idx)。
        已弃用：env.step 期望物理量 (angle_delta_deg, speed_factor)，
        且 PPO 训练使用 16x3 的自定义离散映射（见 Model.idx_to_pair / pair_to_idx）。
        如需统计或回放离散索引，应改用 Model.pair_to_idx。
        """
        angle_clamped = np.clip(angle_deg, -10.0, 10.0)
        angle_action = int(np.clip(np.round(angle_clamped / 10.0) + 10, 0, 20))
        speed_action = int(np.clip(np.round(speed_factor / 0.25) - 1, 0, 3))
        return angle_action, speed_action


def run_battle_batch(args):
    """运行一批episode（减少模型重复加载）"""
    config, episode_indices = args
    
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 一次性加载模型，批次内复用
    tracker_model = None
    target_model = None
    
    if config.tracker_type == "policy":
        tracker_model = Model(device, global_model=False)
        model_dict = torch.load(config.tracker_model_path, map_location=device)
        tracker_model.network.load_state_dict(model_dict['model'])
        tracker_model.network.eval()
    
    if config.target_type == "policy":
        target_model = Model(device, global_model=False)
        model_dict = torch.load(config.target_model_path, map_location=device)
        target_model.network.load_state_dict(model_dict['model'])
        target_model.network.eval()
    
    batch_results = []
    
    for episode_idx in episode_indices:
        try:
            result = run_single_episode(config, episode_idx, tracker_model, target_model, device)
            batch_results.append(result)
        except Exception:
            batch_results.append({
                "episode_id": episode_idx,
                "steps": 0,
                "reward": 0.0,
                "tracker_caught_target": False,
                "target_reached_base": False,
                "tracker_type": config.tracker_type,
                "target_type": config.target_type
            })
    
    return batch_results


def run_single_episode(config, episode_idx, tracker_model, target_model, device):
    """运行单个episode"""
    env = TrackingEnv(mission=config.mission)
    
    try:
        obs, _ = env.reset()
        done = False
        episode_step = 0
        episode_reward = 0.0
        tracker_caught_target = False
        target_reached_base = False
        
        save_gif = (config.save_gif_freq > 0 and episode_idx % config.save_gif_freq == 0)
        episode_frames = []
        
        tracker_hidden = None
        target_hidden = None
        
        if save_gif:
            try:
                frame = env.render(mode='rgb_array')
                if frame is not None:
                    episode_frames.append(frame)
            except Exception:
                save_gif = False
        
        with torch.no_grad():
            while not done and episode_step < EnvParameters.EPISODE_LEN:
                # Tracker 动作
                if config.tracker_type == "policy":
                    # 期望 Model.evaluate 返回: ( (angle, speed_factor), hidden, value, prob, action_idx )
                    tracker_eval = tracker_model.evaluate(obs, tracker_hidden, greedy=True)
                    # 兼容不同实现（如果用户稍后补全 model.evaluate）
                    if isinstance(tracker_eval, tuple) and len(tracker_eval) >= 1:
                        tracker_action = tracker_eval[0]
                        if len(tracker_eval) > 1:
                            tracker_hidden = tracker_eval[1]
                    else:
                        # 回退：如果模型还未实现，使用静态零动作
                        tracker_action = (0.0, 1.0)
                elif config.tracker_type == "expert_rule":
                    tracker_action = get_expert_tracker_action_pair(obs)  # 直接返回 (angle_delta_deg, speed_factor)
                else:
                    tracker_action = -1  # rule
                
                # Target 动作
                if config.target_type == "policy":
                    target_eval = target_model.evaluate(obs, target_hidden, greedy=True)
                    if isinstance(target_eval, tuple) and len(target_eval) >= 1:
                        target_action = target_eval[0]
                        if len(target_eval) > 1:
                            target_hidden = target_eval[1]
                    else:
                        target_action = (0.0, 1.0)
                elif config.target_type == "expert_rule":
                    target_action = get_expert_target_action_pair(obs)
                else:
                    target_action = -1

                # 与环境交互：直接传物理动作/ -1
                obs, reward, terminated, truncated, info = env.step(tracker_action, target_action)
                done = terminated or truncated
                episode_reward += float(reward)

                # 依据环境约定更新胜负标志
                reason = info.get("reason", "")
                if reason == "target_reached_base":
                    target_reached_base = True
                elif reason == "tracker_caught_target":
                    tracker_caught_target = True

                episode_step += 1

                if save_gif and episode_step % 2 == 0 and len(episode_frames) < 500:
                    try:
                        frame = env.render(mode='rgb_array')
                        if frame is not None:
                            episode_frames.append(frame)
                    except Exception:
                        save_gif = False
        
        if save_gif and len(episode_frames) > 1:
            try:
                if tracker_caught_target:
                    winner = "Tracker"
                elif target_reached_base:
                    winner = "Target"
                else:
                    winner = "Draw"
                ep_name = f"ep_{episode_idx:04d}_winner_{winner}.gif"
                gif_path = os.path.join(config.run_dir, ep_name)
                make_gif(episode_frames, gif_path)
            except Exception:
                pass
        
        return {
            "episode_id": episode_idx,
            "steps": episode_step,
            "reward": episode_reward,
            "tracker_caught_target": tracker_caught_target,
            "target_reached_base": target_reached_base,
            "tracker_type": config.tracker_type,
            "target_type": config.target_type
        }
    
    finally:
        env.close()


def run_battle(config):
    """运行完整的对战测试"""
    print(f"Running battle: {config.tracker_type} vs {config.target_type}, {config.episodes} episodes")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"battle_{config.tracker_type}_vs_{config.target_type}_{timestamp}"
    config.run_timestamp = timestamp
    config.run_dir = os.path.join(config.output_dir, run_name)
    os.makedirs(config.run_dir, exist_ok=True)
    
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    num_processes = min(cpu_count() // 2, 6)
    batch_size = max(10, config.episodes // max(num_processes, 1))
    
    start_time = time.time()
    results = []
    
    batches = []
    for batch_start in range(0, config.episodes, batch_size):
        batch_end = min(batch_start + batch_size, config.episodes)
        batch_episodes = list(range(batch_start, batch_end))
        batches.append((config, batch_episodes))
    
    try:
        with Pool(processes=num_processes) as pool:
            batch_results = pool.map(run_battle_batch, batches)
        for batch in batch_results:
            results.extend(batch)
    except Exception as e:
        print(f"Error in parallel execution: {e}")
        return None, None
    
    if not results:
        print("No successful episodes completed!")
        return None, None
    
    df = pd.DataFrame(results)

    avg_steps = float(df["steps"].mean()) if len(df) > 0 else 0.0
    avg_reward = float(df["reward"].mean()) if len(df) > 0 else 0.0
    tracker_win_rate = float(df['tracker_caught_target'].mean()) if len(df) > 0 else 0.0
    target_win_rate = float(df['target_reached_base'].mean()) if len(df) > 0 else 0.0
    last100 = df.tail(100)
    avg_steps_100 = float(last100["steps"].mean()) if len(last100) > 0 else 0.0

    tracker_speed = float(getattr(map_config, "tracker_speed", 4))
    target_speed = float(getattr(map_config, "target_speed", 6.0))
    tracker_turn_deg = float(getattr(map_config, "tracker_max_turn_deg", getattr(map_config, "max_turn_deg", 45.0)))
    target_turn_deg = float(getattr(map_config, "target_max_turn_deg", getattr(map_config, "max_turn_deg", 45.0)))
    
    try:
        results_path = os.path.join(config.run_dir, "results.csv")
        df.to_csv(results_path, index=False)
        
        stats = {
            "total_episodes": len(df),
            "avg_steps": avg_steps,
            "avg_steps_last_100": avg_steps_100,
            "avg_reward": avg_reward,
            "tracker_win_rate": tracker_win_rate,
            "target_win_rate": target_win_rate,
            "draw_rate": 1.0 - tracker_win_rate - target_win_rate,
            "tracker_type": config.tracker_type,
            "target_type": config.target_type,
            "tracker_speed": tracker_speed,
            "target_speed": target_speed,
            "tracker_turn_deg": tracker_turn_deg,
            "target_turn_deg": target_turn_deg,
            "state_space": config.state_space
        }
        stats_path = os.path.join(config.run_dir, "stats.csv")
        pd.DataFrame([stats]).to_csv(stats_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save results: {e}")
    
    total_time = time.time() - start_time
    print(f"\nBattle Results:")
    print(f"Episodes: {len(df)}")
    print(f"Avg steps: {avg_steps:.2f}")
    print(f"Tracker win rate: {tracker_win_rate*100:.2f}%")
    print(f"Target win rate: {target_win_rate*100:.2f}%")
    print(f"Time: {total_time:.2f}s")
    print(f"Saved to: {config.run_dir}")
    
    return df, config.run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Agent Battle Evaluation')
    parser.add_argument('--tracker', type=str, default='policy', choices=['rule', 'expert_rule', 'policy'])
    parser.add_argument('--target', type=str, default='expert_rule', choices=['rule', 'expert_rule', 'policy'])
    parser.add_argument('--tracker_model', type=str, default='./models/TrackingEnv/DualAgent10-09-251052/best_model/tracker_net_checkpoint.pkl')
    parser.add_argument('--target_model', type=str, default=None)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--save_gif_freq', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default='./scrimp_battle/battle_results')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--state_space', type=str, default='vector')

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