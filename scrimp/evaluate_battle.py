import os
import numpy as np
import torch
import argparse
import pandas as pd
import time
from multiprocessing import Pool, cpu_count
import datetime
import map_config
import json
from collections import defaultdict

from env import TrackingEnv
from model import Model
from alg_parameters import *
from util import make_gif
from expert_policies import get_expert_target_action_pair, get_expert_tracker_action_pair
from policymanager import PolicyManager


def get_available_policies(role):
    """获取指定角色的所有可用策略"""
    if role == "tracker":
        return ["expert_tracker", "predictive_tracker", "circle_tracker", 
                "patrol_tracker", "random_tracker", "area_denial_tracker"]
    elif role == "target":
        return ["expert_target", "zigzag_target", "edge_hugging_target", 
                "feinting_target", "spiral_target", "tracker_aware_target"]
    else:
        raise ValueError(f"Unknown role: {role}")


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
                 state_space="vector",
                 specific_tracker_strategy=None,
                 specific_target_strategy=None,
                 main_output_dir=None):  # 新增：主输出目录
        self.tracker_type = tracker_type
        self.target_type = target_type
        self.tracker_model_path = tracker_model_path
        self.target_model_path = target_model_path
        self.episodes = episodes
        self.save_gif_freq = save_gif_freq
        self.output_dir = output_dir
        self.seed = seed
        self.state_space = str(state_space)
        self.specific_tracker_strategy = specific_tracker_strategy
        self.specific_target_strategy = specific_target_strategy
        self.main_output_dir = main_output_dir  # 用于存放所有子文件夹的主目录

        os.makedirs(output_dir, exist_ok=True)
        self.mission = 0
        self.run_dir = None
        self.run_timestamp = None


def run_battle_batch(args):
    """运行一批episode（减少模型重复加载）"""
    config, episode_indices = args
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    
    # 初始化策略管理器（不设置权重，直接使用指定策略）
    policy_manager = None
    if config.tracker_type == "random" or config.target_type == "random":
        policy_manager = PolicyManager()
    
    batch_results = []
    
    for episode_idx in episode_indices:
        try:
            result = run_single_episode(config, episode_idx, tracker_model, target_model, device, policy_manager)
            batch_results.append(result)
        except Exception as e:
            print(f"Error in episode {episode_idx}: {e}")
            batch_results.append({
                "episode_id": episode_idx,
                "steps": 0,
                "reward": 0.0,
                "tracker_caught_target": False,
                "target_reached_base": False,
                "tracker_type": config.tracker_type,
                "target_type": config.target_type,
                "tracker_strategy": config.specific_tracker_strategy or config.tracker_type,
                "target_strategy": config.specific_target_strategy or config.target_type
            })
    
    return batch_results


def run_single_episode(config, episode_idx, tracker_model, target_model, device, policy_manager=None):
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
        
        # 确定使用的策略（如果是random类型且指定了具体策略）
        tracker_strategy = config.specific_tracker_strategy or config.tracker_type
        target_strategy = config.specific_target_strategy or config.target_type
        
        if policy_manager:
            policy_manager.reset()
        
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
                    tracker_eval = tracker_model.evaluate(obs, tracker_hidden, greedy=True)
                    if isinstance(tracker_eval, tuple) and len(tracker_eval) >= 1:
                        tracker_action = tracker_eval[0]
                        if len(tracker_eval) > 1:
                            tracker_hidden = tracker_eval[1]
                    else:
                        tracker_action = (0.0, 1.0)
                elif config.tracker_type == "expert_rule":
                    tracker_action = get_expert_tracker_action_pair(obs)
                elif config.tracker_type == "random":
                    if policy_manager and config.specific_tracker_strategy:
                        tracker_action = policy_manager.get_action(config.specific_tracker_strategy, obs)
                    else:
                        tracker_action = get_expert_tracker_action_pair(obs)
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
                elif config.target_type == "random":
                    if policy_manager and config.specific_target_strategy:
                        target_action = policy_manager.get_action(config.specific_target_strategy, obs)
                    else:
                        target_action = get_expert_target_action_pair(obs)
                else:
                    target_action = -1

                obs, reward, terminated, truncated, info = env.step(tracker_action, target_action)
                done = terminated or truncated
                episode_reward += float(reward)

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
                    
                ep_name = f"ep_{episode_idx:04d}_{tracker_strategy}_vs_{target_strategy}_winner_{winner}.gif"
                gif_path = os.path.join(config.run_dir, ep_name)
                make_gif(episode_frames, gif_path)
            except Exception:
                # 移除GIF生成的错误打印
                pass
        
        return {
            "episode_id": episode_idx,
            "steps": episode_step,
            "reward": episode_reward,
            "tracker_caught_target": tracker_caught_target,
            "target_reached_base": target_reached_base,
            "tracker_type": config.tracker_type,
            "target_type": config.target_type,
            "tracker_strategy": tracker_strategy,
            "target_strategy": target_strategy
        }
    
    finally:
        env.close()


def analyze_strategy_performance(df):
    """分析不同策略组合的表现"""
    strategy_stats = defaultdict(lambda: {
        'episodes': 0,
        'tracker_wins': 0,
        'target_wins': 0,
        'draws': 0,
        'avg_steps': 0.0,
        'avg_reward': 0.0
    })
    
    for _, row in df.iterrows():
        key = f"{row['tracker_strategy']}_vs_{row['target_strategy']}"
        stats = strategy_stats[key]
        
        stats['episodes'] += 1
        if row['tracker_caught_target']:
            stats['tracker_wins'] += 1
        elif row['target_reached_base']:
            stats['target_wins'] += 1
        else:
            stats['draws'] += 1
        
        stats['avg_steps'] += row['steps']
        stats['avg_reward'] += row['reward']
    
    for key, stats in strategy_stats.items():
        if stats['episodes'] > 0:
            stats['avg_steps'] /= stats['episodes']
            stats['avg_reward'] /= stats['episodes']
            stats['tracker_win_rate'] = stats['tracker_wins'] / stats['episodes']
            stats['target_win_rate'] = stats['target_wins'] / stats['episodes']
            stats['draw_rate'] = stats['draws'] / stats['episodes']
    
    return dict(strategy_stats)


def run_strategy_evaluation(base_config):
    """运行策略评估：对每种random策略分别进行100场对战"""
    # 创建主输出目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    main_evaluation_dir = os.path.join(base_config.output_dir, f"evaluation_{base_config.tracker_type}_vs_{base_config.target_type}_{timestamp}")
    os.makedirs(main_evaluation_dir, exist_ok=True)
    
    all_results = []
    all_summaries = []
    
    # 确定需要评估的策略组合
    tracker_strategies = []
    target_strategies = []
    
    if base_config.tracker_type == "random":
        tracker_strategies = get_available_policies("tracker")
    else:
        tracker_strategies = [base_config.tracker_type]
    
    if base_config.target_type == "random":
        target_strategies = get_available_policies("target")
    else:
        target_strategies = [base_config.target_type]
    
    total_combinations = len(tracker_strategies) * len(target_strategies)
    print(f"将评估 {total_combinations} 种策略组合，每种组合 {base_config.episodes} 场对战")
    print(f"所有结果将保存在: {main_evaluation_dir}")
    
    combination_count = 0
    for tracker_strategy in tracker_strategies:
        for target_strategy in target_strategies:
            combination_count += 1
            print(f"\n进度 [{combination_count}/{total_combinations}] 评估策略组合: {tracker_strategy} vs {target_strategy}")
            
            # 创建专门的配置，将子文件夹放在主目录下
            strategy_output_dir = os.path.join(main_evaluation_dir, "individual_battles")
            os.makedirs(strategy_output_dir, exist_ok=True)
            
            config = BattleConfig(
                tracker_type=base_config.tracker_type,
                target_type=base_config.target_type,
                tracker_model_path=base_config.tracker_model_path,
                target_model_path=base_config.target_model_path,
                episodes=base_config.episodes,
                save_gif_freq=base_config.save_gif_freq,
                output_dir=strategy_output_dir,  # 子文件夹放在主目录下
                seed=base_config.seed + combination_count,
                state_space=base_config.state_space,
                specific_tracker_strategy=tracker_strategy if base_config.tracker_type == "random" else None,
                specific_target_strategy=target_strategy if base_config.target_type == "random" else None,
                main_output_dir=main_evaluation_dir  # 设置主输出目录
            )
            
            # 运行单个策略组合的对战
            results, run_dir = run_battle(config, strategy_name=f"{tracker_strategy}_vs_{target_strategy}")
            
            if results is not None:
                all_results.extend(results.to_dict('records'))
                
                # 添加汇总信息
                summary = {
                    'tracker_strategy': tracker_strategy,
                    'target_strategy': target_strategy,
                    'episodes': len(results),
                    'tracker_win_rate': results['tracker_caught_target'].mean(),
                    'target_win_rate': results['target_reached_base'].mean(),
                    'draw_rate': 1.0 - results['tracker_caught_target'].mean() - results['target_reached_base'].mean(),
                    'avg_steps': results['steps'].mean(),
                    'avg_reward': results['reward'].mean()
                }
                all_summaries.append(summary)
    
    # 保存综合结果到主目录
    if all_results:
        # 保存所有详细结果
        all_results_df = pd.DataFrame(all_results)
        all_results_df.to_csv(os.path.join(main_evaluation_dir, "all_results.csv"), index=False)
        
        # 保存汇总结果
        summary_df = pd.DataFrame(all_summaries)
        summary_df.to_csv(os.path.join(main_evaluation_dir, "strategy_summary.csv"), index=False)
        
        # 保存评估配置信息
        config_info = {
            'tracker_type': base_config.tracker_type,
            'target_type': base_config.target_type,
            'episodes_per_strategy': base_config.episodes,
            'total_combinations': total_combinations,
            'total_episodes': len(all_results),
            'evaluation_time': timestamp,
            'tracker_model_path': base_config.tracker_model_path,
            'target_model_path': base_config.target_model_path
        }
        
        with open(os.path.join(main_evaluation_dir, "evaluation_config.json"), 'w') as f:
            json.dump(config_info, f, indent=2)
        
        # 打印最终汇总
        print(f"\n=== 综合评估结果 ===")
        print(f"总共评估了 {total_combinations} 种策略组合")
        print(f"总计 {len(all_results)} 场对战")
        print(f"结果保存在: {main_evaluation_dir}")
        print("\n策略组合表现排名 (按Tracker胜率排序):")
        print(f"{'Tracker策略':<20} {'Target策略':<20} {'场次':<6} {'Tracker胜率':<12} {'Target胜率':<12} {'平均步数':<10}")
        print("-" * 90)
        
        summary_df_sorted = summary_df.sort_values('tracker_win_rate', ascending=False)
        for _, row in summary_df_sorted.iterrows():
            print(f"{row['tracker_strategy']:<20} {row['target_strategy']:<20} {row['episodes']:<6.0f} "
                  f"{row['tracker_win_rate']*100:<11.1f}% {row['target_win_rate']*100:<11.1f}% {row['avg_steps']:<10.1f}")
        
        return all_results_df, main_evaluation_dir
    
    return None, None


def run_battle(config, strategy_name=None):
    """运行完整的对战测试"""
    if strategy_name:
        print(f"运行对战: {strategy_name}, {config.episodes} 场")
    else:
        print(f"运行对战: {config.tracker_type} vs {config.target_type}, {config.episodes} 场")

    # 创建运行目录
    if strategy_name:
        run_name = f"battle_{strategy_name}"
    else:
        run_name = f"battle_{config.tracker_type}_vs_{config.target_type}"
    
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
    
    try:
        results_path = os.path.join(config.run_dir, "results.csv")
        df.to_csv(results_path, index=False)
        
        stats = {
            "total_episodes": len(df),
            "avg_steps": avg_steps,
            "avg_reward": avg_reward,
            "tracker_win_rate": tracker_win_rate,
            "target_win_rate": target_win_rate,
            "draw_rate": 1.0 - tracker_win_rate - target_win_rate,
            "tracker_strategy": config.specific_tracker_strategy or config.tracker_type,
            "target_strategy": config.specific_target_strategy or config.target_type
        }
        stats_path = os.path.join(config.run_dir, "stats.csv")
        pd.DataFrame([stats]).to_csv(stats_path, index=False)
        
    except Exception as e:
        print(f"Warning: Failed to save results: {e}")
    
    total_time = time.time() - start_time
    print(f"结果: 场次={len(df)}, 平均步数={avg_steps:.1f}, Tracker胜率={tracker_win_rate*100:.1f}%, Target胜率={target_win_rate*100:.1f}%, 用时={total_time:.1f}s")
    
    return df, config.run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Agent Battle Evaluation')
    parser.add_argument('--tracker', type=str, default='policy', choices=['rule', 'expert_rule', 'policy', 'random'])
    parser.add_argument('--target', type=str, default='random', choices=['rule', 'expert_rule', 'policy', 'random'])
    parser.add_argument('--tracker_model', type=str, default='./models/TrackingEnv/DualAgent10-09-251052/best_model/tracker_net_checkpoint.pkl')
    parser.add_argument('--target_model', type=str, default=None)
    parser.add_argument('--episodes', type=int, default=100)
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
    
    # 如果使用random对手，则进行全面评估；否则只进行单次对战
    if config.tracker_type == "random" or config.target_type == "random":
        run_strategy_evaluation(config)
    else:
        # 单次对战也统一放在时间戳文件夹中
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        main_dir = os.path.join(config.output_dir, f"single_battle_{config.tracker_type}_vs_{config.target_type}_{timestamp}")
        os.makedirs(main_dir, exist_ok=True)
        config.output_dir = main_dir
        run_battle(config)