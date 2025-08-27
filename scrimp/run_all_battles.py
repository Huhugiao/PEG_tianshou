import os
import argparse
from evaluate_battle import BattleConfig, run_battle
import pandas as pd
from datetime import datetime

def run_all_combinations(args):
    """运行所有组合的对战评估"""
    results = []
    
    # 确保输出目录存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.gif_dir):
        os.makedirs(args.gif_dir)
    
    # 1. 规则 vs 规则
    print("\n===== 运行对战: 规则 Tracker vs 规则 Target =====")
    config = BattleConfig(
        tracker_type="rule",
        target_type="rule",
        episodes=args.episodes,
        save_gif_freq=args.save_gif_freq,
        output_dir=args.output_dir,
        gif_dir=args.gif_dir,
        seed=args.seed
    )
    stats, _ = run_battle(config)
    results.append(stats)
    
    # 2. 训练好的 Tracker vs 规则 Target
    if args.tracker_model:
        print("\n===== 运行对战: 训练好的 Tracker vs 规则 Target =====")
        config = BattleConfig(
            tracker_type="policy",
            target_type="rule",
            tracker_model_path=args.tracker_model,
            episodes=args.episodes,
            save_gif_freq=args.save_gif_freq,
            output_dir=args.output_dir,
            gif_dir=args.gif_dir,
            seed=args.seed
        )
        stats, _ = run_battle(config)
        results.append(stats)
    
    # 3. 规则 Tracker vs 训练好的 Target
    if args.target_model:
        print("\n===== 运行对战: 规则 Tracker vs 训练好的 Target =====")
        config = BattleConfig(
            tracker_type="rule",
            target_type="policy",
            target_model_path=args.target_model,
            episodes=args.episodes,
            save_gif_freq=args.save_gif_freq,
            output_dir=args.output_dir,
            gif_dir=args.gif_dir,
            seed=args.seed
        )
        stats, _ = run_battle(config)
        results.append(stats)
    
    # 4. 训练好的 Tracker vs 训练好的 Target
    if args.tracker_model and args.target_model:
        print("\n===== 运行对战: 训练好的 Tracker vs 训练好的 Target =====")
        config = BattleConfig(
            tracker_type="policy",
            target_type="policy",
            tracker_model_path=args.tracker_model,
            target_model_path=args.target_model,
            episodes=args.episodes,
            save_gif_freq=args.save_gif_freq,
            output_dir=args.output_dir,
            gif_dir=args.gif_dir,
            seed=args.seed
        )
        stats, _ = run_battle(config)
        results.append(stats)
    
    # 保存所有结果的汇总
    df_summary = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = f"{args.output_dir}/summary_all_battles_{timestamp}.csv"
    df_summary.to_csv(summary_path, index=False)
    
    print(f"\n所有对战评估完成！汇总结果保存在: {summary_path}")
    
    # 打印汇总表格
    print("\n对战结果汇总:")
    print(df_summary[["tracker_type", "target_type", "tracker_win_rate", "target_win_rate", "draw_rate", "avg_reward", "avg_steps"]].to_string(index=False))
    
    return df_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run all battle combinations')
    parser.add_argument('--tracker_model', type=str, default=None,
                        help='Path to trained tracker model checkpoint')
    parser.add_argument('--target_model', type=str, default=None,
                        help='Path to trained target model checkpoint')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of battle episodes per combination')
    parser.add_argument('--save_gif_freq', type=int, default=10,
                        help='Save GIF every N episodes (0 to disable)')
    parser.add_argument('--output_dir', type=str, default='./battle_results',
                        help='Directory to save battle results')
    parser.add_argument('--gif_dir', type=str, default='./battle_gifs',
                        help='Directory to save battle GIFs')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed')
                        
    args = parser.parse_args()
    
    run_all_combinations(args)