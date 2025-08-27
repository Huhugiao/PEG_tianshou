import os
import csv
import gym
import torch
import re
import argparse
from tianshou.env import SubprocVectorEnv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from policy import policy_maker
from vcollector import VCollector

# 自定义函数，用于以数值排序文件名
def numerical_key(f):
    m = re.search(r'\d+', f)
    return int(m.group()) if m else -1

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run battle evaluation between tracker and target agents")
    parser.add_argument("--logdir", type=str, required=True, help="Directory containing agent models")
    parser.add_argument("--task", type=str, default="Protecting-v0", help="Environment name")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run models on")
    args = parser.parse_args()

    # 设置为双智能体模式 (mission=2)
    mission = 2

    # agents 存放 tracker 和 target 模型
    agents_dir = os.path.join(args.logdir, "agents")
    if not os.path.exists(agents_dir):
        raise FileNotFoundError(f"Agents directory not found: {agents_dir}")

    # 获取所有 tracker 与 target 模型文件，并使用自然排序
    tracker_files = sorted(
        [f for f in os.listdir(agents_dir) if f.startswith("tracker") and f.endswith(".pth")],
        key=numerical_key
    )
    target_files = sorted(
        [f for f in os.listdir(agents_dir) if f.startswith("target") and f.endswith(".pth")],
        key=numerical_key
    )

    # 筛选：tracker 只保留尾号偶数的文件，target 只保留尾号奇数的文件
    tracker_files = [f for f in tracker_files if int(re.search(r'\d+', f).group()) % 2 == 0]
    target_files = [f for f in target_files if int(re.search(r'\d+', f).group()) % 2 == 1]

    # 每对对战运行的总回合数 & 每个并行环境的数量
    num_episodes = args.episodes
    num_envs = min(20, num_episodes)  # 根据机器性能和内存限制，设置并行环境数量

    # CSV 文件路径（每对对战结束后写入，覆盖之前的结果）
    output_file = os.path.join(args.logdir, "head_to_head_results.csv")
    with open(output_file, mode="w", newline="") as csvfile:
        fieldnames = ["tracker_model", "target_model", "tracker_wins", "target_wins", "draws"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 外层遍历 tracker，内层遍历 target
        for t_file in tracker_files:
            for tt_file in target_files:
                tracker_win = 0
                target_win = 0
                draws = 0
                print(f"\n===== Evaluating: {t_file} VS {tt_file} =====")
                
                # 构建 num_envs 个并行评估环境
                eval_envs = SubprocVectorEnv([
                    lambda: gym.make(
                        args.task,
                        mission=mission,
                        base_obs_dim=12,  # Default values
                        use_god_view=False,
                        god_view_dim=4
                    ) for _ in range(num_envs)
                ])
                eval_envs.reset(seed=args.seed)
                
                # 构建策略实例，并加载对应模型权重（注意保持所有并行环境共用同一策略）
                policy, _, _ = policy_maker()
                tracker_path = os.path.join(agents_dir, t_file)
                target_path  = os.path.join(agents_dir, tt_file)
                policy.policy_a.load_state_dict(torch.load(tracker_path, map_location=args.device))
                policy.policy_b.load_state_dict(torch.load(target_path, map_location=args.device))
                policy.eval()
                
                # 创建 VCollector
                collector = VCollector(policy, eval_envs)
                
                # 为了完全复用 Collector.collect 的逻辑，每个采集批次前主动重置环境和缓冲区
                collected_eps = 0
                while collected_eps < num_episodes:
                    collector.reset_env()
                    collector.reset_buffer()
                    result = collector.collect(n_episode=num_envs)
                    episode_infos = result.get("episode_infos", [])
                    for info in episode_infos:
                        # terminated==True 表示目标达到基地 => target 获胜
                        if info.get("terminated", False):
                            target_win += 1
                        # truncated==True 表示追踪者拦截目标 => tracker 获胜
                        elif info.get("truncated", False):
                            tracker_win += 1
                        else:
                            draws += 1
                    collected_eps += result["n/ep"]
                eval_envs.close()
        
                # 每对智能体对战完后，将结果写入 CSV 文件
                row = {
                    "tracker_model": t_file,
                    "target_model": tt_file,
                    "tracker_wins": tracker_win,
                    "target_wins": target_win,
                    "draws": draws
                }
                writer.writerow(row)
                csvfile.flush()  # 立即写入磁盘
                print(f"Results for {t_file} vs {tt_file}: tracker wins {tracker_win}, target wins {target_win}, draws {draws}")

    print(f"\nHead-to-head evaluation results saved to: {output_file}")

if __name__ == "__main__":
    main()