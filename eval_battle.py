import os
import csv
import algo_config

logdir = algo_config.logdir
input_file = os.path.join(logdir, "head_to_head_results.csv")
output_file = os.path.join(logdir, "agent_rankings.csv")

# 用于存放 tracker 角色的统计数据
# 格式：tracker_stats[tracker_agent] = {"wins": 总胜场, "matches": 总局数, "vs": {target_agent: {"wins": 对该对手获胜数, "matches": 对局局数}}}
tracker_stats = {}
# 用于存放 target 角色的统计数据
# 格式：target_stats[target_agent] = {"wins": 总胜场, "matches": 总局数, "vs": {tracker_agent: {"wins": 对该对手获胜数, "matches": 对局局数}}}
target_stats = {}

with open(input_file, mode="r", newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        tracker = row["tracker_model"]
        target = row["target_model"]
        tracker_wins = int(row["tracker_wins"])
        target_wins = int(row["target_wins"])
        draws = int(row["draws"])
        total_games = tracker_wins + target_wins + draws

        # 更新 tracker 数据
        if tracker not in tracker_stats:
            tracker_stats[tracker] = {"wins": 0, "matches": 0, "vs": {}}
        tracker_stats[tracker]["wins"] += tracker_wins
        tracker_stats[tracker]["matches"] += total_games
        if target not in tracker_stats[tracker]["vs"]:
            tracker_stats[tracker]["vs"][target] = {"wins": 0, "matches": 0}
        tracker_stats[tracker]["vs"][target]["wins"] += tracker_wins
        tracker_stats[tracker]["vs"][target]["matches"] += total_games

        # 更新 target 数据
        if target not in target_stats:
            target_stats[target] = {"wins": 0, "matches": 0, "vs": {}}
        target_stats[target]["wins"] += target_wins
        target_stats[target]["matches"] += total_games
        if tracker not in target_stats[target]["vs"]:
            target_stats[target]["vs"][tracker] = {"wins": 0, "matches": 0}
        target_stats[target]["vs"][tracker]["wins"] += target_wins
        target_stats[target]["vs"][tracker]["matches"] += total_games

def compute_metrics(stats_dict):
    """
    针对每个智能体，计算总体胜率、各个对手胜率的平均值和最差的对手胜率
    返回含有指标的列表
    """
    ranking = []
    for agent, rec in stats_dict.items():
        overall_win_rate = rec["wins"] / rec["matches"] if rec["matches"] > 0 else 0
        opponent_rates = []
        for opp, opp_rec in rec["vs"].items():
            rate = opp_rec["wins"] / opp_rec["matches"] if opp_rec["matches"] > 0 else 0
            opponent_rates.append(rate)
        if opponent_rates:
            avg_opp_win_rate = sum(opponent_rates) / len(opponent_rates)
            worst_opp_win_rate = min(opponent_rates)
        else:
            avg_opp_win_rate = 0
            worst_opp_win_rate = 0
        ranking.append({
            "agent": agent,
            "wins": rec["wins"],
            "matches": rec["matches"],
            "overall_win_rate": overall_win_rate,
            "average_opponent_win_rate": avg_opp_win_rate,
            "worst_opponent_win_rate": worst_opp_win_rate
        })
    return ranking

# 分别计算 tracker 与 target 的评估指标
tracker_ranking = compute_metrics(tracker_stats)
target_ranking  = compute_metrics(target_stats)

# 按照总体胜率降序排序
tracker_ranking.sort(key=lambda x: x["overall_win_rate"], reverse=True)
target_ranking.sort(key=lambda x: x["overall_win_rate"], reverse=True)

# 将两部分结果写入同一个 CSV 文件中，分别作为不同的部分
with open(output_file, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    
    # 写入 tracker 部分
    writer.writerow(["=== Tracker Rankings ==="])
    writer.writerow(["agent", "wins", "matches", "overall_win_rate", "average_opponent_win_rate", "worst_opponent_win_rate"])
    for record in tracker_ranking:
        writer.writerow([
            record["agent"],
            record["wins"],
            record["matches"],
            record["overall_win_rate"],
            record["average_opponent_win_rate"],
            record["worst_opponent_win_rate"]
        ])
        
    writer.writerow([])  # 空行分隔

    # 写入 target 部分
    writer.writerow(["=== Target Rankings ==="])
    writer.writerow(["agent", "wins", "matches", "overall_win_rate", "average_opponent_win_rate", "worst_opponent_win_rate"])
    for record in target_ranking:
        writer.writerow([
            record["agent"],
            record["wins"],
            record["matches"],
            record["overall_win_rate"],
            record["average_opponent_win_rate"],
            record["worst_opponent_win_rate"]
        ])

print(f"Combined ranking results saved to: {output_file}")