import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from scipy.ndimage import gaussian_filter1d
import algo_config  # 确保 algo_config 中包含 logdir 变量

def load_scalars_from_file(event_file: str, tag: str):
    """
    从一个 event 文件里加载某个 scalar tag 的 (step, value) 列表
    """
    ea = event_accumulator.EventAccumulator(
        event_file,
        size_guidance={event_accumulator.SCALARS: 0}  # 不限数量
    )
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        raise ValueError(f"Tag {tag!r} not found in {event_file}")
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    vals  = [e.value for e in events]
    return steps, vals

def load_scalars_from_dir(event_dir: str, tag: str):
    """
    遍历一个目录下所有的 event 文件（文件名包含“tfevents”），
    每个文件返回一对 (steps, vals)
    """
    series = []
    for file in sorted(os.listdir(event_dir)):
        if "tfevents" in file:
            event_file = os.path.join(event_dir, file)
            try:
                steps, vals = load_scalars_from_file(event_file, tag)
                series.append((steps, vals))
            except Exception as e:
                print(f"忽略文件 {event_file}：{e}")
    return series

def plot_comparison(tracker_dir: str,
                    target_dir: str,
                    tag: str,
                    label1: str = "tracker",
                    label2: str = "target",
                    save_path: str = None):
    # 读数据：每个目录可能包含多个 event 文件
    tracker_series = load_scalars_from_dir(tracker_dir, tag)
    target_series  = load_scalars_from_dir(target_dir, tag)

    plt.figure(figsize=(8,5))
    
    # 固定颜色：同一文件夹内所有曲线采用相同颜色
    tracker_color = "blue"
    target_color = "red"
    
    # 绘制 tracker 目录下的每个 event 文件
    for i, (s, v) in enumerate(tracker_series):
        current_label = label1 if i == 0 else None
        # 原始曲线，设置透明度以实现淡化
        plt.plot(s, v, '-', color=tracker_color, alpha=0.3, linewidth=2, label=current_label)
        # 计算平滑曲线
        smooth_v = gaussian_filter1d(np.array(v), sigma=5)
        plt.plot(s, smooth_v, '-', color=tracker_color, linewidth=2)
    
    # 绘制 target 目录下的每个 event 文件
    for i, (s, v) in enumerate(target_series):
        current_label = label2 if i == 0 else None
        plt.plot(s, v, '-', color=target_color, alpha=0.3, linewidth=2, label=current_label)
        smooth_v = gaussian_filter1d(np.array(v), sigma=5)
        plt.plot(s, smooth_v, '-', color=target_color, linewidth=2)

    plt.xlabel("step")
    plt.ylabel(tag)
    plt.title(f"{tag} comparison")
    plt.legend()
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
    plt.show()

if __name__ == "__main__":
    # 根据 algo_config.logdir 构造实验内的日志路径
    tracker_events = os.path.join(algo_config.logdir, "events")
    target_events  = os.path.join(algo_config.logdir, "events", "others")
    tag = "test/reward"   # 可根据需要调整 scalar 标签

    save_path = os.path.join(algo_config.logdir, "compare_reward.png")
    plot_comparison(tracker_events, target_events, tag,
                    label1="tracker policy", label2="target policy",
                    save_path=save_path)