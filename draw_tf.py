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

def downsample_data(x, y, max_points=1000):
    """
    如果数据量超过 max_points，则下采样，否则直接返回原数据
    """
    if len(x) <= max_points:
        return x, y
    indices = np.linspace(0, len(x)-1, max_points, dtype=int)
    return [x[i] for i in indices], [y[i] for i in indices]

def plot_comparison(tracker_dir: str,
                    target_dir: str,
                    tag: str,
                    label1: str = "tracker policy",
                    label2: str = "target policy",
                    save_path: str = None,
                    max_points: int = 1000):
    # 读数据：每个目录可能包含多个 event 文件
    tracker_series = load_scalars_from_dir(tracker_dir, tag)
    target_series  = load_scalars_from_dir(target_dir, tag)

    # 构造交替排列的列表：
    # tracker 的事件顺序为 0,2,4,...; target 的事件顺序为 1,3,5,...
    combined = []
    for i, series in enumerate(tracker_series):
        combined.append(("tracker", 2 * i, series))
    for i, series in enumerate(target_series):
        combined.append(("target", 2 * i + 1, series))
    combined.sort(key=lambda x: x[1])

    plt.figure(figsize=(15,5))
    
    tracker_color = "blue"
    target_color = "red"
    tracker_label_shown = False
    target_label_shown = False

    for agent, order, (s, v) in combined:
        # 使用一个独立区间绘制该 tfevents 曲线：区间为 [order, order+1]
        # 重新生成横坐标，不再使用原来的 step 数据，确保不同 tfevents 曲线横坐标不会重合
        x = np.linspace(order, order+1, len(s))
        # 下采样处理
        down_x, down_v = downsample_data(x.tolist(), v, max_points)
        if agent == "tracker":
            current_label = label1 if not tracker_label_shown else None
            tracker_label_shown = True
            color = tracker_color
        else:
            current_label = label2 if not target_label_shown else None
            target_label_shown = True
            color = target_color
        
        # 绘制原始数据曲线（淡化样式）
        plt.plot(down_x, down_v, '-', color=color, alpha=0.3, linewidth=2, label=current_label)
        
        # 计算平滑后的数据和下采样
        smooth_v = gaussian_filter1d(np.array(v), sigma=5)
        down_x_smooth, down_smooth_v = downsample_data(x.tolist(), smooth_v.tolist(), max_points)
        plt.plot(down_x_smooth, down_smooth_v, '-', color=color, linewidth=2)

    plt.xlabel("Epochs")
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