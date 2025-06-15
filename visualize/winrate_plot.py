import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import io  # 导入io模块替代pandas.compat.StringIO

# 构建文件路径
current_dir = os.path.dirname(__file__)
rank_path = os.path.join(
    current_dir, '..',
    'tblogs/protect_vs_invade_adjusting_43',
    'agent_rankings.csv'
)

# 获取文件的绝对路径并打印（用于调试）
absolute_path = os.path.abspath(rank_path)

# 读取CSV文件内容
try:
    with open(absolute_path, 'r') as f:
        content = f.read()
except FileNotFoundError:
    print(f"Error: File not found at {absolute_path}")
    exit(1)
except Exception as e:
    print(f"Error reading file: {e}")
    exit(1)

# 分割为Tracker和Target两个部分
tracker_content = re.search(r'=== Tracker Rankings ===(.*?)=== Target Rankings ===', content, re.DOTALL).group(1).strip()
target_content = re.search(r'=== Target Rankings ===(.*)', content, re.DOTALL).group(1).strip()

# 将字符串转换为DataFrame - 使用io.StringIO替代pandas.compat.StringIO
tracker_df = pd.read_csv(io.StringIO(tracker_content))
target_df = pd.read_csv(io.StringIO(target_content))

# 从agent名称中提取阶段编号
tracker_df['stage'] = tracker_df['agent'].str.extract(r'(\d+)').astype(int)
target_df['stage'] = target_df['agent'].str.extract(r'(\d+)').astype(int)

# 按阶段编号排序
tracker_df = tracker_df.sort_values('stage')
target_df = target_df.sort_values('stage')

# 创建图表
plt.figure(figsize=(12, 6))

# 绘制Tracker曲线（蓝色）
plt.plot(tracker_df['stage'], tracker_df['overall_win_rate'], 
         'b-', label='Tracker', marker='o', markersize=5, linewidth=2)

# 绘制Target曲线（红色）
plt.plot(target_df['stage'], target_df['overall_win_rate'], 
         'r-', label='Target', marker='s', markersize=5, linewidth=2)

# 设置图表标题和标签
plt.title('Agent Win Rate by Training Stage', fontsize=14)
plt.xlabel('Training Stage', fontsize=12)
plt.ylabel('Overall Win Rate', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# 设置横轴范围
min_stage = min(min(tracker_df['stage']), min(target_df['stage']))
max_stage = max(max(tracker_df['stage']), max(target_df['stage']))
plt.xlim(min_stage - 1, max_stage + 1)

# 添加图例
plt.legend(fontsize=12)

# 优化布局
plt.tight_layout()

# 保存并显示图表
output_dir = os.path.dirname(absolute_path)
output_path = os.path.join(output_dir, 'agent_win_rates.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Chart saved to: {output_path}")
plt.show()