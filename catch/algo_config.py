import torch

task = "Catching-v0"  # 环境名称，指定使用的环境类型
reward_threshold = 500000 # 奖励阈值，用于评估算法性能，None表示没有预设阈值
cl_flag = 1 # 用于某些特定任务的标志，可能控制某个任务或训练的特殊参数
training_stage = 1  # 当前训练阶段，通常用于区分不同的训练阶段（例如，预训练、微调等）
seed = 1  # 随机种子，用于确保实验可重复性，保证每次运行时结果一致
eps_test = 0.05  # 测试阶段的epsilon值，使用 epsilon-greedy 策略时的探索率
eps_train = 0.1  # 训练阶段的epsilon值，使用 epsilon-greedy 策略时的探索率
buffer_size = 5e5  # 经验回放缓冲区的大小，存储代理的历史经验，用于训练时的采样
lr = 1e-4  # 学习率，控制模型更新的步幅，学习率过大会导致训练不稳定，过小则收敛过慢
gamma = 0.95  # 折扣因子，用于计算未来奖励的当前价值，值越大代表更看重长期回报
num_atoms = 64  # 分布式价值函数中原子的数量，C51等算法中将Q值分为多个原子（原子数为64）
v_min = -10.0  # 价值函数的最小值，分布式Q学习中Q值的最小边界
v_max = 10.0  # 价值函数的最大值，分布式Q学习中Q值的最大边界
noisy_std = 0.1  # 噪声标准差，用于在训练时添加噪声以打破对称性，有助于探索
n_step = 3  # n步回报，考虑多个未来步骤的回报，增强短期奖励的考虑
target_update_freq = 100  # 目标网络更新频率，控制每多少步更新目标网络
epoch = 200  # 训练周期总数，表示模型训练的总迭代次数
step_per_epoch = 10000  # 每个训练周期内的步数，表示每个周期中进行的环境交互步数
step_per_collect = 256  # 每收集一次数据的步数，指定每次数据收集时的环境交互步数
update_per_step = 0.05  # 每步更新的频率，表示每多少步进行一次训练更新
batch_size = 256  # 批处理大小，表示每次训练时从经验回放中采样的数据量
hidden_sizes = [128, 128]  # 神经网络的隐含层大小，表示每一层的神经元数量
training_num = 16  # 训练环境数量，表示并行训练时的环境数目
test_num = 8  # 测试环境数量，表示并行测试时的环境数目
render = 0.1  # 渲染频率，用于可视化训练过程，指定每多少步进行一次环境渲染
prioritized_replay = False  # 是否使用优先级经验回放，用于增强训练过程中重要经验的采样
alpha = 0.6  # 优先级经验回放中的alpha参数，用于计算重要性采样权重
beta = 0.4  # 优先级经验回放中的beta参数，用于初始的重要性采样
beta_final = 1.0  # 优先级经验回放中的beta参数的最终值，表示训练过程中逐渐增加beta的值
device = "cuda" if torch.cuda.is_available() else "cpu"  # 使用的设备，优先使用GPU，如果没有则使用CPU
save_interval = 2  # 保存模型的间隔，每隔多少步或周期保存一次模型
repeat_per_collect = 16  # 每次数据收集后，重复训练的次数，控制训练更新频率

# 算法参数-PPO
vf_coef = 0.25  # 值函数损失的系数，用于平衡策略损失和值函数损失
ent_coef = 0.03  # 熵损失的系数，增加策略的随机性，防止过早收敛
eps_clip = 0.2  # clip epsilon，PPO 中用于限制策略更新幅度的超参数
max_grad_norm = 0.5  # 梯度裁剪的最大值，用于防止梯度爆炸
gae_lambda = 0.95  # 广义优势估计中的lambda参数，平衡偏差和方差
dual_clip = None  # 双重裁剪的阈值，控制Q值更新的范围，默认无使用
value_clip = 1  # 值函数的裁剪参数，控制值函数更新的范围
norm_adv = 1  # 是否对优势进行归一化，默认是对优势进行标准化
recompute_adv = 0  # 是否在每次更新时重新计算优势，0表示不重新计算
reward_normalization = True  # 是否对奖励进行规范化，通常用于加速训练并提高稳定性


# 特权学习参数
use_god_view = False
god_view_shape = (32,)

watch_agent = False
resume = False

using_tensorboard = True
logdir = "performance_test_0317"

wb_name = "his_and_future_lstm"
run_id = "10"
wb_project = "catching"
