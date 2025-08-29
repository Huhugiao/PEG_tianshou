import os

# 画布尺寸
width = 600
height = 600

# 基础单位与速度（浮点）
pixel_size = 4
target_speed = 8
tracker_speed = 6.0

# 判定半径
capture_radius = 10
base_radius = 12

# 质量/速度开关（环境变量：fast 或 quality）
FAST = os.getenv('SCRIMP_RENDER_MODE', 'fast').lower() == 'fast'

# 渲染与美观设置
background_color = (245, 247, 250)
grid_color = (220, 225, 232)
grid_step = 50

# 轨迹与实体外观
trail_color_tracker = (80, 120, 255, 160)   # RGBA
trail_color_target  = (255, 100, 100, 140)  # RGBA
trail_max_len = 60 if FAST else 600
trail_width = 1 if FAST else 2

base_color_inner = (70, 160, 255)
base_color_outer = (30, 110, 220)
tracker_color = (40, 90, 255)
target_color = (230, 60, 60)
heading_color = (255, 255, 255)

# 形状尺寸
agent_radius = 7
base_radius_draw = 12

# 抗锯齿与超级采样（fast 关闭，quality 打开）
ssaa = 1 if FAST else 2
enable_aa = False if FAST else True
draw_grid = False if FAST else True

# 训练相关
test_flag = False
mask_flag = False
collision_penalty = -20
loss_penalty = -20
success_reward = 20

max_detection_distance = 300
best_distance = 20
max_detection_angle = 90
best_angle = 0

max_loss_step = 50
total_steps = 500

def set_render_quality(mode: str):
    """运行时切换渲染质量: 'fast' 或 'quality'"""
    global FAST, ssaa, enable_aa, draw_grid, trail_max_len, trail_width
    FAST = (mode.lower() == 'fast')
    ssaa = 1 if FAST else 2
    enable_aa = False if FAST else True
    draw_grid = False if FAST else True
    trail_max_len = 60 if FAST else 600
    trail_width = 1 if FAST else 2