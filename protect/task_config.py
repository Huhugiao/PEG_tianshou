import pygame

# 定义窗口的尺寸
width = 500   # 地图的宽度，单位像素
height = 500  # 地图的高度，单位像素

# 像素大小、移动步长、障碍物尺寸
pixel_size = 10  # 每个像素代表的实际距离单位
target_speed = 6.5  
tracker_speed = 5 
min_obstacle_size = 4  # 最小障碍物尺寸
max_obstacle_size = 6  # 最大障碍物尺寸

# 掩码标志和惩罚值
test_flag = False
mask_flag = False  # 是否使用掩码，布尔值
collision_penalty = -20  # 碰撞时的惩罚分数
loss_penalty = -20  # 丢失目标时的惩罚分数
success_reward = 20

# 检测距离和角度的最大值
max_detection_distance = 300  # 最大检测距离
best_distance = 20  # 最佳检测距离
max_detection_angle = 90  # 最大检测角度
best_angle = 0  # 最佳检测角度

# 过程中的一些限制
max_loss_step = 50  # 最大丢失步数
total_steps = 500  # 总步数限制

# 加载图像资源
tracker_img = pygame.image.load("./robot.png") 
target_img = pygame.image.load("./running_kid.png") 
base_img = pygame.image.load("./blue_base.png")

# 障碍物种类的概率分布
obstacle_kind_probability = [0.4, 0.1, 0.1, 0.2, 0.2]  # 不同类型障碍物出现的概率
