import os, torch

# ================================
# General Settings
# ================================
mission = 0  # 0: train tracker (target uses rules); 1: train target (tracker uses policy); 2: train tracker (target uses policy)
task = "Protecting-v0"  # Environment name
seed = 1  # Random seed for reproducibility
device = "cuda" if torch.cuda.is_available() else "cpu"

# ================================
# Training Hyperparameters
# ================================
epoch = 250                     # Total training epochs
step_per_epoch = 30000          # Steps per epoch
batch_size = 256                # Batch size for training
update_per_step = 0.05          # Updates per environment step
repeat_per_collect = 16         # Training repeats for each data collection
save_interval = 1               # Interval for logging/saving checkpoints

# ================================
# Environment & Replay Buffer
# ================================
training_num = 16             # Number of parallel training environments
buffer_size = int(5e5)        # Replay buffer capacity

# Prioritized Replay settings (if used)
prioritized_replay = False
alpha = 0.6                   # Priority exponent
beta = 0.4                    # Initial importance-sampling exponent

# ================================
# Network & PPO Hyperparameters
# ================================
lr = 1e-4                     # Learning rate
gamma = 0.99                  # Discount factor for future rewards
hidden_sizes = [144, 144]     # Hidden layer sizes

# PPO-specific parameters
vf_coef = 0.25                # Value function loss coefficient
ent_coef = 0.01               # Entropy loss coefficient
eps_clip = 0.2                # PPO clip epsilon
max_grad_norm = 0.5           # Maximum gradient norm for clipping
gae_lambda = 0.95             # Lambda for Generalized Advantage Estimation
dual_clip = None              # Dual clipping threshold (if applicable)
value_clip = 1                # Clipping parameter for the value function
norm_adv = 1                  # Normalize advantages (1: yes)
recompute_adv = 0             # Recompute advantages per update (0: no)
reward_normalization = True   # Flag indicating reward normalization

# ================================
# Privileged Observations & LSTM Setup
# ================================
base_obs_dim = 12            # Base observation dimension
use_god_view = False         # Whether to use additional (god view) observation
god_view_dim = 4             # Dimension for privileged observations

# ================================
# Checkpoint & Logging
# ================================
watch_agent = 1             # If enabled, load agent for demo/test instead of training
resume = False              # Resume training from a checkpoint
using_tensorboard = True    # Log with Tensorboard (else Wandb will be used)

wb_project = "protect_vs_invade"  # Weights & Biases project name
wb_name = "_adjusting_44"         # Experiment name for logging
logdir = os.path.join("tblogs", wb_project + wb_name)  # Log directory
run_id = "16"                     # Run identifier