import datetime

""" Hyperparameters for Tracking Environment"""


class EnvParameters:
    N_ACTIONS = 48  # action space size
    EPISODE_LEN = 256  # maximum episode length


class TrainingParameters:
    lr = 1e-5  # learning rate
    GAMMA = 0.95  # discount factor
    LAM = 0.95  # GAE lambda (for future use)
    CLIP_RANGE = 0.2  # PPO clipping range
    MAX_GRAD_NORM = 10  # gradient clipping
    ENTROPY_COEF = 0.01  # entropy coefficient
    EX_VALUE_COEF = 0.08  # value loss coefficient
    N_EPOCHS = 10  # number of training epochs
    N_ENVS = 4  # number of parallel environments
    N_MAX_STEPS = 3e7  # maximum training steps
    N_STEPS = 2 ** 10  # steps per environment per collection
    MINIBATCH_SIZE = int(2 ** 10)  # minibatch size for training
    DEMONSTRATION_PROB = 0.1  # probability of imitation learning

    # Agent training configuration
    AGENT_TO_TRAIN = "target"  # "tracker" or "target"
    OPPONENT_TYPE = "rule"      # "rule" or "policy"


class NetParameters:
    VECTOR_LEN = 12  # observation vector length


class SetupParameters:
    SEED = 1234
    USE_GPU_LOCAL = False
    USE_GPU_GLOBAL = True
    NUM_GPU = 1
    
    # Pre-trained model paths (for policy opponents)
    PRETRAINED_TRACKER_PATH = "./models/TrackingEnv/pretrained_tracker/tracker_net_checkpoint.pkl"
    PRETRAINED_TARGET_PATH = "./models/TrackingEnv/pretrained_target/target_net_checkpoint.pkl"


class RecordingParameters:
    RETRAIN = False
    WANDB = False
    TENSORBOARD = True
    TXT_WRITER = True
    ENTITY = 'yutong'
    TIME = datetime.datetime.now().strftime('%d-%m-%y%H%M')
    EXPERIMENT_PROJECT = 'TrackingEnv'
    EXPERIMENT_NAME = 'DualAgent'
    EXPERIMENT_NOTE = ''
    
    # Intervals
    SAVE_INTERVAL = 5e5
    BEST_INTERVAL = 0
    GIF_INTERVAL = 1e6
    EVAL_INTERVAL = TrainingParameters.N_ENVS * TrainingParameters.N_STEPS
    EVAL_EPISODES = 1
    
    # Paths
    RECORD_BEST = False
    MODEL_PATH = './models' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + TIME
    GIFS_PATH = './gifs' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + TIME
    SUMMARY_PATH = './summaries' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + TIME
    TXT_NAME = 'alg.txt'
    
    # Loss names for logging
    LOSS_NAME = ['total_loss', 'policy_loss', 'entropy', 'value_loss', 
                 'value_loss2', 'valid_loss', 'blocking_loss', 'clipfrac', 
                 'grad_norm', 'advantages']