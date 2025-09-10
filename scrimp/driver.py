import os
import os.path as osp
import math
import numpy as np
import torch
import ray

try:
    import setproctitle
except Exception:
    setproctitle = None

from torch.utils.tensorboard import SummaryWriter

from alg_parameters import *
from env import TrackingEnv
from model import Model
from runner import Runner
from util import set_global_seeds, write_to_tensorboard, write_to_wandb, make_gif
from expert_policies import get_expert_tracker_action_pair, get_expert_target_action_pair

try:
    import wandb
except Exception:
    wandb = None

# IL cosine annealing
IL_INITIAL_PROB = 0.8
IL_FINAL_PROB = 0.1
IL_DECAY_STEPS = 1e7

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
if not ray.is_initialized():
    ray.init(num_gpus=SetupParameters.NUM_GPU)
print("Welcome to SCRIMP on Protecting Environment!\n")
print(f"Training agent: {TrainingParameters.AGENT_TO_TRAIN} with {TrainingParameters.OPPONENT_TYPE} opponent")
print(f"IL type: {getattr(TrainingParameters, 'IL_TYPE', 'expert')}")
print(f"IL probability will cosine anneal from {IL_INITIAL_PROB*100:.1f}% to {IL_FINAL_PROB*100:.1f}% over {IL_DECAY_STEPS} steps")

# Defaults for missing RecordingParameters fields
def_attr = lambda name, default: getattr(RecordingParameters, name, default)
SUMMARY_PATH = def_attr('SUMMARY_PATH', f'./runs/TrackingEnv/{RecordingParameters.EXPERIMENT_NAME}{RecordingParameters.TIME}')
MODEL_PATH = def_attr('MODEL_PATH', f'./models/TrackingEnv/{RecordingParameters.EXPERIMENT_NAME}{RecordingParameters.TIME}')
GIFS_PATH = def_attr('GIFS_PATH', osp.join(MODEL_PATH, 'gifs'))
EVAL_INTERVAL = int(def_attr('EVAL_INTERVAL', 20000))
SAVE_INTERVAL = int(def_attr('SAVE_INTERVAL', 5e5))
BEST_INTERVAL = int(def_attr('BEST_INTERVAL', 0))
GIF_INTERVAL = int(def_attr('GIF_INTERVAL', 1e5))
EVAL_EPISODES = int(def_attr('EVAL_EPISODES', 5))

all_args = {
    'seed': SetupParameters.SEED,
    'n_envs': TrainingParameters.N_ENVS,
    'n_steps': TrainingParameters.N_STEPS,
    'learning_rate': TrainingParameters.lr,
    'max_steps': TrainingParameters.N_MAX_STEPS,
    'episode_len': EnvParameters.EPISODE_LEN,
    'n_actions': EnvParameters.N_ACTIONS,
    'agent_to_train': TrainingParameters.AGENT_TO_TRAIN,
    'opponent_type': TrainingParameters.OPPONENT_TYPE,
    'il_type': getattr(TrainingParameters, 'IL_TYPE', 'expert'),  # 添加这一行
    'il_initial_prob': IL_INITIAL_PROB,
    'il_final_prob': IL_FINAL_PROB,
    'il_decay_steps': IL_DECAY_STEPS
}


def get_cosine_annealing_il_prob(current_step):
    if current_step >= IL_DECAY_STEPS:
        return IL_FINAL_PROB
    cosine_decay = 0.5 * (1 + math.cos(math.pi * current_step / IL_DECAY_STEPS))
    return IL_FINAL_PROB + (IL_INITIAL_PROB - IL_FINAL_PROB) * cosine_decay


def main():
    # preparing for training
    model_dict = None
    wandb_id = None

    if def_attr('RETRAIN', False):
        restore_path = def_attr('RESTORE_DIR', None)
        if restore_path:
            model_path = restore_path + f"/{TrainingParameters.AGENT_TO_TRAIN}_net_checkpoint.pkl"
            if os.path.exists(model_path):
                model_dict = torch.load(model_path, map_location='cpu')

    if def_attr('WANDB', False) and wandb is not None:
        wandb_id = model_dict.get('wandb_id', None) if model_dict else None
        wandb.init(project=RecordingParameters.EXPERIMENT_PROJECT,
                   name=RecordingParameters.EXPERIMENT_NAME,
                   entity=getattr(RecordingParameters, 'ENTITY', None),
                   notes=getattr(RecordingParameters, 'EXPERIMENT_NOTE', ''),
                   config=all_args,
                   id=wandb_id,
                   resume='allow')
        print('Launching wandb...\n')

    global_summary = None
    if def_attr('TENSORBOARD', True):
        os.makedirs(SUMMARY_PATH, exist_ok=True)
        global_summary = SummaryWriter(SUMMARY_PATH)
        print('Launching tensorboard...\n')

    if setproctitle is not None:
        setproctitle.setproctitle(
            RecordingParameters.EXPERIMENT_PROJECT + RecordingParameters.EXPERIMENT_NAME + "@" + getattr(RecordingParameters, 'ENTITY', 'user'))
    set_global_seeds(SetupParameters.SEED)

    # Devices and models
    global_device = torch.device('cuda') if SetupParameters.USE_GPU_GLOBAL else torch.device('cpu')
    training_model = Model(global_device, True)

    if model_dict is not None:
        training_model.network.load_state_dict(model_dict['model'])
        training_model.net_optimizer.load_state_dict(model_dict['optimizer'])

    opponent_model = None
    opponent_weights = None
    if TrainingParameters.OPPONENT_TYPE == "policy":
        opponent_model = Model(global_device, False)
        if TrainingParameters.AGENT_TO_TRAIN == "tracker":
            opp_path = SetupParameters.PRETRAINED_TARGET_PATH
        else:
            opp_path = SetupParameters.PRETRAINED_TRACKER_PATH
        if opp_path and os.path.exists(opp_path):
            opponent_dict = torch.load(opp_path, map_location='cpu')
            opponent_model.network.load_state_dict(opponent_dict['model'])
            opponent_weights = opponent_model.get_weights()

    # Mission mapping: 0->train tracker, 1->train target (reward flipped in env)
    env_mission = 0 if TrainingParameters.AGENT_TO_TRAIN == "tracker" else 1

    # Envs
    envs = [Runner.remote(i + 1, env_mission) for i in range(TrainingParameters.N_ENVS)]
    eval_env = TrackingEnv(mission=env_mission)

    # State
    curr_steps = int(model_dict.get("step", 0)) if model_dict is not None else 0
    curr_episodes = int(model_dict.get("episode", 0)) if model_dict is not None else 0
    best_perf = float(model_dict.get("reward", -1e9)) if model_dict is not None else -1e9

    last_test_t = -int(EVAL_INTERVAL) - 1
    last_model_t = -int(SAVE_INTERVAL) - 1
    last_best_t = -int(BEST_INTERVAL) - 1
    last_gif_t = -int(GIF_INTERVAL) - 1

    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(GIFS_PATH, exist_ok=True)

    try:
        while curr_steps < TrainingParameters.N_MAX_STEPS:
            # Decide IL vs RL
            il_prob = get_cosine_annealing_il_prob(curr_steps)
            do_il = (np.random.rand() < il_prob)

            weights = training_model.get_weights()
            jobs = []
            
            if do_il:
                # IL训练分支
                for e in envs:
                    jobs.append(e.imitation.remote(weights, opponent_weights, curr_steps))
                il_batches = ray.get(jobs)

                vec = np.concatenate([b['vector'] for b in il_batches], axis=0)
                lbl = np.concatenate([b['actions'] for b in il_batches], axis=0)
                idx = np.random.permutation(len(vec))
                vec = vec[idx]
                lbl = lbl[idx]

                mb_loss = []
                for start in range(0, len(vec), TrainingParameters.MINIBATCH_SIZE):
                    end = min(start + TrainingParameters.MINIBATCH_SIZE, len(vec))
                    mb_loss.append(training_model.imitation_train(vec[start:end], lbl[start:end]))

                # 创建一个虚拟的性能字典用于IL训练记录
                avg_perf = {'per_r': 0.0, 'per_ex_r': 0.0, 'per_in_r': 0.0, 'per_valid_rate': 1.0,
                        'per_episode_len': 0.0, 'rewarded_rate': 0.0}
                
                write_to_tensorboard(global_summary, curr_steps, imitation_loss=np.nanmean(mb_loss, axis=0), evaluate=False)
                if getattr(RecordingParameters, 'WANDB', False) and (wandb is not None) and (getattr(wandb, 'run', None) is not None):
                    write_to_wandb(curr_steps, performance_dict=avg_perf, mb_loss=mb_loss, evaluate=False)

                curr_steps += int(TrainingParameters.N_ENVS * TrainingParameters.N_STEPS)
                
            else:
                # RL rollout
                jobs = [e.run.remote(weights, opponent_weights, curr_steps) for e in envs]
                results = ray.get(jobs)

                vectors = np.concatenate([r[0]['vector'] for r in results], axis=0)
                returns = np.concatenate([r[0]['returns'] for r in results], axis=0)
                values = np.concatenate([r[0]['values'] for r in results], axis=0)
                actions = np.concatenate([r[0]['actions'] for r in results], axis=0)
                probs = np.concatenate([r[0]['ps'] for r in results], axis=0)
                steps_batch = int(sum(r[1] for r in results))
                episodes_batch = int(sum(r[2] for r in results))

                # Aggregate performance
                performance_dict = {'per_r': [], 'per_ex_r': [], 'per_in_r': [], 'per_valid_rate': [],
                                'per_episode_len': [], 'rewarded_rate': []}
                for r in results:
                    perf = r[3]
                    for k, v in perf.items():
                        performance_dict[k].extend(v)

                # Train PPO
                mb_loss = []
                inds = np.arange(len(vectors))
                for epoch in range(TrainingParameters.N_EPOCHS):
                    np.random.shuffle(inds)
                    for start in range(0, len(vectors), TrainingParameters.MINIBATCH_SIZE):
                        end = min(start + TrainingParameters.MINIBATCH_SIZE, len(vectors))
                        mb = inds[start:end]
                        mb_loss.append(training_model.train(
                            vectors[mb], returns[mb], values[mb], actions[mb], probs[mb], None
                        ))

                avg_perf = {k: (float(np.nanmean(v)) if len(v) > 0 else 0.0) for k, v in performance_dict.items()}
                write_to_tensorboard(global_summary, curr_steps, performance_dict=avg_perf, mb_loss=mb_loss, evaluate=False)
                if getattr(RecordingParameters, 'WANDB', False) and (wandb is not None) and (getattr(wandb, 'run', None) is not None):
                    write_to_wandb(curr_steps, performance_dict=avg_perf, mb_loss=mb_loss, evaluate=False)

                curr_steps += steps_batch
                curr_episodes += episodes_batch

            # 将评估、保存模型等逻辑移到这里，让IL和RL都能触发
            # Save latest
            if curr_steps - last_model_t >= SAVE_INTERVAL:
                last_model_t = curr_steps
                model_path = osp.join(MODEL_PATH, 'latest')
                os.makedirs(model_path, exist_ok=True)
                save_path = model_path + f"/{TrainingParameters.AGENT_TO_TRAIN}_net_checkpoint.pkl"
                checkpoint = {"model": training_model.network.state_dict(),
                            "optimizer": training_model.net_optimizer.state_dict(),
                            "step": curr_steps, "episode": curr_episodes, "reward": avg_perf['per_r']}
                torch.save(checkpoint, save_path)
                print(f"Saved latest model at step {curr_steps}")

            # Save best (只在有实际性能数据时更新，即RL模式下)
            if not do_il:  # 只在RL模式下更新best performance
                mean_r = avg_perf['per_r']
                if mean_r > best_perf and (curr_steps - last_best_t >= BEST_INTERVAL):
                    best_perf = mean_r
                    last_best_t = curr_steps
                    model_path = osp.join(MODEL_PATH, 'best_model')
                    os.makedirs(model_path, exist_ok=True)
                    save_path = model_path + f"/{TrainingParameters.AGENT_TO_TRAIN}_net_checkpoint.pkl"
                    checkpoint = {"model": training_model.network.state_dict(),
                                "optimizer": training_model.net_optimizer.state_dict(),
                                "step": curr_steps, "episode": curr_episodes, "reward": best_perf}
                    torch.save(checkpoint, save_path)
                    print(f"New best model saved with reward {best_perf:.4f}")

            # Evaluate (现在IL和RL都会评估)
            if curr_steps - last_test_t >= EVAL_INTERVAL:
                last_test_t = curr_steps
                eval_perf = evaluate_single_agent(eval_env, training_model, opponent_model, global_device,
                                                save_gif=(curr_steps - last_gif_t >= GIF_INTERVAL),
                                                curr_steps=curr_steps)
                write_to_tensorboard(global_summary, curr_steps, performance_dict=eval_perf, evaluate=True)
                if getattr(RecordingParameters, 'WANDB', False) and (wandb is not None) and (getattr(wandb, 'run', None) is not None):
                    write_to_wandb(curr_steps, performance_dict=eval_perf, evaluate=True)
                if curr_steps - last_gif_t >= GIF_INTERVAL:
                    last_gif_t = curr_steps

    except KeyboardInterrupt:
        print("CTRL-C pressed. killing remote workers")
    finally:
        # save final model
        print('Saving Final Model !\n')
        model_path = MODEL_PATH + '/final'
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
        save_path = model_path + f"/{TrainingParameters.AGENT_TO_TRAIN}_net_checkpoint.pkl"
        checkpoint = {"model": training_model.network.state_dict(),
                      "optimizer": training_model.net_optimizer.state_dict(),
                      "step": curr_steps, "episode": curr_episodes, "reward": best_perf}
        torch.save(checkpoint, save_path)
        print(f"Saved final model to {save_path}")


def evaluate_single_agent(eval_env, agent_model, opponent_model, device, save_gif, curr_steps):
    eval_performance_dict = {'per_r': [], 'per_ex_r': [], 'per_in_r': [], 'per_valid_rate': [],
                             'per_episode_len': [], 'rewarded_rate': []}
    episode_frames = []

    # 为评估创建策略管理器（如果使用随机对手）
    eval_policy_manager = None
    if TrainingParameters.OPPONENT_TYPE == "random":
        from policymanager import PolicyManager
        eval_policy_manager = PolicyManager()
        if hasattr(TrainingParameters, 'RANDOM_OPPONENT_WEIGHTS'):
            opponent_role = "target" if TrainingParameters.AGENT_TO_TRAIN == "tracker" else "tracker"
            custom_weights = getattr(TrainingParameters, 'RANDOM_OPPONENT_WEIGHTS', {})
            if opponent_role in custom_weights:
                eval_policy_manager.set_policy_weights(opponent_role, custom_weights[opponent_role])

    for _ in range(EVAL_EPISODES):
        obs, _ = eval_env.reset()
        done = False
        ep_r = 0.0
        ep_len = 0
        reward_cnt = 0
        
        # 为每个episode选择随机对手策略
        current_eval_policy = None
        if eval_policy_manager:
            opponent_role = "target" if TrainingParameters.AGENT_TO_TRAIN == "tracker" else "tracker"
            current_eval_policy = eval_policy_manager.sample_policy(opponent_role)
            eval_policy_manager.reset()
        
        while not done and ep_len < EnvParameters.EPISODE_LEN:
            agent_pair, _, _, _, _ = agent_model.evaluate(obs, greedy=True)
            
            # 获取对手动作
            if TrainingParameters.OPPONENT_TYPE == "policy":
                if opponent_model is None:
                    raise RuntimeError("OPPONENT_TYPE=policy but opponent_model is None in evaluation")
                opp_pair, _, _, _, _ = opponent_model.evaluate(obs, greedy=True)
            elif TrainingParameters.OPPONENT_TYPE == "expert":
                if TrainingParameters.AGENT_TO_TRAIN == "tracker":
                    opp_pair = get_expert_target_action_pair(obs)
                else:
                    opp_pair = get_expert_tracker_action_pair(obs)
            elif TrainingParameters.OPPONENT_TYPE == "random":
                if eval_policy_manager and current_eval_policy:
                    opp_pair = eval_policy_manager.get_action(current_eval_policy, obs)
                else:
                    # 回退到专家策略
                    if TrainingParameters.AGENT_TO_TRAIN == "tracker":
                        opp_pair = get_expert_target_action_pair(obs)
                    else:
                        opp_pair = get_expert_tracker_action_pair(obs)
            else:
                raise ValueError(f"Unsupported OPPONENT_TYPE: {TrainingParameters.OPPONENT_TYPE}")
            
            # 根据训练的agent类型确定tracker和target动作
            if TrainingParameters.AGENT_TO_TRAIN == "tracker":
                tracker_action, target_action = agent_pair, opp_pair
            else:
                tracker_action, target_action = opp_pair, agent_pair
                
            obs, reward, terminated, truncated, info = eval_env.step((tracker_action, target_action))
            done = terminated or truncated
            ep_r += float(reward)
            reward_cnt += 1 if reward > 0 else 0
            ep_len += 1

            frame = eval_env.render(mode='rgb_array')
            if frame is not None:
                episode_frames.append(frame)

        eval_performance_dict['per_r'].append(ep_r)
        eval_performance_dict['per_ex_r'].append(0.0)
        eval_performance_dict['per_in_r'].append(0.0)
        eval_performance_dict['per_valid_rate'].append(1.0)
        eval_performance_dict['per_episode_len'].append(ep_len)
        eval_performance_dict['rewarded_rate'].append(reward_cnt / max(1, ep_len * 2))

    for key in eval_performance_dict.keys():
        vals = eval_performance_dict[key]
        eval_performance_dict[key] = float(np.nanmean(vals)) if len(vals) > 0 else 0.0

    if save_gif and len(episode_frames) > 0:
        gif_path = osp.join(GIFS_PATH, f"eval_{int(curr_steps)}.gif")
        os.makedirs(GIFS_PATH, exist_ok=True)
        make_gif(episode_frames, gif_path, fps=20)

    return eval_performance_dict


if __name__ == "__main__":
    main()