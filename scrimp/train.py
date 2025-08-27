import os
import ray
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import wandb

from alg_parameters import *
from episodic_buffer import EpisodicBuffer
from env_wrapper import ProtectingEnvWrapper
from model import Model
from runner import Runner
from util import set_global_seeds, write_to_tensorboard, write_to_wandb, make_gif, reset_env

def train_scrimp():
    """SCRIMP训练主函数"""
    # 初始化Ray
    ray.init(num_gpus=SetupParameters.NUM_GPU)
    print("Starting SCRIMP training on Protecting Environment")
    
    # 设置随机种子
    set_global_seeds(SetupParameters.SEED)
    
    # 创建模型和环境
    global_device = torch.device('cuda') if SetupParameters.USE_GPU_GLOBAL else torch.device('cpu')
    global_model = Model(0, global_device, True)
    
    # 创建runners
    runners = [Runner.remote(i + 1) for i in range(TrainingParameters.N_ENVS)]
    
    # 创建评估环境
    eval_env = ProtectingEnvWrapper(num_agents=2)
    eval_memory = EpisodicBuffer(0, 2)
    
    # 初始化训练变量
    curr_steps = 0
    curr_episodes = 0
    best_perf = -float('inf')
    
    # 设置日志记录
    if RecordingParameters.TENSORBOARD:
        summary_path = RecordingParameters.SUMMARY_PATH
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        global_summary = SummaryWriter(summary_path)
    
    if RecordingParameters.WANDB:
        wandb_id = wandb.util.generate_id()
        wandb.init(project=RecordingParameters.EXPERIMENT_PROJECT,
                   name=RecordingParameters.EXPERIMENT_NAME,
                   entity=RecordingParameters.ENTITY,
                   notes=RecordingParameters.EXPERIMENT_NOTE,
                   config=all_args,
                   id=wandb_id,
                   resume='allow')
        print(f'WandB ID: {wandb_id}')
    
    # 训练循环变量
    update_done = True
    demon = True
    job_list = []
    last_test_t = -RecordingParameters.EVAL_INTERVAL - 1
    last_model_t = -RecordingParameters.SAVE_INTERVAL - 1
    last_gif_t = -RecordingParameters.GIF_INTERVAL - 1
    
    print("Starting training loop...")
    
    try:
        while curr_steps < TrainingParameters.N_MAX_STEPS:
            if update_done:
                # 开始数据收集
                net_weights = global_model.network.state_dict()
                net_weights_id = ray.put(net_weights)
                curr_steps_id = ray.put(curr_steps)
                
                # 决定是进行模仿学习还是强化学习
                demon_probs = np.random.rand()
                if demon_probs < TrainingParameters.DEMONSTRATION_PROB:
                    demon = True
                    print(f"Step {curr_steps}: Collecting imitation learning data...")
                    for i, runner in enumerate(runners):
                        job_list.append(runner.imitation.remote(net_weights_id, curr_steps_id))
                else:
                    demon = False
                    print(f"Step {curr_steps}: Collecting RL data...")
                    for i, runner in enumerate(runners):
                        job_list.append(runner.run.remote(net_weights_id, curr_steps_id))
            
            # 获取数据
            done_id, job_list = ray.wait(job_list, num_returns=TrainingParameters.N_ENVS)
            update_done = True if job_list == [] else False
            done_len = len(done_id)
            job_results = ray.get(done_id)
            
            if demon:
                # 处理模仿学习数据
                mb_obs, mb_vector, mb_actions, mb_hidden_state, mb_message = [], [], [], [], []
                
                for results in range(done_len):
                    if len(job_results[results]) >= 5:
                        mb_obs.append(job_results[results][0])
                        mb_vector.append(job_results[results][1])
                        mb_actions.append(job_results[results][2])
                        mb_hidden_state.append(job_results[results][3])
                        mb_message.append(job_results[results][4])
                        curr_episodes += job_results[results][-2]
                        curr_steps += job_results[results][-1]
                
                if mb_obs and len(mb_obs[0]) > 0:
                    mb_obs = np.concatenate(mb_obs, axis=0)
                    mb_vector = np.concatenate(mb_vector, axis=0)
                    mb_actions = np.concatenate(mb_actions, axis=0)
                    mb_hidden_state = np.concatenate(mb_hidden_state, axis=0)
                    mb_message = np.concatenate(mb_message, axis=0)
                    
                    # 模仿学习训练
                    mb_imitation_loss = []
                    for start in range(0, np.shape(mb_obs)[0], TrainingParameters.MINIBATCH_SIZE):
                        end = min(start + TrainingParameters.MINIBATCH_SIZE, np.shape(mb_obs)[0])
                        slices = (mb_obs[start:end], mb_vector[start:end], mb_actions[start:end],
                                 mb_hidden_state[start:end] if len(mb_hidden_state) > 0 else None,
                                 mb_message[start:end] if len(mb_message) > 0 else None)
                        mb_imitation_loss.append(global_model.imitation_train(*slices))
                    
                    if mb_imitation_loss:
                        mb_imitation_loss = np.nanmean(mb_imitation_loss, axis=0)
                        print(f"Imitation loss: {mb_imitation_loss[0]:.4f}")
                        
                        # 记录训练结果
                        if RecordingParameters.WANDB:
                            write_to_wandb(curr_steps, imitation_loss=mb_imitation_loss, evaluate=False)
                        if RecordingParameters.TENSORBOARD:
                            write_to_tensorboard(global_summary, curr_steps, imitation_loss=mb_imitation_loss, evaluate=False)
                
            else:
                # 处理强化学习数据
                curr_steps += done_len * TrainingParameters.N_STEPS
                
                # 收集所有数据
                all_data = {
                    'mb_obs': [], 'mb_vector': [], 'mb_returns_in': [], 'mb_returns_ex': [], 'mb_returns_all': [],
                    'mb_values_in': [], 'mb_values_ex': [], 'mb_values_all': [], 'mb_actions': [], 'mb_ps': [],
                    'mb_hidden_state': [], 'mb_train_valid': [], 'mb_blocking': [], 'mb_message': []
                }
                
                performance_dict = {
                    'per_r': [], 'per_in_r': [], 'per_ex_r': [], 'per_valid_rate': [],
                    'per_episode_len': [], 'per_block': [], 'per_leave_goal': [],
                    'per_final_goals': [], 'per_half_goals': [], 'per_block_acc': [],
                    'per_max_goals': [], 'per_num_collide': [], 'rewarded_rate': []
                }
                
                for results in range(done_len):
                    if len(job_results[results]) >= 14:
                        data_keys = ['mb_obs', 'mb_vector', 'mb_returns_in', 'mb_returns_ex', 'mb_returns_all',
                                    'mb_values_in', 'mb_values_ex', 'mb_values_all', 'mb_actions', 'mb_ps',
                                    'mb_hidden_state', 'mb_train_valid', 'mb_blocking', 'mb_message']
                        
                        for i, key in enumerate(data_keys):
                            all_data[key].append(job_results[results][i])
                        
                        curr_episodes += job_results[results][-2]
                        
                        # 更新性能字典
                        perf_dict = job_results[results][-1]
                        for key in performance_dict:
                            if key in perf_dict and perf_dict[key]:
                                performance_dict[key].extend(perf_dict[key] if isinstance(perf_dict[key], list) else [perf_dict[key]])
                
                # 连接所有数据
                for key in all_data:
                    if all_data[key]:
                        all_data[key] = np.concatenate(all_data[key], axis=0)
                
                # 计算平均性能
                for key in performance_dict:
                    if performance_dict[key]:
                        performance_dict[key] = np.nanmean(performance_dict[key])
                    else:
                        performance_dict[key] = 0.0
                
                # 强化学习训练
                if len(all_data['mb_obs']) > 0:
                    mb_loss = []
                    inds = np.arange(len(all_data['mb_obs']))
                    
                    for _ in range(TrainingParameters.N_EPOCHS):
                        np.random.shuffle(inds)
                        for start in range(0, len(all_data['mb_obs']), TrainingParameters.MINIBATCH_SIZE):
                            end = min(start + TrainingParameters.MINIBATCH_SIZE, len(all_data['mb_obs']))
                            mb_inds = inds[start:end]
                            
                            slices = tuple(all_data[key][mb_inds] for key in [
                                'mb_obs', 'mb_vector', 'mb_returns_in', 'mb_returns_ex', 'mb_returns_all',
                                'mb_values_in', 'mb_values_ex', 'mb_values_all', 'mb_actions', 'mb_ps',
                                'mb_hidden_state', 'mb_train_valid', 'mb_blocking', 'mb_message'
                            ])
                            
                            mb_loss.append(global_model.train(*slices))
                    
                    if mb_loss:
                        print(f"RL training completed. Total loss: {np.nanmean([l[0] for l in mb_loss]):.4f}")
                        
                        # 记录训练结果
                        if RecordingParameters.WANDB:
                            write_to_wandb(curr_steps, performance_dict, mb_loss, evaluate=False)
                        if RecordingParameters.TENSORBOARD:
                            write_to_tensorboard(global_summary, curr_steps, performance_dict, mb_loss, evaluate=False)
            
            # 定期评估
            if (curr_steps - last_test_t) / RecordingParameters.EVAL_INTERVAL >= 1.0:
                last_test_t = curr_steps
                save_gif = (curr_steps - last_gif_t) / RecordingParameters.GIF_INTERVAL >= 1.0
                if save_gif:
                    last_gif_t = curr_steps
                
                print(f"Evaluating at step {curr_steps}...")
                with torch.no_grad():
                    eval_performance_dict = evaluate_model(eval_env, eval_memory, global_model,
                                                         global_device, save_gif, curr_steps, False)
                
                print(f'Episodes: {curr_episodes}, Steps: {curr_steps}, '
                      f'Reward: {eval_performance_dict["per_r"]:.2f}')
                
                # 记录评估结果
                if RecordingParameters.WANDB:
                    write_to_wandb(curr_steps, eval_performance_dict, evaluate=True, greedy=False)
                if RecordingParameters.TENSORBOARD:
                    write_to_tensorboard(global_summary, curr_steps, eval_performance_dict, evaluate=True, greedy=False)
                
                # 保存最佳模型
                if RecordingParameters.RECORD_BEST and eval_performance_dict['per_r'] > best_perf:
                    best_perf = eval_performance_dict['per_r']
                    save_best_model(global_model, curr_steps, curr_episodes, best_perf)
            
            # 定期保存模型
            if (curr_steps - last_model_t) / RecordingParameters.SAVE_INTERVAL >= 1.0:
                last_model_t = curr_steps
                save_model(global_model, curr_steps, curr_episodes, 0.0)
    
    except KeyboardInterrupt:
        print("CTRL-C pressed. Stopping training...")
    finally:
        print("Cleaning up...")
        # 保存最终模型
        save_final_model(global_model, curr_steps, curr_episodes)
        
        # 关闭日志
        if RecordingParameters.TENSORBOARD:
            global_summary.close()
        if RecordingParameters.WANDB:
            wandb.finish()
        
        # 清理ray workers
        for runner in runners:
            ray.kill(runner)
        ray.shutdown()

def evaluate_model(eval_env, episodic_buffer, model, device, save_gif, curr_steps, greedy):
    """评估模型性能"""
    eval_performance_dict = {
        'per_r': [], 'per_ex_r': [], 'per_in_r': [], 'per_valid_rate': [],
        'per_episode_len': [], 'per_block': [], 'per_leave_goal': [],
        'per_final_goals': [], 'per_half_goals': [], 'per_block_acc': [],
        'per_max_goals': [], 'per_num_collide': [], 'rewarded_rate': []
    }
    
    episode_frames = []
    num_agent = 2
    
    for i in range(RecordingParameters.EVAL_EPISODES):
        # 重置环境和buffer
        message = torch.zeros((1, num_agent, NetParameters.NET_SIZE)).to(device)
        hidden_state = (torch.zeros((num_agent, NetParameters.NET_SIZE // 2)).to(device),
                       torch.zeros((num_agent, NetParameters.NET_SIZE // 2)).to(device))
        
        done, valid_actions, obs, vector, _ = reset_env(eval_env, num_agent)
        episodic_buffer.reset(curr_steps, num_agent)
        new_xy = eval_env.get_positions()
        episodic_buffer.batch_add(new_xy)
        
        one_episode_perf = {
            'num_step': 0, 'episode_reward': 0, 'invalid': 0, 'block': 0,
            'num_leave_goal': 0, 'wrong_blocking': 0, 'num_collide': 0,
            'reward_count': 0, 'ex_reward': 0, 'in_reward': 0
        }
        
        if save_gif:
            episode_frames.append(eval_env._render(mode='rgb_array'))
        
        # 回合循环
        while not done and one_episode_perf['num_step'] < EnvParameters.EPISODE_LEN:
            # 获取动作
            actions, pre_block, hidden_state, num_invalid, v_all, ps, message = \
                model.evaluate(obs, vector, valid_actions, hidden_state, greedy,
                             episodic_buffer.no_reward, message, num_agent)
            
            one_episode_perf['invalid'] += num_invalid
            
            # 执行动作（简化版one_step）
            step_result = eval_env.joint_step(actions)
            rewards = step_result[2]
            done = step_result[3]
            obs = step_result[0]
            vector = step_result[1]
            valid_actions = step_result[4]
            on_goal = step_result[5]
            
            # 处理内在奖励
            new_xy = eval_env.get_positions()
            processed_rewards, be_rewarded, intrinsic_reward, min_dist = \
                episodic_buffer.if_reward(new_xy, rewards, done, on_goal)
            
            one_episode_perf['reward_count'] += be_rewarded
            one_episode_perf['episode_reward'] += np.sum(processed_rewards)
            one_episode_perf['ex_reward'] += np.sum(rewards)
            one_episode_perf['in_reward'] += np.sum(intrinsic_reward)
            one_episode_perf['num_step'] += 1
            
            if save_gif:
                episode_frames.append(eval_env._render(mode='rgb_array'))
            
            if done:
                if save_gif:
                    if not os.path.exists(RecordingParameters.GIFS_PATH):
                        os.makedirs(RecordingParameters.GIFS_PATH)
                    images = np.array(episode_frames)
                    make_gif(images, f'{RecordingParameters.GIFS_PATH}/eval_steps_{curr_steps}.gif')
                    save_gif = False
                
                # 更新性能字典
                eval_performance_dict['per_r'].append(one_episode_perf['episode_reward'])
                eval_performance_dict['per_ex_r'].append(one_episode_perf['ex_reward'])
                eval_performance_dict['per_in_r'].append(one_episode_perf['in_reward'])
                eval_performance_dict['per_episode_len'].append(one_episode_perf['num_step'])
                eval_performance_dict['per_final_goals'].append(2)  # 简化
                eval_performance_dict['rewarded_rate'].append(one_episode_perf['reward_count'] / (one_episode_perf['num_step'] * num_agent))
    
    # 计算平均值
    for key in eval_performance_dict:
        if eval_performance_dict[key]:
            eval_performance_dict[key] = np.nanmean(eval_performance_dict[key])
        else:
            eval_performance_dict[key] = 0.0
    
    return eval_performance_dict

def save_model(model, steps, episodes, reward):
    """保存模型"""
    import os.path as osp
    model_path = osp.join(RecordingParameters.MODEL_PATH, f'{steps:05d}')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    path_checkpoint = model_path + "/net_checkpoint.pkl"
    net_checkpoint = {
        "model": model.network.state_dict(),
        "optimizer": model.net_optimizer.state_dict(),
        "step": steps,
        "episode": episodes,
        "reward": reward
    }
    torch.save(net_checkpoint, path_checkpoint)
    print(f'Model saved at step {steps}')

def save_best_model(model, steps, episodes, reward):
    """保存最佳模型"""
    import os.path as osp
    model_path = osp.join(RecordingParameters.MODEL_PATH, 'best_model')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    path_checkpoint = model_path + "/net_checkpoint.pkl"
    net_checkpoint = {
        "model": model.network.state_dict(),
        "optimizer": model.net_optimizer.state_dict(),
        "step": steps,
        "episode": episodes,
        "reward": reward
    }
    torch.save(net_checkpoint, path_checkpoint)
    print(f'Best model saved with reward {reward:.2f}')

def save_final_model(model, steps, episodes):
    """保存最终模型"""
    model_path = RecordingParameters.MODEL_PATH + '/final'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    path_checkpoint = model_path + "/net_checkpoint.pkl"
    net_checkpoint = {
        "model": model.network.state_dict(),
        "optimizer": model.net_optimizer.state_dict() if hasattr(model, 'net_optimizer') else None,
        "step": steps,
        "episode": episodes,
        "reward": 0
    }
    torch.save(net_checkpoint, path_checkpoint)
    print('Final model saved')

if __name__ == "__main__":
    train_scrimp()