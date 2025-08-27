import os
import os.path as osp
import numpy as np
import torch
import ray
import wandb
import setproctitle
import imageio
import math
from torch.utils.tensorboard import SummaryWriter

from alg_parameters import *
from episodic_buffer import EpisodicBuffer
from env import TrackingEnv
from model import Model
from runner import Runner
from util import set_global_seeds, write_to_tensorboard, write_to_wandb, make_gif

# Add IL cosine annealing parameters
IL_INITIAL_PROB = 0.9  # Start with 90% IL
IL_FINAL_PROB = 0.1    # End with 10% IL
IL_DECAY_STEPS = int(TrainingParameters.N_MAX_STEPS * 0.5)  # Decay over first 50% of training

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ray.init(num_gpus=SetupParameters.NUM_GPU)
print("Welcome to SCRIMP on Protecting Environment!\n") 
print(f"Training agent: {TrainingParameters.AGENT_TO_TRAIN} with {TrainingParameters.OPPONENT_TYPE} opponent")
print(f"IL probability will cosine anneal from {IL_INITIAL_PROB*100}% to {IL_FINAL_PROB*100}% over {IL_DECAY_STEPS} steps")

# 创建所有参数字典用于wandb配置
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
    'il_initial_prob': IL_INITIAL_PROB,
    'il_final_prob': IL_FINAL_PROB,
    'il_decay_steps': IL_DECAY_STEPS
}

def get_cosine_annealing_il_prob(current_step):
    """Calculate IL probability using cosine annealing schedule"""
    if current_step >= IL_DECAY_STEPS:
        return IL_FINAL_PROB
    
    # Cosine annealing formula
    cosine_decay = 0.5 * (1 + math.cos(math.pi * current_step / IL_DECAY_STEPS))
    return IL_FINAL_PROB + (IL_INITIAL_PROB - IL_FINAL_PROB) * cosine_decay

def main():
    """main code"""
    # preparing for training
    if RecordingParameters.RETRAIN:
        restore_path = './local_model'
        model_path = restore_path + f"/{TrainingParameters.AGENT_TO_TRAIN}_net_checkpoint.pkl"
        model_dict = torch.load(model_path)

    if RecordingParameters.WANDB:
        if RecordingParameters.RETRAIN:
            wandb_id = None
        else:
            wandb_id = wandb.util.generate_id()
        wandb.init(project=RecordingParameters.EXPERIMENT_PROJECT,
                   name=RecordingParameters.EXPERIMENT_NAME,
                   entity=RecordingParameters.ENTITY,
                   notes=RecordingParameters.EXPERIMENT_NOTE,
                   config=all_args,
                   id=wandb_id,
                   resume='allow')
        print('id is:{}'.format(wandb_id))
        print('Launching wandb...\n')

    if RecordingParameters.TENSORBOARD:
        if RecordingParameters.RETRAIN:
            summary_path = ''
        else:
            summary_path = RecordingParameters.SUMMARY_PATH
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        global_summary = SummaryWriter(summary_path)
        print('Launching tensorboard...\n')

        if RecordingParameters.TXT_WRITER:
            txt_path = summary_path + '/' + RecordingParameters.TXT_NAME
            with open(txt_path, "w") as f:
                f.write(str(all_args))
            print('Logging txt...\n')

    setproctitle.setproctitle(
        RecordingParameters.EXPERIMENT_PROJECT + RecordingParameters.EXPERIMENT_NAME + "@" + RecordingParameters.ENTITY)
    set_global_seeds(SetupParameters.SEED)

    # Create models based on training configuration
    global_device = torch.device('cuda') if SetupParameters.USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')
    
    # Initialize training model
    training_model = Model(global_device, True)
    
    # Load model if retraining
    if RecordingParameters.RETRAIN:
        training_model.network.load_state_dict(model_dict['model'])
        training_model.net_optimizer.load_state_dict(model_dict['optimizer'])
    
    # Initialize opponent model if using policy opponent
    opponent_model = None
    if TrainingParameters.OPPONENT_TYPE == "policy":
        opponent_model = Model(global_device, False)  # No optimizer needed for opponent
        
        # Load pretrained opponent model
        if TrainingParameters.AGENT_TO_TRAIN == "tracker":
            opponent_dict = torch.load(SetupParameters.PRETRAINED_TARGET_PATH)
        else:
            opponent_dict = torch.load(SetupParameters.PRETRAINED_TRACKER_PATH)
            
        opponent_model.network.load_state_dict(opponent_dict['model'])

    # Determine environment mission based on configuration
    if TrainingParameters.AGENT_TO_TRAIN == "tracker":
        if TrainingParameters.OPPONENT_TYPE == "rule":
            env_mission = 0  # train tracker (target uses rules)
        else:
            env_mission = 2  # train tracker (target uses policy)
    else:  # training target
        if TrainingParameters.OPPONENT_TYPE == "rule":
            env_mission = 1  # train target (tracker uses rules)
        else:
            env_mission = 3  # train target (tracker uses policy)

    # Create parallel environments and evaluation environment
    envs = [Runner.remote(i + 1, env_mission) for i in range(TrainingParameters.N_ENVS)]
    eval_env = TrackingEnv(mission=env_mission)
    
    # Setup training state
    if RecordingParameters.RETRAIN:
        curr_steps = model_dict["step"]
        curr_episodes = model_dict["episode"]
        best_perf = model_dict["reward"]
    else:
        curr_steps = curr_episodes = best_perf = 0

    update_done = True
    demon = True
    job_list = []
    last_test_t = -RecordingParameters.EVAL_INTERVAL - 1
    last_model_t = -RecordingParameters.SAVE_INTERVAL - 1
    last_best_t = -RecordingParameters.BEST_INTERVAL - 1
    last_gif_t = -RecordingParameters.GIF_INTERVAL - 1

    # start training
    try:
        while curr_steps < TrainingParameters.N_MAX_STEPS:
            if update_done:
                # start a data collection - send model weights
                if global_device != local_device:
                    model_weights = training_model.network.to(local_device).state_dict()
                    training_model.network.to(global_device)
                    
                    if TrainingParameters.OPPONENT_TYPE == "policy" and opponent_model:
                        opponent_weights = opponent_model.network.to(local_device).state_dict()
                        opponent_model.network.to(global_device)
                    else:
                        opponent_weights = None
                else:
                    model_weights = training_model.network.state_dict()
                    opponent_weights = opponent_model.network.state_dict() if opponent_model else None
                
                # Put weights in ray object store
                model_weights_id = ray.put(model_weights)
                opponent_weights_id = ray.put(opponent_weights) if opponent_weights else None
                curr_steps_id = ray.put(curr_steps)
                
                # Calculate current IL probability using cosine annealing
                current_il_prob = get_cosine_annealing_il_prob(curr_steps)
                
                # Log current IL probability
                if RecordingParameters.WANDB:
                    wandb.log({'IL_probability': current_il_prob}, step=curr_steps)
                if RecordingParameters.TENSORBOARD:
                    global_summary.add_scalar('Training/IL_probability', current_il_prob, curr_steps)
                
                # Decide whether to do imitation learning using current probability
                demon_probs = np.random.rand()
                if demon_probs < current_il_prob:
                    demon = True
                    for i, env in enumerate(envs):
                        job_list.append(env.imitation.remote(model_weights_id, opponent_weights_id, curr_steps_id))
                else:
                    demon = False
                    for i, env in enumerate(envs):
                        job_list.append(env.run.remote(model_weights_id, opponent_weights_id, curr_steps_id))

            # get data from multiple processes
            done_id, job_list = ray.wait(job_list, num_returns=TrainingParameters.N_ENVS)
            update_done = True if job_list == [] else False
            done_len = len(done_id)
            job_results = ray.get(done_id)
            
            if demon:
                # Process imitation learning data
                mb_vector, mb_actions, mb_hidden = [], [], []
                total_episodes = 0
                total_steps = 0
                
                for result in job_results:
                    if len(result) >= 4:  # Check for valid results
                        mb_vector.append(result[0])
                        mb_actions.append(result[1])
                        mb_hidden.append(result[2])
                        total_episodes += result[3]
                        total_steps += len(result[0]) if len(result[0]) > 0 else 0

                curr_episodes += total_episodes
                curr_steps += total_steps

                # Train the model using imitation learning
                if mb_vector and any(len(v) > 0 for v in mb_vector):
                    valid_vectors = [v for v in mb_vector if len(v) > 0]
                    valid_actions = [a for a in mb_actions if len(a) > 0]
                    valid_hidden = [h for h in mb_hidden if len(h) > 0]
                    
                    if valid_vectors:
                        mb_vector_concat = np.concatenate(valid_vectors, axis=0)
                        mb_actions_concat = np.concatenate(valid_actions, axis=0)
                        mb_hidden_concat = np.concatenate(valid_hidden, axis=0)

                        mb_loss = []
                        for start in range(0, mb_vector_concat.shape[0], TrainingParameters.MINIBATCH_SIZE):
                            end = min(start + TrainingParameters.MINIBATCH_SIZE, mb_vector_concat.shape[0])
                            slices = (mb_vector_concat[start:end], mb_actions_concat[start:end], mb_hidden_concat[start:end])
                            mb_loss.append(training_model.imitation_train(*slices))

                        if mb_loss:
                            mb_loss = np.nanmean(mb_loss, axis=0)
                            agent_name = TrainingParameters.AGENT_TO_TRAIN.capitalize()
                            
                            # Log training metrics
                            if RecordingParameters.WANDB:
                                wandb.log({f'{agent_name}_Loss/Imitation_loss': mb_loss[0]}, step=curr_steps)
                                wandb.log({f'{agent_name}_Grad/Imitation_grad': mb_loss[1]}, step=curr_steps)
                            if RecordingParameters.TENSORBOARD:
                                global_summary.add_scalar(f'{agent_name}_Loss/Imitation_loss', mb_loss[0], curr_steps)
                                global_summary.add_scalar(f'{agent_name}_Grad/Imitation_grad', mb_loss[1], curr_steps)

            else:
                # Process reinforcement learning data
                curr_steps += done_len * TrainingParameters.N_STEPS
                
                # Gather RL data
                training_data = {'vector': [], 'returns': [], 'values': [], 'actions': [], 'ps': [], 'hidden': []}
                performance_dict = {'per_r': [], 'per_in_r': [], 'per_ex_r': [], 'per_valid_rate': [],
                                    'per_episode_len': [], 'rewarded_rate': []}
                
                for result in job_results:
                    if len(result) >= 8:  # Check for valid results
                        # Training data
                        training_data['vector'].append(result[0])
                        training_data['returns'].append(result[1])
                        training_data['values'].append(result[2])
                        training_data['actions'].append(result[3])
                        training_data['ps'].append(result[4])
                        training_data['hidden'].append(result[5])
                        
                        curr_episodes += result[6]
                        
                        # Performance data
                        perf_data = result[7]
                        for key in performance_dict.keys():
                            if key in perf_data and perf_data[key]:
                                if isinstance(perf_data[key], list):
                                    performance_dict[key].extend(perf_data[key])
                                else:
                                    performance_dict[key].append(perf_data[key])

                # Calculate average performance
                for key in performance_dict.keys():
                    if performance_dict[key]:
                        performance_dict[key] = np.nanmean(performance_dict[key])
                    else:
                        performance_dict[key] = 0.0

                # Train model with RL data
                if training_data['vector']:
                    for key in training_data:
                        training_data[key] = np.concatenate(training_data[key], axis=0)
                    
                    mb_loss = []
                    inds = np.arange(len(training_data['vector']))
                    for _ in range(TrainingParameters.N_EPOCHS):
                        np.random.shuffle(inds)
                        for start in range(0, len(training_data['vector']), TrainingParameters.MINIBATCH_SIZE):
                            end = min(start + TrainingParameters.MINIBATCH_SIZE, len(training_data['vector']))
                            mb_inds = inds[start:end]
                            mb_loss.append(training_model.train(
                                training_data['vector'][mb_inds], 
                                training_data['returns'][mb_inds], 
                                training_data['values'][mb_inds], 
                                training_data['actions'][mb_inds],
                                training_data['ps'][mb_inds], 
                                training_data['hidden'][mb_inds]
                            ))

                    # Record training results
                    agent_name = TrainingParameters.AGENT_TO_TRAIN.capitalize()
                    if RecordingParameters.WANDB:
                        # Log performance metrics
                        for key, value in performance_dict.items():
                            wandb.log({f'{agent_name}_{key}': value}, step=curr_steps)
                        
                        # Log training loss metrics
                        if mb_loss:
                            loss_vals = np.nanmean(mb_loss, axis=0)
                            for i, name in enumerate(RecordingParameters.LOSS_NAME):
                                wandb.log({f'{agent_name}_{name}': loss_vals[i]}, step=curr_steps)
                                
                    if RecordingParameters.TENSORBOARD:
                        # Log performance and loss metrics to tensorboard
                        for key, value in performance_dict.items():
                            global_summary.add_scalar(f'{agent_name}_{key}', value, curr_steps)
                        
                        if mb_loss:
                            loss_vals = np.nanmean(mb_loss, axis=0)
                            for i, name in enumerate(RecordingParameters.LOSS_NAME):
                                if name == 'grad_norm':
                                    global_summary.add_scalar(f'{agent_name}_Grad/{name}', loss_vals[i], curr_steps)
                                else:
                                    global_summary.add_scalar(f'{agent_name}_Loss/{name}', loss_vals[i], curr_steps)

            # Evaluation
            if (curr_steps - last_test_t) / RecordingParameters.EVAL_INTERVAL >= 1.0:
                if (curr_steps - last_gif_t) / RecordingParameters.GIF_INTERVAL >= 1.0:
                    save_gif = True
                    last_gif_t = curr_steps
                else:
                    save_gif = False

                last_test_t = curr_steps
                with torch.no_grad():
                    # Evaluate the model
                    eval_performance_dict = evaluate_single_agent(
                        eval_env, 
                        training_model, 
                        opponent_model,
                        global_device, 
                        save_gif, 
                        curr_steps
                    )
                
                agent_name = TrainingParameters.AGENT_TO_TRAIN.capitalize()
                if RecordingParameters.WANDB:
                    for key, value in eval_performance_dict.items():
                        wandb.log({f'{agent_name}_Eval_{key}': value}, step=curr_steps)
                        
                if RecordingParameters.TENSORBOARD:
                    for key, value in eval_performance_dict.items():
                        global_summary.add_scalar(f'{agent_name}_Eval_{key}', value, curr_steps)

                print('episodes: {}, step: {}, episode reward: {:.2f}\n'.format(
                    curr_episodes, curr_steps, eval_performance_dict.get('per_r', 0)))
                    
                # save model with the best performance
                if RecordingParameters.RECORD_BEST:
                    current_perf = eval_performance_dict.get('per_r', 0)
                    if current_perf > best_perf and (
                            curr_steps - last_best_t) / RecordingParameters.BEST_INTERVAL >= 1.0:
                        best_perf = current_perf
                        last_best_t = curr_steps
                        print('Saving best model \n')
                        model_path = osp.join(RecordingParameters.MODEL_PATH, 'best_model')
                        if not os.path.exists(model_path):
                            os.makedirs(model_path)
                        
                        # Save the trained model
                        save_path = model_path + f"/{TrainingParameters.AGENT_TO_TRAIN}_net_checkpoint.pkl"
                        
                        checkpoint = {"model": training_model.network.state_dict(),
                                     "optimizer": training_model.net_optimizer.state_dict(),
                                     "step": curr_steps, "episode": curr_episodes, "reward": best_perf}
                        
                        torch.save(checkpoint, save_path)

            # save model periodically
            if (curr_steps - last_model_t) / RecordingParameters.SAVE_INTERVAL >= 1.0:
                last_model_t = curr_steps
                print('Saving Model !\n')
                model_path = osp.join(RecordingParameters.MODEL_PATH, '%.5i' % curr_steps)
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                
                # Save the trained model
                save_path = model_path + f"/{TrainingParameters.AGENT_TO_TRAIN}_net_checkpoint.pkl"
                
                checkpoint = {"model": training_model.network.state_dict(),
                             "optimizer": training_model.net_optimizer.state_dict(),
                             "step": curr_steps, "episode": curr_episodes, "reward": 0}
                
                torch.save(checkpoint, save_path)

    except KeyboardInterrupt:
        print("CTRL-C pressed. killing remote workers")
    finally:
        # save final model
        print('Saving Final Model !\n')
        model_path = RecordingParameters.MODEL_PATH + '/final'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        # Save the trained model
        save_path = model_path + f"/{TrainingParameters.AGENT_TO_TRAIN}_net_checkpoint.pkl"
        
        checkpoint = {"model": training_model.network.state_dict(),
                     "optimizer": training_model.net_optimizer.state_dict(),
                     "step": curr_steps, "episode": curr_episodes, "reward": 0}
        
        torch.save(checkpoint, save_path)
        
        if RecordingParameters.TENSORBOARD:
            global_summary.close()
        # killing
        for e in envs:
            ray.kill(e)
        if RecordingParameters.WANDB:
            wandb.finish()


def evaluate_single_agent(eval_env, agent_model, opponent_model, device, save_gif, curr_steps):
    """Single-agent evaluation function"""
    eval_performance_dict = {'per_r': [], 'per_ex_r': [], 'per_in_r': [], 'per_valid_rate': [], 
                             'per_episode_len': [], 'rewarded_rate': []}
    episode_frames = []

    for i in range(RecordingParameters.EVAL_EPISODES):
        # reset environment
        agent_hidden = None
        opponent_hidden = None
        
        obs, _ = eval_env.reset()
        done = False
        episode_step = 0
        episode_reward = 0
        
        if save_gif:
            try:
                frame = eval_env.render(mode='rgb_array')
                if frame is not None:
                    episode_frames.append(frame)
            except Exception as e:
                print(f"Error capturing frame: {e}")

        # stepping
        while not done and episode_step < EnvParameters.EPISODE_LEN:
            # Get action from our agent
            if TrainingParameters.AGENT_TO_TRAIN == "tracker":
                agent_action, agent_hidden, _, _ = agent_model.evaluate(obs, agent_hidden, greedy=True)
                
                # Get opponent (target) action
                if opponent_model is not None:
                    # Using policy opponent
                    opponent_action, opponent_hidden, _, _ = opponent_model.evaluate(obs, opponent_hidden, greedy=True)
                    tracker_action, target_action = agent_action, opponent_action
                else:
                    # Using rule-based opponent - let environment handle it
                    tracker_action, target_action = agent_action, None
            else:
                # Training target agent
                agent_action, agent_hidden, _, _ = agent_model.evaluate(obs, agent_hidden, greedy=True)
                
                # Get opponent (tracker) action
                if opponent_model is not None:
                    # Using policy opponent
                    opponent_action, opponent_hidden, _, _ = opponent_model.evaluate(obs, opponent_hidden, greedy=True)
                    tracker_action, target_action = opponent_action, agent_action
                else:
                    # Using rule-based opponent - let environment handle it
                    tracker_action, target_action = None, agent_action
            
            # Move
            try:
                obs, reward, terminated, truncated, info = eval_env.step(tracker_action, target_action)
                done = terminated or truncated
            except Exception as e:
                print(f"Error in step: {e}")
                # Fallback
                obs, reward, done = obs, 0, True
            
            episode_step += 1
            episode_reward += reward
            
            if save_gif:
                try:
                    frame = eval_env.render(mode='rgb_array')
                    if frame is not None:
                        episode_frames.append(frame)
                except Exception as e:
                    print(f"Error capturing frame: {e}")

        # save gif
        if save_gif and episode_frames:
            try:
                if not os.path.exists(RecordingParameters.GIFS_PATH):
                    os.makedirs(RecordingParameters.GIFS_PATH)
                images = np.array(episode_frames)
                agent_name = TrainingParameters.AGENT_TO_TRAIN
                opponent_type = TrainingParameters.OPPONENT_TYPE
                gif_path = '{}/{}_{}_steps_{:d}_reward{:.1f}.gif'.format(
                    RecordingParameters.GIFS_PATH, agent_name, opponent_type, curr_steps, episode_reward)
                make_gif(images, gif_path)
                save_gif = False
            except Exception as e:
                print(f"Failed to save gif: {e}")

        # Update performance statistics
        eval_performance_dict['per_r'].append(episode_reward)
        eval_performance_dict['per_ex_r'].append(reward if 'reward' in locals() else 0)
        eval_performance_dict['per_in_r'].append(0)  
        eval_performance_dict['per_valid_rate'].append(1.0)
        eval_performance_dict['per_episode_len'].append(episode_step)
        eval_performance_dict['rewarded_rate'].append(0)

    # average performance
    for key in eval_performance_dict.keys():
        if eval_performance_dict[key]:
            eval_performance_dict[key] = np.nanmean(eval_performance_dict[key])
        else:
            eval_performance_dict[key] = 0.0

    return eval_performance_dict


if __name__ == "__main__":
    main()