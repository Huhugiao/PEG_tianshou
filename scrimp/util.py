import random
import imageio
import numpy as np
import torch
import wandb

from alg_parameters import *

def set_global_seeds(i):
    """Set seed for fair comparison"""
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    np.random.seed(i)
    random.seed(i)
    torch.backends.cudnn.deterministic = True

def write_to_tensorboard(global_summary, step, performance_dict=None, mb_loss=None, imitation_loss=None, evaluate=True,
                         greedy=True):
    """Record performance using tensorboard"""
    if imitation_loss is not None:
        global_summary.add_scalar(tag='Loss/Imitation_loss', scalar_value=imitation_loss[0], global_step=step)
        global_summary.add_scalar(tag='Grad/Imitation_grad', scalar_value=imitation_loss[1], global_step=step)
        global_summary.flush()
        return
        
    if evaluate:
        if greedy:
            global_summary.add_scalar(tag='Perf_greedy_eval/Reward', scalar_value=performance_dict['per_r'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/In_Reward', scalar_value=performance_dict['per_in_r'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Ex_Reward', scalar_value=performance_dict['per_ex_r'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Valid_rate', scalar_value=performance_dict['per_valid_rate'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Episode_length', scalar_value=performance_dict['per_episode_len'], global_step=step)
            
        else:
            global_summary.add_scalar(tag='Perf_random_eval/Reward', scalar_value=performance_dict['per_r'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/In_Reward', scalar_value=performance_dict['per_in_r'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Ex_Reward', scalar_value=performance_dict['per_ex_r'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Valid_rate', scalar_value=performance_dict['per_valid_rate'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Episode_length', scalar_value=performance_dict['per_episode_len'], global_step=step)
            
    else:
        loss_vals = np.nanmean(mb_loss, axis=0)
        global_summary.add_scalar(tag='Perf/Reward', scalar_value=performance_dict['per_r'], global_step=step)
        global_summary.add_scalar(tag='Perf/In_Reward', scalar_value=performance_dict['per_in_r'], global_step=step)
        global_summary.add_scalar(tag='Perf/Ex_Reward', scalar_value=performance_dict['per_ex_r'], global_step=step)
        global_summary.add_scalar(tag='Perf/Valid_rate', scalar_value=performance_dict['per_valid_rate'], global_step=step)
        global_summary.add_scalar(tag='Perf/Episode_length', scalar_value=performance_dict['per_episode_len'], global_step=step)
        global_summary.add_scalar(tag='Perf/Rewarded_rate', scalar_value=performance_dict['rewarded_rate'], global_step=step)

        for (val, name) in zip(loss_vals, RecordingParameters.LOSS_NAME):
            if name == 'grad_norm':
                global_summary.add_scalar(tag='Grad/' + name, scalar_value=val, global_step=step)
            else:
                global_summary.add_scalar(tag='Loss/' + name, scalar_value=val, global_step=step)

    global_summary.flush()

def write_to_wandb(step, performance_dict=None, mb_loss=None, imitation_loss=None, evaluate=True, greedy=True):
    """Record performance using wandb"""
    if imitation_loss is not None:
        wandb.log({'Loss/Imitation_loss': imitation_loss[0]}, step=step)
        wandb.log({'Grad/Imitation_grad': imitation_loss[1]}, step=step)
        return
        
    if evaluate:
        if greedy:
            wandb.log({'Perf_greedy_eval/Reward': performance_dict['per_r']}, step=step)
            wandb.log({'Perf_greedy_eval/In_Reward': performance_dict['per_in_r']}, step=step)
            wandb.log({'Perf_greedy_eval/Ex_Reward': performance_dict['per_ex_r']}, step=step)
            wandb.log({'Perf_greedy_eval/Valid_rate': performance_dict['per_valid_rate']}, step=step)
            wandb.log({'Perf_greedy_eval/Episode_length': performance_dict['per_episode_len']}, step=step)
            
        else:
            wandb.log({'Perf_random_eval/Reward': performance_dict['per_r']}, step=step)
            wandb.log({'Perf_random_eval/In_Reward': performance_dict['per_in_r']}, step=step)
            wandb.log({'Perf_random_eval/Ex_Reward': performance_dict['per_ex_r']}, step=step)
            wandb.log({'Perf_random_eval/Valid_rate': performance_dict['per_valid_rate']}, step=step)
            wandb.log({'Perf_random_eval/Episode_length': performance_dict['per_episode_len']}, step=step)
            
    else:
        loss_vals = np.nanmean(mb_loss, axis=0)
        wandb.log({'Perf/Reward': performance_dict['per_r']}, step=step)
        wandb.log({'Perf/In_Reward': performance_dict['per_in_r']}, step=step)
        wandb.log({'Perf/Ex_Reward': performance_dict['per_ex_r']}, step=step)
        wandb.log({'Perf/Valid_rate': performance_dict['per_valid_rate']}, step=step)
        wandb.log({'Perf/Episode_length': performance_dict['per_episode_len']}, step=step)
        wandb.log({'Perf/Rewarded_rate': performance_dict['rewarded_rate']}, step=step)

        for (val, name) in zip(loss_vals, RecordingParameters.LOSS_NAME):
            if name == 'grad_norm':
                wandb.log({'Grad/' + name: val}, step=step)
            else:
                wandb.log({'Loss/' + name: val}, step=step)

def make_gif(images, file_name):
    """Record gif"""
    imageio.mimwrite(file_name, images, subrectangles=True)
    print("Wrote gif")

def update_perf(one_episode_perf, performance_dict):
    """Record batch performance"""
    performance_dict['per_ex_r'].append(one_episode_perf['ex_reward'])
    performance_dict['per_in_r'].append(one_episode_perf['in_reward'])
    performance_dict['per_r'].append(one_episode_perf['episode_reward'])
    performance_dict['per_valid_rate'].append(
        ((one_episode_perf['num_step'] * 2) - one_episode_perf['invalid']) / (
                one_episode_perf['num_step'] * 2))
    performance_dict['per_episode_len'].append(one_episode_perf['num_step'])
    performance_dict['rewarded_rate'].append(one_episode_perf['reward_count'] / (one_episode_perf['num_step'] * 2))
    return performance_dict