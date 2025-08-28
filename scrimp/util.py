import os
import os.path as osp
import random
import numpy as np
import torch
from typing import Dict, List, Optional

try:
    import imageio
except Exception:
    imageio = None

try:
    import wandb
except Exception:
    wandb = None

from alg_parameters import *


def set_global_seeds(i: int):
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    np.random.seed(i)
    random.seed(i)
    torch.backends.cudnn.deterministic = True


def _avg(vals):
    if vals is None:
        return None
    if isinstance(vals, (list, tuple)) and len(vals) > 0 and isinstance(vals[0], (list, tuple, np.ndarray)):
        return np.nanmean(vals, axis=0)
    if isinstance(vals, (list, tuple, np.ndarray)):
        return float(np.nanmean(vals)) if len(vals) > 0 else 0.0
    return vals


def write_to_tensorboard(global_summary, step: int, performance_dict: Optional[Dict] = None,
                         mb_loss: Optional[List] = None, imitation_loss: Optional[List] = None,
                         evaluate: bool = True, greedy: bool = True):
    if global_summary is None:
        return

    if imitation_loss is not None:
        global_summary.add_scalar('Loss/Imitation_loss', imitation_loss[0], step)
        if len(imitation_loss) > 1:
            global_summary.add_scalar('Grad/Imitation_grad', imitation_loss[1], step)

    # Performance
    if performance_dict:
        prefix = 'Eval' if evaluate else 'Train'
        for k, v in performance_dict.items():
            val = _avg(v)
            if val is not None:
                global_summary.add_scalar(f'{prefix}/{k}', val, step)

    # Loss
    if mb_loss:
        loss_vals = np.nanmean(np.asarray(mb_loss, dtype=np.float32), axis=0)
        names = getattr(RecordingParameters, 'LOSS_NAME', [
            'total', 'policy', 'entropy', 'value', 'value_aux', 'aux1', 'aux2', 'clipfrac', 'grad_norm', 'adv_mean'
        ])
        for i, name in enumerate(names[:len(loss_vals)]):
            tag = f'Loss/{name}' if 'grad' not in name else f'Grad/{name}'
            global_summary.add_scalar(tag, float(loss_vals[i]), step)

    global_summary.flush()


def write_to_wandb(step: int, performance_dict: Optional[Dict] = None,
                   mb_loss: Optional[List] = None, imitation_loss: Optional[List] = None,
                   evaluate: bool = True, greedy: bool = True):
    if wandb is None or not getattr(RecordingParameters, 'WANDB', False) or getattr(wandb, 'run', None) is None:
        return

    log_data = {}
    if imitation_loss is not None:
        log_data['Loss/Imitation_loss'] = imitation_loss[0]
        if len(imitation_loss) > 1:
            log_data['Grad/Imitation_grad'] = imitation_loss[1]

    if performance_dict:
        prefix = 'Eval' if evaluate else 'Train'
        for k, v in performance_dict.items():
            val = _avg(v)
            if val is not None:
                log_data[f'{prefix}/{k}'] = val

    if mb_loss:
        loss_vals = np.nanmean(np.asarray(mb_loss, dtype=np.float32), axis=0)
        names = getattr(RecordingParameters, 'LOSS_NAME', [
            'total', 'policy', 'entropy', 'value', 'value_aux', 'aux1', 'aux2', 'clipfrac', 'grad_norm', 'adv_mean'
        ])
        for i, name in enumerate(names[:len(loss_vals)]):
            tag = f'Loss/{name}' if 'grad' not in name else f'Grad/{name}'
            log_data[tag] = float(loss_vals[i])

    if log_data:
        wandb.log(log_data, step=step)


def make_gif(images, file_name, fps=20):
    if imageio is None:
        print("imageio not available, skip gif")
        return
    if isinstance(images, list):
        frames = [np.asarray(img, dtype=np.uint8) for img in images]
    else:
        frames = np.asarray(images, dtype=np.uint8)
    os.makedirs(osp.dirname(file_name), exist_ok=True)
    imageio.mimwrite(file_name, frames, duration=1.0 / max(int(fps), 1), loop=0)
    print(f"Wrote gif: {file_name}")


def update_perf(one_episode_perf: Dict, performance_dict: Dict):
    performance_dict['per_ex_r'].append(one_episode_perf.get('ex_reward', 0.0))
    performance_dict['per_in_r'].append(one_episode_perf.get('in_reward', 0.0))
    performance_dict['per_r'].append(one_episode_perf.get('episode_reward', 0.0))
    num_step = max(int(one_episode_perf.get('num_step', 1)), 1)
    invalid = int(one_episode_perf.get('invalid', 0))
    reward_count = int(one_episode_perf.get('reward_count', 0))
    performance_dict['per_valid_rate'].append(((num_step * 2) - invalid) / (num_step * 2))
    performance_dict['per_episode_len'].append(num_step)
    performance_dict['rewarded_rate'].append(reward_count / (num_step * 2))
    return performance_dict