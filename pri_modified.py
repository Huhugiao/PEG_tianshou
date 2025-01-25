import numpy as np
import torch
from torch import nn
from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.policy import PPOPolicy
from typing import Any, Dict, List, Optional, Type

class pri_PPOPolicy(PPOPolicy):
    def __init__(self, actor, critic, optim, dist_fn, **kwargs):
        super().__init__(actor, critic, optim, dist_fn, **kwargs)

    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray) -> Batch:
        if self._recompute_adv:
            self._buffer, self._indices = buffer, indices
        batch = self._compute_returns(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)
        old_log_prob = []
        with torch.no_grad():
            for minibatch in batch.split(self._batch, shuffle=False, merge_last=True):
                old_log_prob.append(self(minibatch).dist.log_prob(minibatch.act))
        batch.logp_old = torch.cat(old_log_prob, dim=0)
        return batch

    def learn(self, batch: Batch, batch_size: int, repeat: int, **kwargs) -> Dict[str, List[float]]:
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        for step in range(repeat):
            if self._recompute_adv and step > 0:
                batch = self._compute_returns(batch, self._buffer, self._indices)
            for minibatch in batch.split(batch_size, merge_last=True):
                # calculate loss for actor
                dist = self(minibatch).dist
                if self._norm_adv:
                    mean, std = minibatch.adv.mean(), minibatch.adv.std()
                    minibatch.adv = (minibatch.adv - mean) / std  # per-batch norm
                ratio = (dist.log_prob(minibatch.act) -
                         minibatch.logp_old).exp().float()
                ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                surr1 = ratio * minibatch.adv
                surr2 = ratio.clamp(
                    1.0 - self._eps_clip, 1.0 + self._eps_clip
                ) * minibatch.adv
                if self._dual_clip:
                    clip1 = torch.min(surr1, surr2)
                    clip2 = torch.max(clip1, self._dual_clip * minibatch.adv)
                    clip_loss = -torch.where(minibatch.adv < 0, clip2, clip1).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()
                # calculate loss for critic
                value = self.critic(minibatch.obs, minibatch.godview).flatten()
                if self._value_clip:
                    v_clip = minibatch.v_s + \
                        (value - minibatch.v_s).clamp(-self._eps_clip, self._eps_clip)
                    vf1 = (minibatch.returns - value).pow(2)
                    vf2 = (minibatch.returns - v_clip).pow(2)
                    vf_loss = torch.max(vf1, vf2).mean()
                else:
                    vf_loss = (minibatch.returns - value).pow(2).mean()
                # calculate regularization and overall loss
                ent_loss = dist.entropy().mean()
                loss = clip_loss + self._weight_vf * vf_loss \
                    - self._weight_ent * ent_loss
                self.optim.zero_grad()
                loss.backward()
                if self._grad_norm:  # clip large gradient
                    nn.utils.clip_grad_norm_(
                        self._actor_critic.parameters(), max_norm=self._grad_norm
                    )
                self.optim.step()
                clip_losses.append(clip_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())

        return {
            "loss": losses,
            "loss/clip": clip_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
        }



# 处理观测值和信息
def preprocess_fn(
    obs=None, 
    obs_next=None, 
    rew=None, 
    done=None, 
    terminated=None, 
    truncated=None, 
    info=None, 
    policy=None, 
    env_id=None
):
    # 处理 godview 信息
    godview = None
    if info is not None and env_id is not None:
        # 确保 env_id 是一个数组或列表
        env_id = np.asarray(env_id)
        # 确保 info 是一个列表或字典
        if isinstance(info, (list, np.ndarray)):
            # 检查 env_id 是否在 info 的索引范围内
            if len(info) > 0 and isinstance(info[0], dict):
                godview = [info[idx].get("god_view_info", None) if idx < len(info) else None for idx in env_id]
        elif isinstance(info, dict):
            # 如果 info 是单个字典，直接获取 god_view_info
            godview = info.get("god_view_info", None)

    # 处理观测值
    processed_obs = obs if obs is not None else obs_next

    # 处理信息
    processed_info = info

    # 返回处理后的数据
    processed_data = {
        "obs": processed_obs, 
        "obs_next": obs_next,
        "rew": rew,
        "done": done,
        "terminated": terminated,
        "truncated": truncated,
        "info": processed_info,
        "policy": policy,
        "godview": godview,
        "env_id": env_id
    }

    # 过滤掉 None 值
    processed_data = {k: v for k, v in processed_data.items() if v is not None}

    return processed_data
