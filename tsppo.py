from typing import Any, Dict, List, Optional, Type, Sequence, Union, Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.policy import A2CPolicy
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.common import MLP

class PPOPolicy(A2CPolicy):

    def __init__(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        eps_clip: float = 0.2,
        dual_clip: Optional[float] = None,
        value_clip: bool = False,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(actor, critic, optim, dist_fn, **kwargs)
        self._eps_clip = eps_clip
        assert dual_clip is None or dual_clip > 1.0, \
            "Dual-clip PPO parameter should greater than 1.0."
        self._dual_clip = dual_clip
        self._value_clip = value_clip
        if not self._rew_norm:
            assert not self._value_clip, \
                "value clip is available only when `reward_normalization` is True"
        self._norm_adv = advantage_normalization
        self._recompute_adv = recompute_advantage
        self._actor_critic: ActorCritic

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        if self._recompute_adv:
            # buffer input `buffer` and `indices` to be used in `learn()`.
            self._buffer, self._indices = buffer, indices
        batch = self._compute_returns(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)
        old_log_prob = []
        with torch.no_grad():
            for minibatch in batch.split(self._batch, shuffle=False, merge_last=True):
                old_log_prob.append(self(minibatch).dist.log_prob(minibatch.act))
        batch.logp_old = torch.cat(old_log_prob, dim=0)
        return batch

    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
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
                value = self.critic(minibatch.obs, god_view=minibatch.info.god_view_info).flatten()
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


    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        # 结合 god-view 信息
        if 'god_view' in batch.info:
            combined_input = torch.cat([batch.obs, batch.info['god_view']], dim=-1)
        else:
            combined_input = batch.obs
            
        # 调用actor计算logits和隐藏状态
        logits, hidden = self.actor(combined_input, state=state)

        # 根据logits创建分布
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        
        # 确定动作，根据环境决定是确定还是随机抽样
        if self._deterministic_eval and not self.training:
            if self.action_type == "discrete":
                act = logits.argmax(dim=-1)
            elif self.action_type == "continuous":
                act = logits.mean  # 对于连续空间，使用均值作为确定动作
        else:
            act = dist.sample()

        return Batch(logits=logits, act=act, state=hidden, dist=dist)
    
    def _compute_returns(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        v_s, v_s_ = [], []
        with torch.no_grad():
            for minibatch in batch.split(self._batch, shuffle=False, merge_last=True):
                god_view = minibatch.info.god_view_info
                v_s.append(self.critic(minibatch.obs, god_view=god_view))
                v_s_.append(self.critic(minibatch.obs_next, god_view=god_view))
        batch.v_s = torch.cat(v_s, dim=0).flatten()  # old value
        v_s = batch.v_s.cpu().numpy()
        v_s_ = torch.cat(v_s_, dim=0).flatten().cpu().numpy()
        # when normalizing values, we do not minus self.ret_rms.mean to be numerically
        # consistent with OPENAI baselines' value normalization pipeline. Emperical
        # study also shows that "minus mean" will harm performances a tiny little bit
        # due to unknown reasons (on Mujoco envs, not confident, though).
        if self._rew_norm:  # unnormalize v_s & v_s_
            v_s = v_s * np.sqrt(self.ret_rms.var + self._eps)
            v_s_ = v_s_ * np.sqrt(self.ret_rms.var + self._eps)
        unnormalized_returns, advantages = self.compute_episodic_return(
            batch,
            buffer,
            indices,
            v_s_,
            v_s,
            gamma=self._gamma,
            gae_lambda=self._lambda
        )
        if self._rew_norm:
            batch.returns = unnormalized_returns / \
                np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        batch.returns = to_torch_as(batch.returns, batch.v_s)
        batch.adv = to_torch_as(advantages, batch.v_s)
        return batch

class Actor(nn.Module):
    """Simple actor network.
    Will create an actor operated in discrete action space with structure of
    preprocess_net ---> action_shape.
    :param preprocess_net: a self-defined preprocess_net which output a
    flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
    preprocess_net. Default to empty sequence (where the MLP now contains
    only a single linear layer).
    :param bool softmax_output: whether to apply a softmax layer over the last
    layer's output.
    :param int preprocess_net_output_dim: the output dimension of
    preprocess_net.
    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    .. seealso::
    Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
    of how preprocess_net is suggested to be defined.
    """
    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: Sequence[int],
        hidden_sizes: Sequence[int] = (),
        softmax_output: bool = True,
        preprocess_net_output_dim: Optional[int] = None,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = int(np.prod(action_shape))
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.last = MLP(
            input_dim,  # type: ignore
            self.output_dim,
            hidden_sizes,
            device=self.device
        )
        self.softmax_output = softmax_output

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
        god_view: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        if god_view is not None:
            god_view = torch.tensor(god_view).to(obs.device)
            obs = torch.cat([obs, god_view], dim=-1)
        logits, hidden = self.preprocess(obs, state)
        logits = self.last(logits)
        if self.softmax_output:
            logits = F.softmax(logits, dim=-1)
        return logits, hidden

class Critic(nn.Module):

    def __init__(
        self,
        preprocess_net: nn.Module,
        hidden_sizes: Sequence[int] = (),
        last_size: int = 1,
        preprocess_net_output_dim: Optional[int] = None,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = last_size
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.last = MLP(
            input_dim,  # type: ignore
            last_size,
            hidden_sizes,
            device=self.device
        )

    def forward(
        self, obs: Union[np.ndarray, torch.Tensor], 
        state: Optional[Union[np.ndarray, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
        god_view: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Mapping: s -> V(s)."""
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        god_view = torch.tensor(god_view, dtype=torch.float32).to(self.device)
        obs = torch.cat([obs, god_view], dim=-1)
        logits, _ = self.preprocess(obs, state=state)
        return self.last(logits).flatten()
