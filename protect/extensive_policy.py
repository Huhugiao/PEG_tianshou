from net import Recurrent, Actor, Critic
import algo_config, torch, gym
import numpy as np
from torch import nn
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import Net, ActorCritic
from tianshou.data import Batch, ReplayBuffer
from gym.spaces import MultiDiscrete
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


def policy_maker():
    env = gym.make(algo_config.task)
    algo_config.state_shape = env.observation_space.shape
    algo_config.action_shape = env.action_space.n

    actor_net = Recurrent(
        layer_num=1,
        state_shape=algo_config.base_obs_dim,
        action_shape=algo_config.action_shape,
        device=algo_config.device,
        hidden_layer_size=32,
    )

    critic_net = Recurrent(
        layer_num=1,
        state_shape=algo_config.base_obs_dim,
        action_shape=algo_config.action_shape,
        device=algo_config.device,
        hidden_layer_size=32,
    )

    actor = Actor(actor_net, algo_config.action_shape, device=algo_config.device)
    critic = Critic(critic_net, device=algo_config.device)

    tracker_net = ActorCritic(actor, critic)
    target_net = ActorCritic(actor, critic)

    # Initialize networks
    for model in [tracker_net, target_net]:
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)
    
    # Create optimizers
    optim = torch.optim.Adam(list(tracker_net.parameters()) + list(critic_net.parameters()), lr=algo_config.lr)
    dist_fn = torch.distributions.Categorical

    policy = PPOPolicy(
        actor_tracker,
        critic_tracker,
        optim,
        dist_fn,
        discount_factor=algo_config.gamma,
        max_grad_norm=algo_config.max_grad_norm,
        eps_clip=algo_config.eps_clip,
        vf_coef=algo_config.vf_coef,
        ent_coef=algo_config.ent_coef,
        gae_lambda=algo_config.gae_lambda,
        reward_normalization=algo_config.reward_normalization,
        dual_clip=algo_config.dual_clip,
        value_clip=algo_config.value_clip,
        advantage_normalization=algo_config.norm_adv,
        recompute_advantage=algo_config.recompute_adv,
        action_space=env.action_space,
        deterministic_eval=False,
    ).to(algo_config.device)

    return policy, optim