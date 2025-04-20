from net import Recurrent, Actor, Critic
import algo_config, torch, gym
from torch import nn
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import Net, ActorCritic

class Policies(nn.Module):
    def __init__(self, policy_a: PPOPolicy, policy_b: PPOPolicy) -> None:
        super().__init__()
        self.policy_a = policy_a  
        self.policy_b = policy_b 
        self.active_policy = "a" 
    
    def set_active_policy(self, policy_name: str):
        """Set which policy is being trained (a or b)"""
        if policy_name in ["a", "b"]:
            self.active_policy = policy_name
        else:
            raise ValueError("policy_name must be 'a' or 'b'")
    
    def forward(self, batch, state=None, **kwargs):
        """获取两个智能体的动作"""
        if self.active_policy == "a":
            # 如果策略A活跃，允许A计算梯度，阻断B的梯度
            result_a = self.policy_a(batch, state, **kwargs)
            with torch.no_grad():  # 不参与梯度计算
                result_b = self.policy_b(batch, state)

            result = result_a
            result.both_actions = (result_a.act, result_b.act) 
        else:
            # 如果策略B活跃，允许B计算梯度，阻断A的梯度
            with torch.no_grad():  # 不参与梯度计算
                result_a = self.policy_a(batch, state, **kwargs)
            result_b = self.policy_b(batch, state)

            result = result_b
            result.both_actions = (result_a.act, result_b.act)
        
        return result
    
    def process_fn(self, batch, buffer, indices):
        """Process function delegates to active policy"""
        if self.active_policy == "a":
            return self.policy_a.process_fn(batch, buffer, indices)
        else:
            return self.policy_b.process_fn(batch, buffer, indices)
    
    def learn(self, batch, batch_size, repeat, **kwargs):
        """Only the active policy learns"""
        if self.active_policy == "a":
            return self.policy_a.learn(batch, batch_size, repeat, **kwargs)
        else:
            return self.policy_b.learn(batch, batch_size, repeat, **kwargs)
    
    def state_dict(self):
        """Return state dict of both policies"""
        return {
            "policy_a": self.policy_a.state_dict(),
            "policy_b": self.policy_b.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict for both policies"""
        if "policy_a" in state_dict and "policy_b" in state_dict:
            self.policy_a.load_state_dict(state_dict["policy_a"])
            self.policy_b.load_state_dict(state_dict["policy_b"])
        else:
            # Legacy support for single policy state dict
            self.policy_a.load_state_dict(state_dict)
            self.policy_b.load_state_dict(state_dict)

    def exploration_noise(self, act, batch):
        """将探索噪声应用到动作上，代理到活跃策略的exploration_noise方法"""
        if self.active_policy == "a":
            return self.policy_a.exploration_noise(act, batch)
        else:
            return self.policy_b.exploration_noise(act, batch)
    
    def map_action(self, act):
        """映射原始网络输出到动作空间范围，代理到活跃策略"""
        if self.active_policy == "a":
            return self.policy_a.map_action(act)
        else:
            return self.policy_b.map_action(act)

    def map_action_inverse(self, act):
        """动作映射的逆操作，代理到活跃策略"""
        if self.active_policy == "a":
            return self.policy_a.map_action_inverse(act)
        else:
            return self.policy_b.map_action_inverse(act)

    def post_process_fn(self, batch, buffer, indices):
        """后处理函数，代理到活跃策略"""
        if self.active_policy == "a":
            return self.policy_a.post_process_fn(batch, buffer, indices)
        else:
            return self.policy_b.post_process_fn(batch, buffer, indices)
    
    def update(self, sample_size: int, buffer=None, **kwargs):
        """更新策略网络和重放缓冲区，代理到活跃策略的update方法"""
        if self.active_policy == "a":
            return self.policy_a.update(sample_size, buffer, **kwargs)
        else:
            return self.policy_b.update(sample_size, buffer, **kwargs)



def policy_maker():
    env = gym.make(algo_config.task)
    algo_config.state_shape = env.observation_space.shape
    algo_config.action_shape = env.action_space.n

    # Create separate networks for target policy
    net_target_a = Recurrent(
        layer_num=1,
        state_shape=algo_config.base_obs_dim,
        action_shape=algo_config.action_shape,
        device=algo_config.device,
        hidden_layer_size=32,
    )

    net_target_c = Recurrent(
        layer_num=1,
        state_shape=algo_config.state_shape,
        action_shape=algo_config.action_shape,
        device=algo_config.device,
        hidden_layer_size=32,
    )

    # Create separate networks for tracker policy
    net_tracker_a = Recurrent(
        layer_num=1,
        state_shape=algo_config.base_obs_dim,
        action_shape=algo_config.action_shape,
        device=algo_config.device,
        hidden_layer_size=32,
    )

    net_tracker_c = Recurrent(
        layer_num=1,
        state_shape=algo_config.state_shape,
        action_shape=algo_config.action_shape,
        device=algo_config.device,
        hidden_layer_size=32,
    )

    actor_target = Actor(net_target_a, algo_config.action_shape, device=algo_config.device)
    critic_target = Critic(net_target_c, device=algo_config.device)
    
    actor_tracker = Actor(net_tracker_a, algo_config.action_shape, device=algo_config.device)
    critic_tracker = Critic(net_tracker_c, device=algo_config.device)

    # Initialize networks
    for model in [actor_target, critic_target, actor_tracker, critic_tracker]:
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)
    
    # Create optimizers
    optim_target = torch.optim.Adam(list(actor_target.parameters()) + list(critic_target.parameters()), lr=algo_config.lr)
    optim_tracker = torch.optim.Adam(list(actor_tracker.parameters()) + list(critic_tracker.parameters()), lr=algo_config.lr)
    
    dist = torch.distributions.Categorical

    # Create separate policies
    target_policy = PPOPolicy(
        actor_target,
        critic_target,
        optim_target,
        dist,
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

    tracker_policy = PPOPolicy(
        actor_tracker,
        critic_tracker,
        optim_tracker,
        dist,
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

    # Create the dual policy wrapper
    policies = Policies(tracker_policy, target_policy)

    return policies, optim_target, optim_tracker