from net import Recurrent, Actor, Critic
import algo_config, torch, gym
import numpy as np
from torch import nn
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import Net, ActorCritic
from tianshou.data import Batch, ReplayBuffer
from gym.spaces import MultiDiscrete
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class Policies(nn.Module):
    def __init__(self, policy_a: PPOPolicy, policy_b: PPOPolicy) -> None:
        super().__init__()
        self.policy_a = policy_a  
        self.policy_b = policy_b 
        self.active_policy = "a"
        self.mission = 0  # Default mission
        
    def set_mission(self, mission: int):
        """Set the current mission type"""
        self.mission = mission
    
    def set_active_policy(self, policy_name: str):
        """Set which policy is being trained (a or b)"""
        if policy_name in ["a", "b"]:
            self.active_policy = policy_name
        else:
            raise ValueError("policy_name must be 'a' or 'b'")
    
    def forward(self, batch, state=None, **kwargs):
        """获取两个智能体的动作，根据当前激活的策略决定输出形式"""
        # 始终先算出 A/B 两套结果
        result_a = self.policy_a(batch, state, **kwargs)
        with torch.no_grad():
            result_b = self.policy_b(batch, state)

        # 如果当前是tracker激活状态，则使用A的结果作为主要结果
        if self.active_policy == "a":
            result = result_a
        else:
            result = result_b
        
        # 无论是什么模式，都计算组合动作（确保双智能体功能正常）
        result.tracker_act = result_a.act
        result.target_act = result_b.act
        result.act = result_a.act + result_b.act / 100.0
        
        return result
    
    def process_fn(self, batch, buffer, indices):
        """Process function delegates to active policy, but handles encoded actions"""
        # 如果是双智能体模式，先进行动作解码
        if algo_config.mission != 0:
            if isinstance(batch.act, torch.Tensor):
                # 不能创建新batch，而是在现有batch上修改
                if self.active_policy == "a":  # tracker
                    # 提取整数部分作为tracker动作
                    batch.act = batch.act.floor().to(torch.int64)
                else:  # target (policy_b)
                    # 提取小数部分*100作为target动作
                    batch.act = ((batch.act - batch.act.floor()) * 100).round().to(torch.int64)
        
        # 调用原始策略的process_fn来计算优势函数等
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
        self.policy_a.load_state_dict(state_dict["policy_a"])
        self.policy_b.load_state_dict(state_dict["policy_b"])

    def exploration_noise(self, act, batch):
        """将探索噪声应用到动作上，处理单个动作或元组动作"""
        # 处理元组形式动作
        if isinstance(act, tuple) and len(act) == 2:
            tracker_act, target_act = act
            
            # 根据当前活跃策略决定哪个动作添加噪声
            if self.active_policy == "a":
                # tracker活跃时，只给tracker动作添加噪声
                tracker_act_with_noise = self.policy_a.exploration_noise(tracker_act, batch)
                return (tracker_act_with_noise, target_act)
            else:
                # target活跃时，只给target动作添加噪声
                target_act_with_noise = self.policy_b.exploration_noise(target_act, batch)
                return (tracker_act, target_act_with_noise)
        # 默认情况，使用活跃策略处理单个动作
        else:
            if self.active_policy == "a":
                return self.policy_a.exploration_noise(act, batch)
            else:
                return self.policy_b.exploration_noise(act, batch)
    
    def map_action(self, act):
        """
        映射网络输出到环境需要的动作格式：
        - 单智能体：act 为标量、list 或一维 np.ndarray
        - 双智能体：act 为 (arrA, arrB) 或 (scalarA, scalarB)
        返回：
            - 单环境标量：int
            - 多环境列表：长度==env_num 的 list，元素为 int 或 (int,int)
        """
        # 双智能体 tuple
        if isinstance(act, tuple) and len(act) == 2:
            a0, a1 = act
            # torch.Tensor 先转 numpy
            if isinstance(a0, torch.Tensor):
                a0 = a0.cpu().numpy()
                a1 = a1.cpu().numpy()
            # 向量化批量输出：numpy.ndarray 或 list
            if isinstance(a0, (np.ndarray, list)):
                return [(int(a0[i]), int(a1[i])) for i in range(len(a0))]
            # 单环境下的双动作
            return (int(a0), int(a1))

        # 单智能体批量输出：torch.Tensor / np.ndarray / list
        if isinstance(act, torch.Tensor):
            return act.cpu().numpy().tolist()
        if isinstance(act, np.ndarray):
            return act.tolist()
        if isinstance(act, list):
            return act

        # 单环境标量
        return int(act)

    def map_action_inverse(self, act):
        """动作映射的逆操作，处理单个动作或元组动作"""
        # 处理元组形式动作
        if isinstance(act, tuple) and len(act) == 2:
            tracker_act, target_act = act
            
            # 分别对两个智能体的动作进行逆映射
            inv_tracker = self.policy_a.map_action_inverse(tracker_act)
            inv_target = self.policy_b.map_action_inverse(target_act)
            
            # 返回逆映射后的元组
            return (inv_tracker, inv_target)
        
        # 处理numpy数组中的元组动作（向量化环境）
        elif isinstance(act, np.ndarray) and act.size > 0 and isinstance(act[0], tuple):
            inv_actions = []
            for single_act in act:
                inv_actions.append(self.map_action_inverse(single_act))
            return np.array(inv_actions, dtype=object)
            
        # 默认情况，使用活跃策略处理单个动作
        else:
            if self.active_policy == "a":
                return self.policy_a.map_action_inverse(act)
            else:
                return self.policy_b.map_action_inverse(act)
    
    def update(self, sample_size: int, buffer: Optional[ReplayBuffer], **kwargs: Any) -> Dict[str, Any]:
        """使用active_policy处理更新，确保正确的数据流"""
        # 不能直接使用传入的batch和indices，而是要走完整的数据处理流程
        if self.active_policy == "a":
            # 使用原始的policy.update流程处理tracker
            result = self.policy_a.update(sample_size, buffer, **kwargs)
        else:
            # 使用原始的policy.update流程处理target
            result = self.policy_b.update(sample_size, buffer, **kwargs)
        
        return result

class CustomPPOPolicy(PPOPolicy):
    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray) -> Batch:
        """重写process_fn以在原始处理前解码动作"""
        # Store original actions for potential future use
        batch.original_act = batch.act.copy()
        
        # Extract action part based on policy type
        if isinstance(self, type(policy_maker()[0].policy_a)):  # Is tracker policy
            batch.act = np.floor(batch.act).astype(np.int64)
        else:  # Is target policy
            batch.act = np.round((batch.act - np.floor(batch.act)) * 100).astype(np.int64)
                    
        # 调用父类方法完成剩余处理
        return super().process_fn(batch, buffer, indices)

def policy_maker():
    env = gym.make(algo_config.task)
    algo_config.state_shape = env.observation_space.shape
    algo_config.action_shape = env.action_space.n

    # target_net = Recurrent(
    #     layer_num=1,
    #     state_shape=algo_config.base_obs_dim,
    #     action_shape=algo_config.action_shape,
    #     device=algo_config.device,
    #     hidden_layer_size=32,
    # )

    # 为 tracker 和 target 各自的 actor/critic 分别创建 Net
    tracker_actor_net = Net(
        state_shape=algo_config.base_obs_dim,
        action_shape=algo_config.action_shape,
        hidden_sizes=algo_config.hidden_sizes,
        device=algo_config.device
    )
    tracker_critic_net = Net(
        state_shape=algo_config.base_obs_dim,
        action_shape=algo_config.action_shape,
        hidden_sizes=algo_config.hidden_sizes,
        device=algo_config.device
    )
    target_actor_net = Net(
        state_shape=algo_config.base_obs_dim,
        action_shape=algo_config.action_shape,
        hidden_sizes=algo_config.hidden_sizes,
        device=algo_config.device
    )
    target_critic_net = Net(
        state_shape=algo_config.base_obs_dim,
        action_shape=algo_config.action_shape,
        hidden_sizes=algo_config.hidden_sizes,
        device=algo_config.device
    )

    actor_tracker = Actor(tracker_actor_net, algo_config.action_shape, device=algo_config.device)
    critic_tracker = Critic(tracker_critic_net, device=algo_config.device)
    actor_target = Actor(target_actor_net, algo_config.action_shape, device=algo_config.device)
    critic_target = Critic(target_critic_net, device=algo_config.device)

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
    target_policy = CustomPPOPolicy(
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

    tracker_policy = CustomPPOPolicy(
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

if __name__ == "__main__":
    policy, optim_target, optim_tracker = policy_maker()
    print (policy)