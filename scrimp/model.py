import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler

from alg_parameters import *
from nets import ProtectingNet


class Model(object):
    """Standard PPO model for Protecting environment - single agent"""

    # 离散索引 <-> (角度, 速度因子) 映射，用于训练与采样
    ANGLE_OFFSETS = np.linspace(-45.0, 45.0, 16, dtype=np.float32)  # 16个角度bin
    SPEED_FACTORS = np.array([0.0, 0.5, 1.0], dtype=np.float32)     # 3个速度档

    @staticmethod
    def idx_to_pair(action_idx: int):
        aidx = int(action_idx) // 3
        sidx = int(action_idx) % 3
        angle = float(Model.ANGLE_OFFSETS[int(np.clip(aidx, 0, 15))])
        speed_factor = float(Model.SPEED_FACTORS[int(np.clip(sidx, 0, 2))])
        return (angle, speed_factor)

    @staticmethod
    def pair_to_idx(angle_delta_deg: float, speed_factor: float):
        # 裁切
        angle = float(np.clip(angle_delta_deg, -45.0, 45.0))
        sf = float(np.clip(speed_factor, 0.0, 1.0))
        # 角度bin（-45~45，共16个，约6度一档）
        direction_step = 90.0 / 15.0  # 6度
        dir_idx = int(np.clip(int(round(angle / direction_step)) + 8, 0, 15))
        # 速度bin就近映射到[0, 0.5, 1.0]
        sidx = int(np.argmin(np.abs(Model.SPEED_FACTORS - sf)))
        return int(dir_idx * 3 + sidx)

    def __init__(self, device, global_model=False):
        self.device = device
        self.network = ProtectingNet().to(device)

        if global_model:
            self.net_optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=TrainingParameters.lr
            )
            self.net_scaler = GradScaler()

        self.network.train()

    def _to_tensor(self, vector):
        if isinstance(vector, np.ndarray):
            input_vector = torch.from_numpy(vector).float().to(self.device)
        elif torch.is_tensor(vector):
            input_vector = vector.to(self.device).float()
        else:
            input_vector = torch.tensor(vector, dtype=torch.float32, device=self.device)
        if input_vector.dim() == 1:
            input_vector = input_vector.unsqueeze(0)
        return input_vector

    @torch.no_grad()
    def step(self, vector, hidden_state=None):
        input_vector = self._to_tensor(vector)
        policy, value, _ = self.network(input_vector)
        prob = policy[0].clamp_min(1e-8)
        prob = prob / prob.sum()
        action_index = int(torch.multinomial(prob, 1).item())
        action_pair = Model.idx_to_pair(action_index)
        return action_pair, None, float(value.squeeze().cpu().numpy()), prob.cpu().numpy(), action_index

    @torch.no_grad()
    def evaluate(self, vector, hidden_state=None, greedy=True):
        input_vector = self._to_tensor(vector)
        policy, value, _ = self.network(input_vector)
        prob = policy[0].clamp_min(1e-8)
        prob = prob / prob.sum()
        action_index = int(torch.argmax(prob).item()) if greedy else int(torch.multinomial(prob, 1).item())
        action_pair = Model.idx_to_pair(action_index)
        return action_pair, None, float(value.squeeze().cpu().numpy()), prob.cpu().numpy(), action_index

    def train(self, vector, returns, values, actions, old_probs, hidden_state, train_valid=None, blocking=None, message=None):
        self.net_optimizer.zero_grad()

        vector = torch.as_tensor(vector, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        values = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        old_probs = torch.as_tensor(old_probs, dtype=torch.float32, device=self.device)

        if vector.dim() == 1:
            vector = vector.unsqueeze(0)
        if values.dim() > 1:
            values = values.squeeze(-1)
        if returns.dim() > 1:
            returns = returns.squeeze(-1)
        if old_probs.dim() == 1:
            old_probs = old_probs.unsqueeze(0)

        advantages = returns - values
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        with autocast():
            new_policy, new_values, logits = self.network(vector)    # [B, A], [B,1], [B,A]
            new_values = new_values.squeeze(-1)
            new_action_probs = new_policy.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            old_action_probs = old_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

            ratio = new_action_probs / (old_action_probs + 1e-8)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - TrainingParameters.CLIP_RANGE, 1.0 + TrainingParameters.CLIP_RANGE) * advantages
            policy_loss = -torch.mean(torch.min(surr1, surr2))

            entropy = -torch.mean(torch.sum(new_policy * torch.log(new_policy + 1e-8), dim=-1))
            value_loss = torch.mean((returns - new_values) ** 2)
            total_loss = policy_loss + TrainingParameters.EX_VALUE_COEF * value_loss - TrainingParameters.ENTROPY_COEF * entropy

        self.net_scaler.scale(total_loss).backward()
        self.net_scaler.unscale_(self.net_optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), TrainingParameters.MAX_GRAD_NORM)
        self.net_scaler.step(self.net_optimizer)
        self.net_scaler.update()

        clipfrac = torch.mean((torch.abs(ratio - 1.0) > TrainingParameters.CLIP_RANGE).float())

        return [float(total_loss.detach().cpu().numpy()),
                float(policy_loss.detach().cpu().numpy()),
                float(entropy.detach().cpu().numpy()),
                float(value_loss.detach().cpu().numpy()),
                float(value_loss.detach().cpu().numpy()),
                0.0,
                0.0,
                float(clipfrac.detach().cpu().numpy()),
                float(grad_norm.detach().cpu().numpy()),
                float(advantages.mean().detach().cpu().numpy())]

    def imitation_train(self, vector, optimal_actions, hidden_state=None):
        self.net_optimizer.zero_grad()

        vector = torch.as_tensor(vector, dtype=torch.float32, device=self.device)
        optimal_actions = torch.as_tensor(optimal_actions, dtype=torch.long, device=self.device)

        if vector.dim() == 1:
            vector = vector.unsqueeze(0)
        if optimal_actions.dim() == 0:
            optimal_actions = optimal_actions.unsqueeze(0)

        with autocast():
            policy, _, logits = self.network(vector)
            imitation_loss = torch.nn.functional.cross_entropy(logits, optimal_actions)

        self.net_scaler.scale(imitation_loss).backward()
        self.net_scaler.unscale_(self.net_optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), TrainingParameters.MAX_GRAD_NORM)
        self.net_scaler.step(self.net_optimizer)
        self.net_scaler.update()

        return [float(imitation_loss.detach().cpu().numpy()),
                float(grad_norm.detach().cpu().numpy())]

    @torch.no_grad()
    def set_weights(self, weights):
        self.network.load_state_dict(weights)

    @torch.no_grad()
    def get_weights(self):
        return self.network.state_dict()

    @torch.no_grad()
    def value(self, vector):
        input_vector = self._to_tensor(vector)
        _, value, _ = self.network(input_vector)
        return float(value.squeeze().cpu().numpy())