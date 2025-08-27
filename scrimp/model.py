import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler

from alg_parameters import *
from nets import ProtectingNet

class Model(object):
    """Standard PPO model for Protecting environment - single agent"""

    def __init__(self, device, global_model=False):
        """Initialize model"""
        self.device = device
        self.network = ProtectingNet().to(device)
        
        if global_model:
            self.net_optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=TrainingParameters.lr
            )
            self.net_scaler = GradScaler()
        
        self.network.train()

    def step(self, vector, hidden_state=None):
        """Single step inference for data collection"""
        with torch.no_grad():
            # 确保输入是正确的tensor格式
            if isinstance(vector, np.ndarray):
                input_vector = torch.FloatTensor(vector).to(self.device)
            else:
                input_vector = torch.FloatTensor([vector]).to(self.device)
            
            # 确保批次维度
            if input_vector.dim() == 1:
                input_vector = input_vector.unsqueeze(0)
            
            # Forward pass
            policy, value, _ = self.network(input_vector)
            
            # Convert to numpy
            policy = policy.cpu().numpy()
            value = value.cpu().numpy()
            
            # 获取当前智能体的策略
            prob = policy[0].copy()
            # 确保概率和为1且非负
            prob = np.maximum(prob, 1e-8)
            prob = prob / prob.sum()
            
            # 采样动作
            action = np.random.choice(EnvParameters.N_ACTIONS, p=prob)
            
            return action, hidden_state, value, prob

    def evaluate(self, vector, hidden_state=None, greedy=True):
        """Evaluation mode for training evaluation"""
        with torch.no_grad():
            # 处理输入
            if isinstance(vector, np.ndarray):
                input_vector = torch.from_numpy(vector).float().to(self.device)
            else:
                input_vector = torch.FloatTensor([vector]).to(self.device)
            
            if input_vector.dim() == 1:
                input_vector = input_vector.unsqueeze(0)
            
            # Forward pass
            policy, value, _ = self.network(input_vector)
            
            policy = policy.cpu().numpy()
            value = value.cpu().numpy()
            
            # 获取当前智能体的策略
            prob = policy[0].copy()
            prob = np.maximum(prob, 1e-8)
            prob = prob / prob.sum()
            
            if greedy:
                eval_action = np.argmax(prob)
            else:
                eval_action = np.random.choice(EnvParameters.N_ACTIONS, p=prob)
            
            return eval_action, hidden_state, value, policy

    def train(self, vector, returns, values, actions, old_probs, hidden_state, train_valid=None, blocking=None, message=None):
        """Simplified PPO training"""
        self.net_optimizer.zero_grad()
        
        # Convert to tensors
        vector = torch.from_numpy(vector).float().to(self.device)
        returns = torch.from_numpy(returns).float().to(self.device)
        values = torch.from_numpy(values).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        old_probs = torch.from_numpy(old_probs).float().to(self.device)
        
        # 确保维度正确
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)
        if returns.dim() == 0:
            returns = returns.unsqueeze(0)
        if values.dim() > 1:
            values = values.squeeze()
        if returns.dim() > 1:
            returns = returns.squeeze()
        if actions.dim() == 0:
            actions = actions.unsqueeze(0)
        
        # 处理old_probs的维度
        if old_probs.dim() == 1:
            old_probs = old_probs.unsqueeze(0)
            
        # Calculate advantages
        advantages = returns - values
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        with autocast():
            # Forward pass
            new_policy, new_values, _ = self.network(vector)
            
            new_values = new_values.squeeze()
            
            # 计算动作概率 - 现在policy是[batch_size, n_actions]
            new_action_probs = new_policy.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            old_action_probs = old_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            
            # PPO ratio和clip
            ratio = new_action_probs / (old_action_probs + 1e-8)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - TrainingParameters.CLIP_RANGE, 
                               1.0 + TrainingParameters.CLIP_RANGE) * advantages
            policy_loss = -torch.mean(torch.min(surr1, surr2))
            
            # Value loss
            value_loss = F.mse_loss(new_values, returns)
            
            # Entropy loss
            entropy = -torch.mean(torch.sum(new_policy * torch.log(new_policy + 1e-8), dim=-1))
            
            # Total loss
            total_loss = policy_loss + TrainingParameters.EX_VALUE_COEF * value_loss - \
                        TrainingParameters.ENTROPY_COEF * entropy
        
        # Backpropagation
        self.net_scaler.scale(total_loss).backward()
        self.net_scaler.unscale_(self.net_optimizer)
        
        grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 
                                                  TrainingParameters.MAX_GRAD_NORM)
        
        self.net_scaler.step(self.net_optimizer)
        self.net_scaler.update()
        
        # Calculate clipfrac
        clipfrac = torch.mean((torch.abs(ratio - 1.0) > TrainingParameters.CLIP_RANGE).float())
        
        return [total_loss.cpu().detach().numpy(), 
                policy_loss.cpu().detach().numpy(),
                entropy.cpu().detach().numpy(),
                value_loss.cpu().detach().numpy(),
                value_loss.cpu().detach().numpy(),
                0.0, 0.0,  # valid_loss, blocking_loss (unused)
                clipfrac.cpu().detach().numpy(),
                grad_norm.cpu().detach().numpy(),
                advantages.mean().cpu().detach().numpy()]

    def imitation_train(self, vector, optimal_actions, hidden_state=None):
        """Imitation learning training"""
        self.net_optimizer.zero_grad()
        
        vector = torch.from_numpy(vector).float().to(self.device)
        optimal_actions = torch.from_numpy(optimal_actions).long().to(self.device)
        
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)
        if optimal_actions.dim() == 0:
            optimal_actions = optimal_actions.unsqueeze(0)
        
        with autocast():
            # Forward pass
            _, _, logits = self.network(vector)
            
            # 现在logits是[batch_size, n_actions]，直接计算交叉熵损失
            imitation_loss = F.cross_entropy(logits, optimal_actions)
        
        self.net_scaler.scale(imitation_loss).backward()
        self.net_scaler.unscale_(self.net_optimizer)
        
        grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 
                                                  TrainingParameters.MAX_GRAD_NORM)
        
        self.net_scaler.step(self.net_optimizer)
        self.net_scaler.update()
        
        return [imitation_loss.cpu().detach().numpy(), 
                grad_norm.cpu().detach().numpy()]

    def set_weights(self, weights):
        """Load global weights to local model"""
        self.network.load_state_dict(weights)

    def get_weights(self):
        """Get current model weights"""
        return self.network.state_dict()

    def value(self, vector):
        """Predict state value"""
        with torch.no_grad():
            if isinstance(vector, np.ndarray):
                input_vector = torch.from_numpy(vector).float().to(self.device)
            else:
                input_vector = torch.FloatTensor([vector]).to(self.device)
            
            if input_vector.dim() == 1:
                input_vector = input_vector.unsqueeze(0)
            
            _, value, _ = self.network(input_vector)
            return value.cpu().numpy()