import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg_parameters import *

class ProtectingNet(nn.Module):
    """简化的MLP网络架构，只使用vector输入，输出单个智能体的策略"""

    def __init__(self):
        """初始化网络"""
        super(ProtectingNet, self).__init__()
        
        # Vector编码器
        self.vector_encoder = nn.Sequential(
            nn.Linear(NetParameters.VECTOR_LEN, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        # 策略头 - 只为单个智能体输出策略
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, EnvParameters.N_ACTIONS)  # 单个智能体的动作空间
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)
                
    def forward(self, vector):
        """
        前向传播，只处理vector输入
        
        Args:
            vector: [batch_size, vector_dim]
        
        Returns:
            policy: [batch_size, n_actions]
            value: [batch_size, 1] 
            policy_logits: [batch_size, n_actions]
        """
        # 确保输入维度正确
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)  # [1, vector_dim]
        
        batch_size = vector.shape[0]
        
        # 向量编码
        encoded = self.vector_encoder(vector)  # [batch_size, 256]
        
        # 生成策略logits
        policy_logits = self.policy_head(encoded)  # [batch_size, n_actions]
        
        # 生成策略概率
        policy = F.softmax(policy_logits, dim=-1)
        
        # 生成价值
        value = self.value_head(encoded)  # [batch_size, 1]
        
        return policy, value, policy_logits