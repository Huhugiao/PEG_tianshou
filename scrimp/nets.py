import torch
import torch.nn as nn
import torch.nn.functional as F
from alg_parameters import EnvParameters, NetParameters


class ProtectingNet(nn.Module):
    """简化的MLP网络架构，只使用vector输入，输出单个智能体的策略"""

    def __init__(self):
        super(ProtectingNet, self).__init__()

        self.vector_encoder = nn.Sequential(
            nn.Linear(NetParameters.VECTOR_LEN, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, EnvParameters.N_ACTIONS)
        )

        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, vector):
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)

        encoded = self.vector_encoder(vector)      # [B, 256]
        policy_logits = self.policy_head(encoded)  # [B, A]
        policy = F.softmax(policy_logits, dim=-1)
        value = self.value_head(encoded)           # [B, 1]
        return policy, value, policy_logits