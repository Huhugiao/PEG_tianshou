from typing import Any, Dict, List, Optional, Type, Sequence, Union, Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.policy import A2CPolicy
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.common import MLP
import algo_config

class Actor(nn.Module):
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
        info: Dict[str, Any] = {}
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        obs = obs[:, :-2]
        obs = torch.tensor(obs)
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
        self, obs: Union[np.ndarray, torch.Tensor], **kwargs: Any
    ) -> torch.Tensor:
        """Mapping: s -> V(s)."""
        # with open("output.txt", "a") as file:
        #         for item in 'start this\n':
        #             file.write(str(item))
        #         for item in obs:
        #             file.write(str(item) + "\n")
        #         for item in 'over\n':
        #             file.write(str(item))
        # obs = np.hstack((obs[:, [0, 2]], obs[:, -2:]))
        if not algo_config.use_god_view:
            obs = obs[:, :-2]
        logits, _ = self.preprocess(obs, state=kwargs.get("state", None))
        return self.last(logits)
    

