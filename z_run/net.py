import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union


class CNN(nn.Module):
    # 9x9, 11x11适用. 7x7则删掉最后的卷积层
    def __init__(
            self, 
            c: int, 
            h: int, 
            w: int, 
            option:int = 0,
            obs_1: int=0,
            obs_2: int=0,
            action_shape: Union[int, Sequence[int]] = 0,
            hidden_num: int = 512,
            device: Union[str, int, torch.device] = "cpu", 
            features_only: bool = False, 
            output_dim_added_layer: Optional[int] = None, 
            layer_init: Callable[[nn.Module], nn.Module] = lambda x: x
        ) -> None:
        super().__init__()
        self.device = device
        self.option = option
        self.obs_1 = obs_1
        self.obs_2 = obs_2
        if w > 7:
            selected_layer = layer_init(nn.Conv2d(hidden_num // 2, hidden_num, 2, 1, 0))  # 8x8 9x9 11x11 input
        else:
            selected_layer = layer_init(nn.Conv2d(hidden_num // 2, hidden_num, 3, 1, 1))  # 7x7 5x5 input

        self.net = nn.Sequential(
            layer_init(nn.Conv2d(c, hidden_num // 4, 3, 1, 1)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(hidden_num // 4, hidden_num // 4, 3, 1, 1)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(hidden_num // 4, hidden_num // 4, 3, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            layer_init(nn.Conv2d(hidden_num // 4, hidden_num // 2, 3, 1, 1)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(hidden_num // 2, hidden_num // 2, 3, 1, 1)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(hidden_num // 2, hidden_num // 2, 3, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            selected_layer,
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        with torch.no_grad():
            base_cnn_output_dim = int(np.prod(self.net(torch.zeros(1, c, h, w)).shape[1:]))

        if not features_only:
            # 输出到action
            action_dim = int(np.prod(action_shape))
            self.net = nn.Sequential(
                self.net,
                layer_init(nn.Linear(base_cnn_output_dim, 512)),
                nn.ReLU(inplace=True),
                layer_init(nn.Linear(512, action_dim)),
            )
            self.output_dim = action_dim
        elif output_dim_added_layer is not None:
            # 输出添加额外层
            self.net = nn.Sequential(
                self.net,
                layer_init(nn.Linear(base_cnn_output_dim, output_dim_added_layer)),
                nn.ReLU(inplace=True),
            )
            self.output_dim = output_dim_added_layer
        else:
            # 直接输出CNN
            self.output_dim = base_cnn_output_dim

    def forward(
            self, 
            obs: Union[np.ndarray, torch.Tensor], 
            state: Any = None, 
            info: Dict[str, Any] = {}
        ) -> Tuple[torch.Tensor, Any]:
        if self.option == 1:
            obs = obs[:, :3*self.obs_1*self.obs_1].reshape(-1, 3, self.obs_1, self.obs_1)  # A 输入
        elif self.option == 2:
            obs = obs[:, 3*self.obs_1*self.obs_1:].reshape(-1, 3, self.obs_2, self.obs_2)  # C 输入
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)

        return self.net(obs), state

class RCNN(nn.Module):
    def __init__(
        self,
        c: int,  
        h: int,  
        w: int,  
        option: int = 0,
        obs_1: int = 0,
        obs_2: int = 0,
        stack_num: int = 1,  # 指定帧堆叠数
        action_shape: Union[int, Sequence[int]] = 0, 
        hidden_num: int = 512, 
        device: Union[str, int, torch.device] = "cpu",  

    ) -> None:
        super().__init__()
        self.option = option
        self.obs_1 = obs_1
        self.obs_2 = obs_2
        self.w = w
        self.device = device
        self.hidden_num = hidden_num
        self.action_dim = int(np.prod(action_shape))
        self.stack_num = stack_num

        # CNN for image feature extraction (two versions)
        self.conv1_stack = nn.Conv2d(c * stack_num, hidden_num // 4, 3, 1, 1)  # 处理堆叠帧

        self.conv1a = nn.Conv2d(hidden_num // 4, hidden_num // 4, 3, 1, 1)
        self.conv1b = nn.Conv2d(hidden_num // 4, hidden_num // 4, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(hidden_num // 4, hidden_num // 2, 3, 1, 1)
        self.conv2a = nn.Conv2d(hidden_num // 2, hidden_num // 2, 3, 1, 1)
        self.conv2b = nn.Conv2d(hidden_num // 2, hidden_num // 2, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3a = nn.Conv2d(hidden_num // 2, hidden_num, 2, 1, 0)  #  9x9、11x11输入
        self.conv3b = nn.Conv2d(hidden_num // 2, hidden_num, 3, 1, 1)  #  7x7输入
        # Fully connected layers
        self.fc_1 = nn.Linear(hidden_num, hidden_num)
        self.fc_2 = nn.Linear(hidden_num, hidden_num)

        # LSTM for processing time-series data
        self.lstm_memory = nn.LSTMCell(input_size=hidden_num, hidden_size=hidden_num // 2)
        self.fc_3 = nn.Linear(hidden_num + hidden_num // 2, hidden_num)

        # Output heads
        self.fc_4 = nn.Linear(hidden_num, self.action_dim)    # DQN
        self.fc_5 = nn.Linear(hidden_num, hidden_num)   # AC

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Dict[str, torch.Tensor]] = None,
        info: Dict[str, Any] = {}
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Mapping: (image sequence -> CNN -> LSTM -> logits)."""

        if obs.ndim == 2:
            obs = obs.unsqueeze(1)
        batch_size, seq_len, _ = obs.shape
        if self.option == 1:
            obs = obs[:, :, :3*self.obs_1*self.obs_1].reshape(batch_size , 3 * seq_len, self.obs_1, self.obs_1)  # A 输入
        elif self.option == 2:
            obs = obs[:, :, 3*self.obs_1*self.obs_1:].reshape(batch_size, 3 * seq_len, self.obs_2, self.obs_2)  # C 输入

        # # 检查输入的维度，并选择合适的卷积层
        # if len(obs.shape) == 5:
        #     # 输入为 [batch, stack_num, channel, height, width]
        #     batch, stack_num, c, h, w = obs.shape
        #     obs = obs.view(batch, stack_num * c, h, w)  # 合并堆叠维度和通道维度
        #     x1 = F.relu(self.conv1_stack(obs))  # 使用适合堆叠输入的卷积层
        # elif len(obs.shape) == 4:
        #     # 输入为 [batch, channel, height, width]
        #     batch, c, h, w = obs.shape
        #     x1 = F.relu(self.conv1_single(obs)) # 直接使用单帧卷积
        # else:
        #     raise ValueError(f"Unexpected input shape: {obs.shape}")

        x1 = F.relu(self.conv1_stack(obs))

        # CNN 其余部分
        x1 = F.relu(self.conv1a(x1))
        x1 = F.relu(self.conv1b(x1))
        x1 = self.pool1(x1)
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv2a(x1))
        x1 = F.relu(self.conv2b(x1))
        x1 = self.pool2(x1)
        if self.w > 7:
            x1 = self.conv3a(x1)       
        else:
            x1 = self.conv3b(x1)       #  7x7 则 删掉最后一层卷积
        x1 = F.relu(x1.view(x1.size(0), -1))

        # ResNet
        x2 = F.relu(self.fc_1(x1))
        x2 = self.fc_2(x2)
        x3 = F.relu(x2 + x1)

        # LSTM
        if state is None:
            batch_size = x3.size(0)
            h_0 = torch.zeros(batch_size, self.lstm_memory.hidden_size, device=x3.device)
            c_0 = torch.zeros(batch_size, self.lstm_memory.hidden_size, device=x3.device)
            state = {"hidden": h_0, "cell": c_0}
        else:
            h_0 = state["hidden"]
            c_0 = state["cell"]

        memories, cells = self.lstm_memory(x3, (h_0, c_0))
        output_state = {"hidden": memories, "cell": cells}
        memories = torch.reshape(memories, (-1, self.hidden_num // 2))
        x3 = torch.reshape(x3, (-1, self.hidden_num))

        # Output
        out = torch.cat([memories, x3], -1)
        out = F.relu(self.fc_3(out))

        if self.action_dim:
            # DQN
            out = F.relu(self.fc_4(out))
            self.output_dim = self.action_dim
        else:
            # AC
            out = F.relu(self.fc_5(out))
            self.output_dim = self.hidden_num

        return out, output_state
    
    
class CNN_LSTM(nn.Module):
    def __init__(self, 
                 c: int, 
                 h: int, 
                 w: int, 
                 option: int = 0,
                 obs_1: int = 0,
                 obs_2: int = 0,
                 action_shape: Union[int, Sequence[int]] = 0,
                 hidden_num: int = 512,
                 num_lstm_layers: int = 1,
                 device: Union[str, int, torch.device] = "cpu", 
                 layer_init: Callable[[nn.Module], nn.Module] = lambda x: x) -> None:
        super().__init__()
        self.device = device
        self.option = option
        self.obs_1 = obs_1
        self.obs_2 = obs_2

        # CNN
        if w > 7:
            selected_layer = layer_init(nn.Conv2d(hidden_num // 2, hidden_num, 2, 1, 0))  # 8x8 9x9 11x11 input
        else:
            selected_layer = layer_init(nn.Conv2d(hidden_num // 2, hidden_num, 3, 1, 1))  # 7x7 5x5 input

        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(c, hidden_num // 4, 3, 1, 1)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(hidden_num // 4, hidden_num // 4, 3, 1, 1)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(hidden_num // 4, hidden_num // 4, 3, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            layer_init(nn.Conv2d(hidden_num // 4, hidden_num // 2, 3, 1, 1)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(hidden_num // 2, hidden_num // 2, 3, 1, 1)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(hidden_num // 2, hidden_num // 2, 3, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            selected_layer,
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        with torch.no_grad():
            base_cnn_output_dim = int(np.prod(self.cnn(torch.zeros(1, c, h, w)).shape[1:]))

        # LSTM
        self.lstm = nn.LSTM(input_size=base_cnn_output_dim, 
                            hidden_size=hidden_num // 2, 
                            num_layers=num_lstm_layers, 
                            batch_first=True)

        # Out_put
        if action_shape:
            action_dim = int(np.prod(action_shape))
            self.output_dim = action_dim
        else:
            self.output_dim = hidden_num // 2

        self.fc = nn.Sequential(
            layer_init(nn.Linear(hidden_num // 2, hidden_num // 2)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(hidden_num // 2, self.output_dim)),
        )

    def forward(self, 
                obs: Union[np.ndarray, torch.Tensor], 
                state: Any = None, 
                info: Dict[str, Any] = {}) -> Tuple[torch.Tensor, Any]:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if obs.ndim == 2:
            obs = obs.unsqueeze(1)
        batch_size, seq_len, _ = obs.shape
        if self.option == 1:
            obs = obs[:, :, :3*self.obs_1*self.obs_1].reshape(batch_size , 3 * seq_len, self.obs_1, self.obs_1)  # A 输入
        elif self.option == 2:
            obs = obs[:, :, 3*self.obs_1*self.obs_1:].reshape(batch_size , 3 * seq_len, self.obs_2, self.obs_2)  # C 输入

        cnn_out = self.cnn(obs)
        cnn_out = cnn_out.view(batch_size, seq_len, -1)  # 恢复为原来的序列形状
        self.lstm.flatten_parameters()
        if state is None:
            rnn_out, (hidden, cell) = self.lstm(cnn_out)
        else:
            # TS store the stack data in [bsz, len, ...] format
            # but pytorch rnn needs [len, bsz, ...]
            rnn_out, (hidden, cell) = self.lstm(
                rnn_out, (
                    state["hidden"].transpose(0, 1).contiguous(),
                    state["cell"].transpose(0, 1).contiguous()
                )
            )
        lstm_out, _ = self.lstm(rnn_out)
        output = self.fc(lstm_out[:, -1])  # Take the output of the last LSTM time step

        return output, {
            "hidden": hidden.transpose(0, 1).detach(),
            "cell": cell.transpose(0, 1).detach()
        }
