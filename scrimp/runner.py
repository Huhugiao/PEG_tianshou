import numpy as np
import torch
import ray

from alg_parameters import *
from model import Model
from util import set_global_seeds, update_perf
from env import TrackingEnv
from expert_policies import get_expert_tracker_action_pair, get_expert_target_action_pair


@ray.remote(num_cpus=1, num_gpus=SetupParameters.NUM_GPU / max((TrainingParameters.N_ENVS + 1), 1))
class Runner(object):
    """Runner for Protecting environment with single training agent"""

    def __init__(self, env_id, mission):
        self.ID = env_id
        self.mission = mission  # 0: train tracker, 1: train target (env already flips reward)
        set_global_seeds(env_id * 123)

        # Envs
        self.env = TrackingEnv(mission=mission)
        self.imitation_env = TrackingEnv(mission=mission)

        # Models
        self.local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')
        self.agent_model = Model(self.local_device)

        if TrainingParameters.OPPONENT_TYPE == "policy":
            self.opponent_model = Model(self.local_device)
        else:
            self.opponent_model = None

        # Reset
        self.vector, _ = self.env.reset()
        self.done = False
        self.agent_hidden = None
        self.opponent_hidden = None

    def run(self, model_weights, opponent_weights, total_steps):
        """Collect RL data for training agent"""
        with torch.no_grad():
            self.agent_model.set_weights(model_weights)
            if opponent_weights is not None and self.opponent_model is not None:
                self.opponent_model.set_weights(opponent_weights)

            data = {'vector': [], 'rewards': [], 'values': [], 'actions': [], 'ps': [], 'hidden': [], 'dones': []}
            performance_dict = {
                'per_r': [], 'per_in_r': [], 'per_ex_r': [], 'per_valid_rate': [],
                'per_episode_len': [], 'rewarded_rate': []
            }

            episode_reward = 0.0
            in_reward = 0.0
            ex_reward = 0.0
            invalid = 0
            reward_count = 0
            ep_len = 0

            steps_collected = 0
            episodes = 0

            for _ in range(TrainingParameters.N_STEPS):
                agent_pair, self.agent_hidden, v_pred, prob, agent_index = self.agent_model.step(self.vector, self.agent_hidden)

                if self.mission == 0:
                    if self.opponent_model is not None:
                        opp_pair, self.opponent_hidden, _, _, _ = self.opponent_model.evaluate(self.vector, self.opponent_hidden, greedy=True)
                        tracker_action, target_action = agent_pair, opp_pair
                    else:
                        tracker_action, target_action = agent_pair, -1
                else:
                    if self.opponent_model is not None:
                        opp_pair, self.opponent_hidden, _, _, _ = self.opponent_model.evaluate(self.vector, self.opponent_hidden, greedy=True)
                        tracker_action, target_action = opp_pair, agent_pair
                    else:
                        tracker_action, target_action = -1, agent_pair

                obs, reward, terminated, truncated, info = self.env.step((tracker_action, target_action))
                done = terminated or truncated

                agent_reward = float(reward)

                data['vector'].append(self.vector.astype(np.float32))
                data['values'].append(np.float32(v_pred))
                data['actions'].append(np.int64(agent_index))      # 仍保存离散索引用于PPO
                data['ps'].append(prob.astype(np.float32))
                data['hidden'].append(0.0)
                data['rewards'].append(np.float32(agent_reward))
                data['dones'].append(bool(done))

                episode_reward += agent_reward
                reward_count += 1 if agent_reward > 0 else 0
                ep_len += 1

                self.vector = obs
                steps_collected += 1

                if done or ep_len >= EnvParameters.EPISODE_LEN:
                    one_ep = {
                        'ex_reward': float(ex_reward),
                        'in_reward': float(in_reward),
                        'episode_reward': float(episode_reward),
                        'invalid': int(invalid),
                        'num_step': int(ep_len),
                        'reward_count': int(reward_count)
                    }
                    update_perf(one_ep, performance_dict)

                    # Reset episode
                    episode_reward = 0.0
                    in_reward = 0.0
                    ex_reward = 0.0
                    invalid = 0
                    reward_count = 0
                    ep_len = 0
                    episodes += 1
                    self.vector, _ = self.env.reset()
                    self.agent_hidden = None
                    self.opponent_hidden = None

            last_value = self.agent_model.value(self.vector)
            data = self._compute_gae_returns(data, last_value)
            return data, steps_collected, episodes, performance_dict, self.vector

    def imitation(self, model_weights, opponent_weights, total_steps):
        """Collect imitation data (expert labels) for the training agent"""
        with torch.no_grad():
            self.agent_model.set_weights(model_weights)
            if opponent_weights is not None and self.opponent_model is not None:
                self.opponent_model.set_weights(opponent_weights)

            obs, _ = self.imitation_env.reset()
            vectors = []
            actions = []
            for _ in range(TrainingParameters.N_STEPS):
                vectors.append(obs.astype(np.float32))
                if self.mission == 0:
                    expert_pair = get_expert_tracker_action_pair(obs)
                else:
                    expert_pair = get_expert_target_action_pair(obs)
                expert_index = Model.pair_to_idx(expert_pair[0], expert_pair[1])  # 量化为标签
                actions.append(np.int64(expert_index))

                if self.mission == 0:
                    tracker_action, target_action = expert_pair, -1
                else:
                    tracker_action, target_action = -1, expert_pair
                obs, _, terminated, truncated, _ = self.imitation_env.step((tracker_action, target_action))
                if terminated or truncated:
                    obs, _ = self.imitation_env.reset()

            data = {
                'vector': np.asarray(vectors, dtype=np.float32),
                'actions': np.asarray(actions, dtype=np.int64)
            }
            return data

    def _compute_gae_returns(self, data, last_value):
        rewards = np.asarray(data['rewards'], dtype=np.float32)
        values = np.asarray(data['values'], dtype=np.float32).astype(np.float32)
        dones = np.asarray(data['dones'], dtype=np.bool_)
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in range(T - 1, -1, -1):
            next_non_terminal = 0.0 if dones[t] else 1.0
            next_value = np.float32(last_value) if t == T - 1 else values[t + 1]
            delta = rewards[t] + TrainingParameters.GAMMA * next_value * next_non_terminal - values[t]
            gae = delta + TrainingParameters.GAMMA * TrainingParameters.LAM * next_non_terminal * gae
            adv[t] = gae
        returns = adv + values
        data['returns'] = returns.astype(np.float32)
        data['vector'] = np.asarray(data['vector'], dtype=np.float32)
        data['values'] = values.astype(np.float32)
        data['actions'] = np.asarray(data['actions'], dtype=np.int64)
        data['ps'] = np.asarray(data['ps'], dtype=np.float32)
        data['hidden'] = np.asarray(data['hidden'], dtype=np.float32)
        return data