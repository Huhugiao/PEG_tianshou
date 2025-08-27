import numpy as np
import ray
import torch

from alg_parameters import *
from model import Model
from util import set_global_seeds
from env import TrackingEnv

@ray.remote(num_cpus=1, num_gpus=SetupParameters.NUM_GPU / (TrainingParameters.N_ENVS + 1))
class Runner(object):
    """Runner for Protecting environment with single training agent"""

    def __init__(self, env_id, mission):
        """Initialize models and environment"""
        self.ID = env_id
        self.mission = mission
        set_global_seeds(env_id*123)
        
        # Create environments
        self.env = TrackingEnv(mission=mission)
        self.imitation_env = TrackingEnv(mission=mission)

        # Set up model for training agent
        self.local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')
        self.agent_model = Model(self.local_device)
        
        # Set up opponent model if using policy-based opponent
        if TrainingParameters.OPPONENT_TYPE == "policy":
            self.opponent_model = Model(self.local_device)
        else:
            self.opponent_model = None

        # Reset environment and get initial observations
        self.vector, _ = self.env.reset()
        self.done = False

        # Hidden states 
        self.agent_hidden = None
        self.opponent_hidden = None

    def run(self, model_weights, opponent_weights, total_steps):
        """Collect reinforcement learning data for training agent"""
        with torch.no_grad():
            # Set weights for agent model
            self.agent_model.set_weights(model_weights)
            
            # Set weights for opponent model if using policy opponent
            if opponent_weights is not None and self.opponent_model:
                self.opponent_model.set_weights(opponent_weights)
            
            # Data collection for training agent
            training_data = {
                'vector': [], 'returns': [], 'values': [], 
                'actions': [], 'ps': [], 'hidden': []
            }
            
            # Performance statistics
            performance_dict = {
                'per_r': [], 'per_in_r': [], 'per_ex_r': [], 'per_valid_rate': [],
                'per_episode_len': [], 'rewarded_rate': []
            }
            
            episode_rewards = []
            episode_step = 0
            
            for step in range(TrainingParameters.N_STEPS):
                # Store current observation
                training_data['vector'].append(self.vector.copy())
                
                # Hidden states
                training_data['hidden'].append(np.zeros((2, 64), dtype=np.float32))
                
                # Get agent action
                if TrainingParameters.AGENT_TO_TRAIN == "tracker":
                    agent_action, self.agent_hidden, agent_v, agent_ps = \
                        self.agent_model.step(self.vector, self.agent_hidden)
                    
                    if TrainingParameters.OPPONENT_TYPE == "policy" and self.opponent_model:
                        # Get opponent action from policy
                        opponent_action, self.opponent_hidden, _, _ = \
                            self.opponent_model.step(self.vector, self.opponent_hidden)
                        
                        tracker_action, target_action = agent_action, opponent_action
                    else:
                        # Rule-based opponent
                        tracker_action, target_action = agent_action, None
                else:
                    # Training target agent
                    agent_action, self.agent_hidden, agent_v, agent_ps = \
                        self.agent_model.step(self.vector, self.agent_hidden)
                    
                    if TrainingParameters.OPPONENT_TYPE == "policy" and self.opponent_model:
                        # Get opponent action from policy
                        opponent_action, self.opponent_hidden, _, _ = \
                            self.opponent_model.step(self.vector, self.opponent_hidden)
                        
                        tracker_action, target_action = opponent_action, agent_action
                    else:
                        # Rule-based opponent
                        tracker_action, target_action = None, agent_action
                
                # Store values and predictions
                training_data['values'].append(agent_v.flatten())
                training_data['ps'].append(agent_ps.copy())
                training_data['actions'].append(agent_action)
                
                # Execute actions and get environment feedback
                try:
                    vector_new, rewards, terminated, truncated, info = self.env.step(tracker_action, target_action)
                    done = terminated or truncated or episode_step >= EnvParameters.EPISODE_LEN - 1
                except Exception as e:
                    print(f"Error in step: {e}")
                    vector_new = self.vector
                    rewards = 0
                    done = True
                
                # Update state
                self.vector = vector_new
                self.done = done
                
                # Store rewards (will be inverted for target agent)
                if TrainingParameters.AGENT_TO_TRAIN == "target":
                    agent_reward = -rewards  # Target wants to avoid being tracked
                else:
                    agent_reward = rewards  # Tracker wants to track target
                
                training_data['returns'].append(agent_reward)
                
                # Accumulate episode rewards
                episode_rewards.append(agent_reward)
                episode_step += 1
                
                # Reset if episode ends
                if done:
                    # Calculate episode statistics
                    episode_total_reward = np.sum(episode_rewards)
                    
                    # Update performance statistics
                    performance_dict['per_r'].append(float(episode_total_reward))
                    performance_dict['per_in_r'].append(0)  # Not using intrinsic rewards
                    performance_dict['per_ex_r'].append(float(episode_rewards[-1]) if episode_rewards else 0)
                    performance_dict['per_valid_rate'].append(1.0)  # Assuming all actions are valid
                    performance_dict['per_episode_len'].append(episode_step)
                    performance_dict['rewarded_rate'].append(0)  # Not using rewarded rate
                    
                    # Reset environment
                    self.vector, _ = self.env.reset()
                    self.done = False
                    
                    # Reset hidden states
                    self.agent_hidden = None
                    self.opponent_hidden = None
                    
                    # Reset episode records
                    episode_rewards = []
                    episode_step = 0
            
            # Calculate GAE returns
            training_data = self._compute_gae_returns(training_data)
            
            # Return collected data
            return (training_data['vector'], training_data['returns'], training_data['values'],
                   training_data['actions'], training_data['ps'], training_data['hidden'],
                   len(performance_dict['per_r']), performance_dict)

    def imitation(self, model_weights, opponent_weights, total_steps):
        """Collect imitation learning data for training agent"""
        with torch.no_grad():
            self.agent_model.set_weights(model_weights)
            
            if opponent_weights is not None and self.opponent_model:
                self.opponent_model.set_weights(opponent_weights)
            
            # Data collection arrays
            training_data = {'vector': [], 'actions': [], 'hidden': []}
            
            episodes_count = 0
            steps_count = 0
            
            # Run multiple episodes to collect demonstration data
            while steps_count < TrainingParameters.N_STEPS and episodes_count < 5:
                # Reset imitation environment
                vector, _ = self.imitation_env.reset()
                
                # Reset hidden states
                agent_hidden = None
                opponent_hidden = None
                
                done = False
                episode_steps = 0
                
                while not done and episode_steps < EnvParameters.EPISODE_LEN and steps_count < TrainingParameters.N_STEPS:
                    # Get expert action for training agent
                    if TrainingParameters.AGENT_TO_TRAIN == "tracker":
                        expert_action = self._get_expert_tracker_action(vector)
                    else:
                        expert_action = self._get_expert_target_action(vector)
                    
                    # Store training data
                    training_data['vector'].append(vector.copy())
                    training_data['actions'].append(expert_action)
                    training_data['hidden'].append(np.zeros((2, 64), dtype=np.float32))
                    
                    # Get opponent action if needed
                    if TrainingParameters.AGENT_TO_TRAIN == "tracker":
                        if TrainingParameters.OPPONENT_TYPE == "policy" and self.opponent_model:
                            opponent_action, opponent_hidden, _, _ = \
                                self.opponent_model.step(vector, opponent_hidden)
                            tracker_action, target_action = expert_action, opponent_action
                        else:
                            tracker_action, target_action = expert_action, None
                    else:
                        if TrainingParameters.OPPONENT_TYPE == "policy" and self.opponent_model:
                            opponent_action, opponent_hidden, _, _ = \
                                self.opponent_model.step(vector, opponent_hidden)
                            tracker_action, target_action = opponent_action, expert_action
                        else:
                            tracker_action, target_action = None, expert_action
                    
                    # Execute actions
                    try:
                        vector, _, terminated, truncated, _ = self.imitation_env.step(tracker_action, target_action)
                        done = terminated or truncated
                    except Exception as e:
                        print(f"Error in imitation step: {e}")
                        done = True
                    
                    episode_steps += 1
                    steps_count += 1
                
                episodes_count += 1
            
            # Convert to numpy arrays
            if len(training_data['vector']) > 0:
                obs_dim = self.env.observation_space.shape[0]
                
                mb_vector = np.array(training_data['vector'], dtype=np.float32)
                mb_actions = np.array(training_data['actions'], dtype=np.int32)
                mb_hidden = np.array(training_data['hidden'], dtype=np.float32)
            else:
                # Return empty arrays with correct dimensions
                obs_dim = self.env.observation_space.shape[0]
                mb_vector = np.zeros((0, obs_dim), dtype=np.float32)
                mb_actions = np.zeros((0,), dtype=np.int32)
                mb_hidden = np.zeros((0, 2, 64), dtype=np.float32)
            
            return (mb_vector, mb_actions, mb_hidden, episodes_count)

    def _compute_gae_returns(self, data):
        """Calculate GAE returns"""
        rewards = np.array(data['returns'], dtype=np.float32)
        values = np.array(data['values'], dtype=np.float32)
        
        # Simple discounted returns calculation
        returns = np.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + TrainingParameters.GAMMA * running_return
            returns[t] = running_return
        
        data['returns'] = returns
        
        # Convert to numpy arrays
        for key in data:
            if key == 'actions':
                data[key] = np.asarray(data[key], dtype=np.int32)
            else:
                data[key] = np.asarray(data[key], dtype=np.float32)
        
        return data

    def _get_expert_tracker_action(self, observation):
        """Expert action for tracker agent"""
        # Extract relative position (tracker to target)
        rel_x = observation[4] if len(observation) > 4 else 0
        rel_y = observation[5] if len(observation) > 5 else 0
        
        # Tracker expert strategy: move towards the target
        if abs(rel_x) < 0.02 and abs(rel_y) < 0.02:
            tracker_action = 1  # no movement or slow movement
        else:
            # Calculate direction to target
            angle = np.arctan2(rel_y, rel_x)
            if angle < 0:
                angle += 2 * np.pi
            
            direction_index = int(angle / (2 * np.pi) * 16) % 16
            distance = np.sqrt(rel_x**2 + rel_y**2)
            
            if distance > 0.1:
                speed_level = 2  # fast
            else:
                speed_level = 1  # slow
            
            tracker_action = direction_index * 3 + speed_level
            
        return tracker_action

    def _get_expert_target_action(self, observation):
        """Expert action for target agent"""
        # 提取观察中的相关信息
        # 假设observation包含tracker和target之间的相对位置以及base和target之间的相对位置
        tracker_rel_x = observation[4]  # tracker相对于target的x坐标
        tracker_rel_y = observation[5]  # tracker相对于target的y坐标
        base_rel_x = observation[6]  # base相对于target的x坐标
        base_rel_y = observation[7]  # base相对于target的y坐标
        
        # 计算逃离方向（远离tracker）
        escape_angle = np.arctan2(-tracker_rel_y, -tracker_rel_x)
        if escape_angle < 0:
            escape_angle += 2 * np.pi
        
        # 计算基地方向
        base_angle = np.arctan2(base_rel_y, base_rel_x)
        if base_angle < 0:
            base_angle += 2 * np.pi
        
        # 混合方向（70%朝向基地，30%远离tracker）
        mixed_angle = (base_angle * 0.7) + (escape_angle * 0.3)
        mixed_angle = mixed_angle % (2 * np.pi)
        
        # 将角度转换为离散动作
        direction_index = int(mixed_angle / (2 * np.pi) * 16) % 16
        
        # 始终使用最高速度（不再基于距离调整速度）
        speed_level = 2  # 始终使用最快速度
        
        # 组合方向和速度得到最终动作
        expert_action = direction_index * 3 + speed_level
        
        return expert_action