import random
import numpy as np
from alg_parameters import *

class EpisodicBuffer(object):
    """Episodic buffer for tracker and target agents"""

    def __init__(self, total_step):
        """Initialize buffer"""
        self._capacity = int(IntrinsicParameters.CAPACITY)
        self.xy_memory = np.zeros((self._capacity, 2, 2))  # [capacity, agent, (x,y)]
        self._count = np.zeros(2, dtype=np.int64)  # For tracker and target
        self.min_step = IntrinsicParameters.N_ADD_INTRINSIC
        self.surrogate1 = IntrinsicParameters.SURROGATE1
        self.surrogate2 = IntrinsicParameters.SURROGATE2
        self.no_reward = False
        if total_step < self.min_step:
            self.no_reward = True

    @property
    def capacity(self):
        return self._capacity

    def id_len(self, agent_id):
        """Current size for an agent"""
        return min(self._count[agent_id], self._capacity)

    def reset(self, total_step):
        """Reset the buffer"""
        self.no_reward = False
        if total_step < self.min_step:
            self.no_reward = True
        self._count = np.zeros(2, dtype=np.int64)
        self.xy_memory = np.zeros((self._capacity, 2, 2))

    def add(self, xy_position, agent_id):
        """Add position to the buffer"""
        if self._count[agent_id] >= self._capacity:
            index = np.random.randint(low=0, high=self._capacity)
        else:
            index = self._count[agent_id]

        self.xy_memory[index, agent_id] = xy_position
        self._count[agent_id] += 1

    def batch_add(self, xy_positions):
        """Add position batch to buffer"""
        self.xy_memory[0] = xy_positions
        self._count += 1

    def if_reward(self, new_xy, rewards, done, on_goal):
        """Calculate intrinsic rewards based on familiarity"""
        processed_rewards = np.zeros((1, 2))
        bonus = np.zeros((1, 2))
        reward_count = 0
        min_dist = np.zeros((1, 2))

        for i in range(2):
            size = self.id_len(i)
            new_xy_array = np.array([new_xy[i]] * int(size))
            dist = np.sqrt(np.sum(np.square(new_xy_array - self.xy_memory[:size, i]), axis=-1))
            novelty = np.asarray(dist < random.randint(1, IntrinsicParameters.K), dtype=np.int64)

            aggregated = np.max(novelty)
            bonus[:, i] = np.asarray([0.0 if done or on_goal[i] else self.surrogate2 - aggregated])
            scale_factor = self.surrogate1
            if self.no_reward:
                scale_factor = 0.0
            intrinsic_reward = scale_factor * bonus[:, i]
            processed_rewards[:, i] = rewards[:, i] + intrinsic_reward
            if intrinsic_reward[0] != 0:
                reward_count += 1

            min_dist[:, i] = np.min(dist)
            if min_dist[:, i] >= IntrinsicParameters.ADD_THRESHOLD:
                self.add(new_xy[i], i)

        return processed_rewards, reward_count, bonus, min_dist