import numpy as np
from expert_policies import get_expert_tracker_action_pair, get_expert_target_action_pair
from random_policies import create_policy

class PolicyManager:
    """
    Manages the creation and selection of different policies for tracker and target
    with support for weighted random selection
    """
    def __init__(self):
        # Initialize policy objects
        self._policies = {}
        self._init_policies()
        
        # Default weights (equal probability)
        self._tracker_weights = self._get_default_weights("tracker")
        self._target_weights = self._get_default_weights("target")
        
    def _init_policies(self):
        """Initialize all available policies"""
        # Expert policies (handled specially)
        self._policies["expert_tracker"] = None  # Function-based, no object
        self._policies["expert_target"] = None   # Function-based, no object
        
        # Tracker policies from random_policies.py
        self._policies["predictive_tracker"] = create_policy("predictive_tracker")
        self._policies["circle_tracker"] = create_policy("circle_tracker")
        self._policies["patrol_tracker"] = create_policy("patrol_tracker")
        self._policies["random_tracker"] = create_policy("random_tracker")
        self._policies["area_denial_tracker"] = create_policy("area_denial_tracker")
        
        # Target policies from random_policies.py
        self._policies["zigzag_target"] = create_policy("zigzag_target")
        self._policies["edge_hugging_target"] = create_policy("edge_hugging_target")
        self._policies["feinting_target"] = create_policy("feinting_target")
        self._policies["spiral_target"] = create_policy("spiral_target")
        self._policies["tracker_aware_target"] = create_policy("tracker_aware_target")
    
    def _get_default_weights(self, role):
        """Get default weights (equal probability) for all policies of given role"""
        policies = self.get_policies_by_role(role)
        return {policy: 1.0 for policy in policies}
    
    def get_policies_by_role(self, role):
        """Get list of policy names for the given role"""
        if role == "tracker":
            return ["expert_tracker", "predictive_tracker", "circle_tracker", 
                    "patrol_tracker", "random_tracker", "area_denial_tracker"]
        elif role == "target":
            return ["expert_target", "zigzag_target", "edge_hugging_target", 
                    "feinting_target", "spiral_target", "tracker_aware_target"]
        else:
            raise ValueError(f"Unknown role: {role}")
    
    def set_policy_weights(self, role, weights_dict):
        """
        Set custom weights for policy selection
        
        Args:
            role: "tracker" or "target"
            weights_dict: Dict mapping policy names to weights (probabilities)
        """
        if role == "tracker":
            # Validate all keys exist
            for policy in weights_dict:
                if policy not in self.get_policies_by_role("tracker"):
                    raise ValueError(f"Unknown tracker policy: {policy}")
            self._tracker_weights = weights_dict.copy()
        elif role == "target":
            for policy in weights_dict:
                if policy not in self.get_policies_by_role("target"):
                    raise ValueError(f"Unknown target policy: {policy}")
            self._target_weights = weights_dict.copy()
        else:
            raise ValueError(f"Unknown role: {role}")
    
    def sample_policy(self, role):
        """Randomly sample a policy name based on weights"""
        if role == "tracker":
            weights = self._tracker_weights
        elif role == "target":
            weights = self._target_weights
        else:
            raise ValueError(f"Unknown role: {role}")
        
        # Get policies and their weights
        policies = list(weights.keys())
        weight_values = [weights[p] for p in policies]
        
        # Normalize weights
        total = sum(weight_values)
        if total <= 0:
            # Fallback to uniform if weights sum to zero
            probs = [1.0 / len(weight_values)] * len(weight_values)
        else:
            probs = [w / total for w in weight_values]
        
        # Sample
        return np.random.choice(policies, p=probs)
    
    def reset(self):
        """Reset all stateful policies"""
        for policy_name, policy in self._policies.items():
            if policy is not None:
                policy.reset()
    
    def get_action(self, policy_name, observation):
        """Get action from the specified policy"""
        if policy_name == "expert_tracker":
            return get_expert_tracker_action_pair(observation)
        elif policy_name == "expert_target":
            return get_expert_target_action_pair(observation)
        elif policy_name in self._policies and self._policies[policy_name] is not None:
            return self._policies[policy_name].get_action(observation)
        else:
            raise ValueError(f"Unknown policy: {policy_name}")