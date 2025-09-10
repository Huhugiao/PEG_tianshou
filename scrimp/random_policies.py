import numpy as np
import math
import map_config
import time
from expert_policies import _clip_pair, _wrap_deg

class OpponentPolicy:
    """Base class for all opponent policies"""
    def __init__(self, role="tracker"):
        self.role = role  # "tracker" or "target"
        self.step_count = 0
        self.last_decision_time = 0
        
    def reset(self):
        self.step_count = 0
        self.last_decision_time = 0
        
    def get_action(self, observation):
        self.step_count += 1
        return _clip_pair(0.0, 1.0)

#########################
# Tracker Policy Classes
#########################

class PredictiveTracker(OpponentPolicy):
    """
    Predicts where the target is going and tries to intercept
    Uses the target's velocity to estimate future position
    """
    def __init__(self, prediction_steps=10):
        super().__init__(role="tracker")
        self.prediction_steps = prediction_steps
        self.prev_target_pos = None
        
    def reset(self):
        super().reset()
        self.prev_target_pos = None
        
    def get_action(self, observation):
        self.step_count += 1
        
        # Extract positions from observation
        tracker_pos = np.array([observation[0], observation[1]])  # Normalized tracker position
        target_pos = np.array([observation[2], observation[3]])   # Normalized target position
        
        # Calculate target velocity if we have previous position
        if self.prev_target_pos is not None:
            target_vel = (target_pos - self.prev_target_pos)
            # Predict future position
            future_pos = target_pos + target_vel * self.prediction_steps
            # Clamp future position to stay within bounds [-1, 1]
            future_pos = np.clip(future_pos, -1.0, 1.0)
            
            # Calculate direction to future position
            direction_to_future = future_pos - tracker_pos
            if np.linalg.norm(direction_to_future) > 1e-6:
                direction_to_future /= np.linalg.norm(direction_to_future)
            
            # Calculate desired angle in degrees
            desired_angle_deg = np.degrees(np.arctan2(direction_to_future[1], direction_to_future[0]))
            curr_angle_deg = observation[10] * 360.0
            
            # Calculate angle difference
            angle_diff = _wrap_deg(desired_angle_deg - curr_angle_deg)
            
            # Store current target position for next step
            self.prev_target_pos = target_pos
            return _clip_pair(angle_diff, 1.0)
        else:
            # First step, no velocity information yet
            self.prev_target_pos = target_pos
            # Fall back to direct pursuit
            v_d_to_a = np.array([observation[4], observation[5]])  # tracker->target
            r = float(np.linalg.norm(v_d_to_a))
            if r < 1e-6:
                return _clip_pair(0.0, 1.0)
                
            u_da = v_d_to_a / r
            desired_angle_deg = float(np.degrees(np.arctan2(u_da[1], u_da[0])))
            curr_angle_deg = observation[10] * 360.0
            angle_diff = _wrap_deg(desired_angle_deg - curr_angle_deg)
            return _clip_pair(angle_diff, 1.0)


class CircleTracker(OpponentPolicy):
    """
    Tracker that circles around the base at a certain radius,
    only breaking the circle to intercept the target when it gets close to the base
    """
    def __init__(self, circle_radius=0.3, intercept_threshold=0.4):
        super().__init__(role="tracker")
        self.circle_radius = circle_radius  # Relative to map size
        self.intercept_threshold = intercept_threshold  # When target gets this close to base, intercept
        self.angular_velocity = 0.05  # Radians per step
        self.angle = 0.0
        
    def reset(self):
        super().reset()
        self.angle = 0.0
        
    def get_action(self, observation):
        self.step_count += 1
        
        # Get positions
        base_pos = np.array([0.0, 0.0])  # Base is at center in normalized coords
        target_to_base = np.array([observation[8], observation[9]])
        R = float(np.linalg.norm(target_to_base))  # Distance from target to base
        
        # If target is close to base, switch to intercept mode
        if R < self.intercept_threshold:
            # Intercept mode - go directly after target
            v_d_to_a = np.array([observation[4], observation[5]])  # tracker->target
            r = float(np.linalg.norm(v_d_to_a))
            if r < 1e-6:
                return _clip_pair(0.0, 1.0)
                
            u_da = v_d_to_a / r
            desired_angle_deg = float(np.degrees(np.arctan2(u_da[1], u_da[0])))
            curr_angle_deg = observation[10] * 360.0
            angle_diff = _wrap_deg(desired_angle_deg - curr_angle_deg)
            return _clip_pair(angle_diff, 0.9)  # Slightly reduced speed for precision
        else:
            # Circle mode - orbit around the base
            self.angle += self.angular_velocity
            circle_pos = np.array([
                self.circle_radius * np.cos(self.angle),
                self.circle_radius * np.sin(self.angle)
            ])
            
            tracker_pos = np.array([observation[0], observation[1]])  # Normalized tracker position
            direction_to_circle = circle_pos - tracker_pos
            
            if np.linalg.norm(direction_to_circle) > 1e-6:
                direction_to_circle /= np.linalg.norm(direction_to_circle)
                
            desired_angle_deg = np.degrees(np.arctan2(direction_to_circle[1], direction_to_circle[0]))
            curr_angle_deg = observation[10] * 360.0
            angle_diff = _wrap_deg(desired_angle_deg - curr_angle_deg)
            
            # Adjust speed based on how far we are from the desired circle position
            distance_to_circle = np.linalg.norm(circle_pos - tracker_pos)
            speed_factor = min(1.0, max(0.3, distance_to_circle / 0.5))
            
            return _clip_pair(angle_diff, speed_factor)


class PatrolTracker(OpponentPolicy):
    """
    Tracker that patrols between strategic points and chases the target when detected
    """
    def __init__(self, patrol_points=None, detection_radius=0.4):
        super().__init__(role="tracker")
        self.detection_radius = detection_radius
        
        # Default patrol points if none provided
        if patrol_points is None:
            self.patrol_points = [
                np.array([0.0, 0.0]),     # Center/Base
                np.array([0.5, 0.5]),     # Top-right
                np.array([0.5, -0.5]),    # Bottom-right
                np.array([-0.5, -0.5]),   # Bottom-left
                np.array([-0.5, 0.5])     # Top-left
            ]
        else:
            self.patrol_points = patrol_points
            
        self.current_point_idx = 0
        self.point_reached_threshold = 0.05
        
    def reset(self):
        super().reset()
        self.current_point_idx = 0
        
    def get_action(self, observation):
        self.step_count += 1
        
        tracker_pos = np.array([observation[0], observation[1]])  # Normalized tracker position
        target_pos = np.array([observation[2], observation[3]])   # Normalized target position
        
        # Check if target is within detection radius
        distance_to_target = np.linalg.norm(target_pos - tracker_pos)
        
        if distance_to_target < self.detection_radius:
            # Target detected, chase it
            v_d_to_a = np.array([observation[4], observation[5]])  # tracker->target
            r = float(np.linalg.norm(v_d_to_a))
            if r < 1e-6:
                return _clip_pair(0.0, 1.0)
                
            u_da = v_d_to_a / r
            desired_angle_deg = float(np.degrees(np.arctan2(u_da[1], u_da[0])))
            curr_angle_deg = observation[10] * 360.0
            angle_diff = _wrap_deg(desired_angle_deg - curr_angle_deg)
            return _clip_pair(angle_diff, 1.0)
        else:
            # Patrol mode - move between patrol points
            current_goal = self.patrol_points[self.current_point_idx]
            direction_to_goal = current_goal - tracker_pos
            distance_to_goal = np.linalg.norm(direction_to_goal)
            
            # Check if we've reached the current patrol point
            if distance_to_goal < self.point_reached_threshold:
                # Move to next patrol point
                self.current_point_idx = (self.current_point_idx + 1) % len(self.patrol_points)
                current_goal = self.patrol_points[self.current_point_idx]
                direction_to_goal = current_goal - tracker_pos
                distance_to_goal = np.linalg.norm(direction_to_goal)
            
            if distance_to_goal > 1e-6:
                direction_to_goal /= distance_to_goal
                
            desired_angle_deg = np.degrees(np.arctan2(direction_to_goal[1], direction_to_goal[0]))
            curr_angle_deg = observation[10] * 360.0
            angle_diff = _wrap_deg(desired_angle_deg - curr_angle_deg)
            
            # Slower speed for patrol
            return _clip_pair(angle_diff, 0.7)


class RandomTracker(OpponentPolicy):
    """
    Tracker that moves somewhat randomly but with momentum and occasional target pursuit
    """
    def __init__(self, pursuit_probability=0.2, decision_interval=15):
        super().__init__(role="tracker")
        self.pursuit_probability = pursuit_probability  # Probability of pursuing target
        self.decision_interval = decision_interval      # Steps between random direction changes
        self.current_angle = 0.0
        self.current_speed = 0.5
        
    def reset(self):
        super().reset()
        self.current_angle = 0.0
        self.current_speed = 0.5
        self.last_decision_time = 0
        
    def get_action(self, observation):
        self.step_count += 1
        
        # Occasionally switch to pursuit mode
        is_pursuit_mode = np.random.random() < self.pursuit_probability
        
        curr_angle_deg = observation[10] * 360.0
        
        if is_pursuit_mode:
            # Pursuit mode: chase the target directly
            v_d_to_a = np.array([observation[4], observation[5]])  # tracker->target
            r = float(np.linalg.norm(v_d_to_a))
            if r < 1e-6:
                return _clip_pair(0.0, 1.0)
                
            u_da = v_d_to_a / r
            desired_angle_deg = float(np.degrees(np.arctan2(u_da[1], u_da[0])))
            angle_diff = _wrap_deg(desired_angle_deg - curr_angle_deg)
            return _clip_pair(angle_diff, 0.9)
        
        # Check if it's time to make a new random decision
        if self.step_count - self.last_decision_time >= self.decision_interval:
            self.last_decision_time = self.step_count
            
            # Choose a random angle change
            self.current_angle = np.random.uniform(-45, 45)
            
            # Choose a random speed
            self.current_speed = np.random.uniform(0.3, 1.0)
            
        return _clip_pair(self.current_angle, self.current_speed)


class AreaDenialTracker(OpponentPolicy):
    """
    Tracker that positions itself between the target and base to deny access
    """
    def __init__(self, intercept_factor=0.7):
        super().__init__(role="tracker")
        self.intercept_factor = intercept_factor  # How far along the target-base line to position (0-1)
        
    def get_action(self, observation):
        self.step_count += 1
        
        tracker_pos = np.array([observation[0], observation[1]])  # Normalized tracker position
        target_to_base = np.array([observation[8], observation[9]])  # target->base vector
        
        # Compute the interception point along the target-base line
        if np.linalg.norm(target_to_base) > 1e-6:
            intercept_pos = np.array([observation[2], observation[3]]) + self.intercept_factor * target_to_base
            # Clamp the intercept position to stay within bounds [-1, 1]
            intercept_pos = np.clip(intercept_pos, -1.0, 1.0)
            
            # Direction to intercept point
            direction_to_intercept = intercept_pos - tracker_pos
            if np.linalg.norm(direction_to_intercept) > 1e-6:
                direction_to_intercept /= np.linalg.norm(direction_to_intercept)
                
            desired_angle_deg = np.degrees(np.arctan2(direction_to_intercept[1], direction_to_intercept[0]))
            curr_angle_deg = observation[10] * 360.0
            angle_diff = _wrap_deg(desired_angle_deg - curr_angle_deg)
            
            # Speed depends on distance to intercept point
            distance_to_intercept = np.linalg.norm(intercept_pos - tracker_pos)
            speed_factor = min(1.0, max(0.5, distance_to_intercept))
            
            return _clip_pair(angle_diff, speed_factor)
        else:
            # If target is at the base, go directly to base
            v_d_to_t = np.array([observation[6], observation[7]])  # tracker->base
            d = float(np.linalg.norm(v_d_to_t))
            if d < 1e-6:
                return _clip_pair(0.0, 1.0)
                
            u_dt = v_d_to_t / d
            desired_angle_deg = float(np.degrees(np.arctan2(u_dt[1], u_dt[0])))
            curr_angle_deg = observation[10] * 360.0
            angle_diff = _wrap_deg(desired_angle_deg - curr_angle_deg)
            return _clip_pair(angle_diff, 1.0)

#########################
# Target Policy Classes
#########################

class ZigzagTarget(OpponentPolicy):
    """
    Target that moves in a zigzag pattern toward the base
    """
    def __init__(self, zigzag_amplitude=20, zigzag_period=30):
        super().__init__(role="target")
        self.zigzag_amplitude = zigzag_amplitude  # degrees
        self.zigzag_period = zigzag_period        # steps
        
    def get_action(self, observation):
        self.step_count += 1
        
        # Base direction is toward the base
        v_at = np.array([observation[8], observation[9]])  # target->base
        R = float(np.linalg.norm(v_at))
        
        if R < 1e-6:
            return _clip_pair(0.0, 1.0)
            
        # Calculate base direction toward the base
        u_at = v_at / R
        base_direction_deg = float(np.degrees(np.arctan2(u_at[1], u_at[0])))
        
        # Add zigzag pattern based on sine wave
        zigzag_offset = self.zigzag_amplitude * np.sin(2 * np.pi * self.step_count / self.zigzag_period)
        desired_angle_deg = base_direction_deg + zigzag_offset
        
        # Calculate angle difference from current orientation
        curr_angle_deg = observation[11] * 360.0
        angle_diff = _wrap_deg(desired_angle_deg - curr_angle_deg)
        
        # Full speed always
        return _clip_pair(angle_diff, 1.0)


class EdgeHuggingTarget(OpponentPolicy):
    """
    Target that tries to stay close to the edges of the environment while moving to the base
    """
    def __init__(self, edge_bias=0.4, base_focus=0.6):
        super().__init__(role="target")
        self.edge_bias = edge_bias      # Weight for edge-seeking behavior (0-1)
        self.base_focus = base_focus    # Weight for base-seeking behavior (0-1)
        
    def get_action(self, observation):
        self.step_count += 1
        
        target_pos = np.array([observation[2], observation[3]])   # Normalized target position (-1 to 1)
        
        # Vector toward the nearest edge
        edge_x = 1.0 if target_pos[0] >= 0 else -1.0
        edge_y = 1.0 if target_pos[1] >= 0 else -1.0
        
        # Distance to each edge (0 to 1)
        dist_to_x_edge = 1.0 - abs(target_pos[0])
        dist_to_y_edge = 1.0 - abs(target_pos[1])
        
        # Choose the closest edge direction
        if dist_to_x_edge < dist_to_y_edge:
            edge_direction = np.array([edge_x - target_pos[0], 0.0])
        else:
            edge_direction = np.array([0.0, edge_y - target_pos[1]])
            
        if np.linalg.norm(edge_direction) > 1e-6:
            edge_direction /= np.linalg.norm(edge_direction)
        
        # Vector toward the base
        v_at = np.array([observation[8], observation[9]])  # target->base
        R = float(np.linalg.norm(v_at))
        
        if R < 1e-6:
            return _clip_pair(0.0, 1.0)
            
        u_at = v_at / R
        
        # Combine edge-hugging and base-seeking behaviors
        combined_direction = self.edge_bias * edge_direction + self.base_focus * u_at
        
        if np.linalg.norm(combined_direction) > 1e-6:
            combined_direction /= np.linalg.norm(combined_direction)
            
        desired_angle_deg = np.degrees(np.arctan2(combined_direction[1], combined_direction[0]))
        curr_angle_deg = observation[11] * 360.0
        angle_diff = _wrap_deg(desired_angle_deg - curr_angle_deg)
        
        return _clip_pair(angle_diff, 1.0)


class FeintingTarget(OpponentPolicy):
    """
    Target that occasionally feints in a different direction before returning to base-seeking
    """
    def __init__(self, feint_probability=0.05, feint_duration=20, feint_angle_range=90):
        super().__init__(role="target")
        self.feint_probability = feint_probability
        self.feint_duration = feint_duration
        self.feint_angle_range = feint_angle_range
        self.feinting = False
        self.feint_direction = 0.0
        self.feint_steps_left = 0
        
    def reset(self):
        super().reset()
        self.feinting = False
        self.feint_direction = 0.0
        self.feint_steps_left = 0
        
    def get_action(self, observation):
        self.step_count += 1
        
        # Check if we should start a new feint
        if not self.feinting and np.random.random() < self.feint_probability:
            self.feinting = True
            self.feint_steps_left = self.feint_duration
            
            # Choose a random feint direction offset
            self.feint_direction = np.random.uniform(-self.feint_angle_range, self.feint_angle_range)
            
        # If currently feinting, decrement counter
        if self.feinting:
            self.feint_steps_left -= 1
            if self.feint_steps_left <= 0:
                self.feinting = False
        
        # Base direction is toward the base
        v_at = np.array([observation[8], observation[9]])  # target->base
        R = float(np.linalg.norm(v_at))
        
        if R < 1e-6:
            return _clip_pair(0.0, 1.0)
            
        u_at = v_at / R
        base_direction_deg = float(np.degrees(np.arctan2(u_at[1], u_at[0])))
        
        # If feinting, add the feint direction offset
        if self.feinting:
            desired_angle_deg = base_direction_deg + self.feint_direction
        else:
            desired_angle_deg = base_direction_deg
            
        # Calculate angle difference from current orientation
        curr_angle_deg = observation[11] * 360.0
        angle_diff = _wrap_deg(desired_angle_deg - curr_angle_deg)
        
        # Slow down slightly during feints
        speed_factor = 0.8 if self.feinting else 1.0
        
        return _clip_pair(angle_diff, speed_factor)


class SpiralTarget(OpponentPolicy):
    """
    Target that spirals toward the base
    """
    def __init__(self, spiral_factor=30.0, decay_rate=0.005):
        super().__init__(role="target")
        self.spiral_factor = spiral_factor  # Controls how tight the spiral is
        self.decay_rate = decay_rate        # Controls how quickly the spiral tightens
        
    def get_action(self, observation):
        self.step_count += 1
        
        # Vector to base
        v_at = np.array([observation[8], observation[9]])  # target->base
        R = float(np.linalg.norm(v_at))
        
        if R < 1e-6:
            return _clip_pair(0.0, 1.0)
            
        u_at = v_at / R
        base_direction_deg = float(np.degrees(np.arctan2(u_at[1], u_at[0])))
        
        # Calculate spiral offset, which decreases as we get closer to the base
        spiral_offset = self.spiral_factor * np.exp(-self.decay_rate * self.step_count)
        
        # Alternate spiral direction based on step count
        if self.step_count % 2 == 0:
            desired_angle_deg = base_direction_deg + spiral_offset
        else:
            desired_angle_deg = base_direction_deg - spiral_offset
            
        # Calculate angle difference from current orientation
        curr_angle_deg = observation[11] * 360.0
        angle_diff = _wrap_deg(desired_angle_deg - curr_angle_deg)
        
        return _clip_pair(angle_diff, 1.0)


class TrackerAwareTarget(OpponentPolicy):
    """
    Target that adapts its strategy based on the tracker's position
    Uses direct path when tracker is far, and evasive maneuvers when tracker is close
    """
    def __init__(self, danger_radius=0.3, evasion_angle=60):
        super().__init__(role="target")
        self.danger_radius = danger_radius
        self.evasion_angle = evasion_angle
        
    def get_action(self, observation):
        self.step_count += 1
        
        # Calculate distance to tracker
        v_ad = -np.array([observation[4], observation[5]])  # target->tracker
        r = float(np.linalg.norm(v_ad))
        
        # Vector to base
        v_at = np.array([observation[8], observation[9]])  # target->base
        R = float(np.linalg.norm(v_at))
        
        if R < 1e-6:
            return _clip_pair(0.0, 1.0)
            
        u_at = v_at / R
        base_direction_deg = float(np.degrees(np.arctan2(u_at[1], u_at[0])))
        
        # Check if tracker is within danger radius
        if r < self.danger_radius:
            # Tracker is close - use evasion
            if r > 1e-6:
                u_ad = v_ad / r
                tracker_direction_deg = float(np.degrees(np.arctan2(u_ad[1], u_ad[0])))
                
                # Calculate perpendicular directions
                perp1_deg = tracker_direction_deg + 90
                perp2_deg = tracker_direction_deg - 90
                
                # Choose the perpendicular direction that's closer to the base direction
                diff1 = abs(_wrap_deg(perp1_deg - base_direction_deg))
                diff2 = abs(_wrap_deg(perp2_deg - base_direction_deg))
                
                if diff1 <= diff2:
                    evasion_direction_deg = perp1_deg
                else:
                    evasion_direction_deg = perp2_deg
                    
                # Mix evasion direction with base direction
                # The closer the tracker, the more we prioritize evasion
                evasion_weight = 1.0 - min(1.0, r / self.danger_radius)
                base_weight = 1.0 - evasion_weight
                
                # Weighted average of directions
                desired_angle_deg = evasion_weight * evasion_direction_deg + base_weight * base_direction_deg
            else:
                # If tracker is extremely close, just move perpendicular to the base direction
                desired_angle_deg = base_direction_deg + 90
        else:
            # Tracker is far - head directly to base
            desired_angle_deg = base_direction_deg
            
        # Calculate angle difference from current orientation
        curr_angle_deg = observation[11] * 360.0
        angle_diff = _wrap_deg(desired_angle_deg - curr_angle_deg)
        
        # Full speed always
        return _clip_pair(angle_diff, 1.0)


def create_policy(policy_name, **kwargs):
    """Factory function to create policy objects by name"""
    policy_classes = {
        # Tracker policies
        "predictive_tracker": PredictiveTracker,
        "circle_tracker": CircleTracker,
        "patrol_tracker": PatrolTracker,
        "random_tracker": RandomTracker,
        "area_denial_tracker": AreaDenialTracker,
        
        # Target policies
        "zigzag_target": ZigzagTarget,
        "edge_hugging_target": EdgeHuggingTarget,
        "feinting_target": FeintingTarget,
        "spiral_target": SpiralTarget,
        "tracker_aware_target": TrackerAwareTarget,
    }
    
    if policy_name not in policy_classes:
        raise ValueError(f"Unknown policy name: {policy_name}")
    
    return policy_classes[policy_name](**kwargs)


# Simple usage examples
if __name__ == "__main__":
    # Example observation (dummy data for testing)
    dummy_obs = np.array([
        0.0, 0.0,       # tracker position (normalized)
        0.5, 0.5,       # target position (normalized)
        0.5, 0.5,       # tracker->target
        0.0, 0.0,       # tracker->base
        -0.5, -0.5,     # target->base
        0.0, 0.0,       # tracker and target orientation (normalized)
    ])
    
    # Create and test a policy
    policy = create_policy("predictive_tracker")
    action = policy.get_action(dummy_obs)
    print(f"Predictive Tracker action: {action}")
    
    policy = create_policy("zigzag_target")
    action = policy.get_action(dummy_obs)
    print(f"Zigzag Target action: {action}")