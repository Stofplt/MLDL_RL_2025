import numpy as np

class CurriculumManager:
    def __init__(self, env_name, threshold=0.8, window_size=5000):
        self.env_name = env_name
        self.threshold = threshold  # Success rate threshold to advance curriculum
        self.window_size = window_size  # Number of timesteps to calculate success rate
        self.current_stage = 0
        self.success_history = []  # List of 0/1 for each timestep (not episode)
        
        # Define curriculum stages
        if env_name == "FetchPush-v1":
            self.stages = [
                "reach_cube",      # Stage 0: Learn to reach the cube
                "push_to_target"   # Stage 1: Learn to push cube to target
            ]
        else:
            self.stages = ["default"]  # No curriculum for other environments
            
    def get_current_stage(self):
        return self.current_stage
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        if self.env_name != "FetchPush-v1":
            return 0 if info is None else info.get('original_reward', 0)
            
        if self.current_stage == 0:  # reach_cube stage
            # For reaching stage, treat cube position as the goal
            if info is None:
                return -1.0
                
            # Use cube position as the target for reaching
            achieved = info.get('gripper_pos')
            desired = info.get('cube_pos')
            if achieved is None or desired is None:
                return -1.0
                
            dist = np.linalg.norm(achieved - desired)
            reach_threshold = 0.05  # Match FetchReach threshold
            return -(dist > reach_threshold).astype(np.float32)  # Match gym fetch reward style
        else:  # push_to_target stage
            # Try to get positions from info if available
            if info is not None and 'achieved_goal' in info and 'desired_goal' in info:
                achieved = info['achieved_goal']
                desired = info['desired_goal']
            else:
                achieved = achieved_goal
                desired = desired_goal
            d = np.linalg.norm(achieved - desired)
            return -(d > 0.05).astype(np.float32)
    
    def update(self, successes):
        """Update curriculum based on recent success rate. successes: list of 0/1 for each timestep in episode."""
        self.success_history.extend(successes)
        # Only keep recent history (last 5000 timesteps)
        if len(self.success_history) > self.window_size:
            self.success_history = self.success_history[-self.window_size:]
        # Check if we should advance to next stage
        if len(self.success_history) >= self.window_size:
            success_rate = np.mean(self.success_history)
            if success_rate >= self.threshold and self.current_stage < len(self.stages) - 1:
                print("\n" + "="*50)
                print(f"CURRICULUM UPDATE: Switching from {self.stages[self.current_stage]} to {self.stages[self.current_stage + 1]}")
                print(f"Success rate achieved: {success_rate:.3f}")
                print("="*50 + "\n")
                self.current_stage += 1
                self.success_history = []  # Reset history for new stage
                return True  # Indicates curriculum advanced
        return False
    
    def get_info(self):
        """Get information about current curriculum state"""
        return {
            'stage': self.stages[self.current_stage],
            'stage_number': self.current_stage,
            'success_rate': np.mean(self.success_history) if self.success_history else 0.0
        }
