import os
import numpy as np
from gym import utils
from gym.envs.robotics import fetch_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join("fetch", "pick_and_place.xml")


class FetchPickAndPlaceEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type="sparse"):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        self._max_episode_steps = 50  # Add this line for compatibility
        self.variabilemia = 0
        self.current_subgoal = 0  # 0: move gripper to object, 1: move object to target
        self.subgoal_threshold = 0.5  # Distance threshold for sub-goal completion
        fetch_env.FetchEnv.__init__(
            self,
            MODEL_XML_PATH,
            has_object=True,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=1,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
        )
        utils.EzPickle.__init__(self, reward_type=reward_type)

    def reset(self):
        self._step_count = 0
        self.current_subgoal = 0  # Reset subgoal at episode start
        # Store the object's initial position for sub-goal 0
        obs = super().reset()
        self._object_initial_pos = self.sim.data.get_site_xpos('object0').copy()
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self._step_count += 1
        # Terminate after 50 steps, like official Fetch envs
        if self._step_count >= 50:
            done = True
        # Add all relevant coordinates to info
        info['object_pos'] = self.sim.data.get_site_xpos('object0').copy()
        info['gripper_pos'] = self.sim.data.get_site_xpos('robot0:grip').copy()
        info['target_pos'] = self.goal.copy() if hasattr(self, 'goal') else None
        # Sub-goal switching logic: delay switch until after reward is given
        if self.current_subgoal == 0:
            dist = np.linalg.norm(info['gripper_pos'] - info['object_pos'])
            if dist < self.subgoal_threshold:
                # Set a flag to switch subgoal on the next step
                self._pending_subgoal_switch = True
            else:
                self._pending_subgoal_switch = False
        # Actually switch subgoal if flagged (happens after reward is given)
        if getattr(self, '_pending_subgoal_switch', False):
            self.current_subgoal = 1
            self._pending_subgoal_switch = False
        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        # If info is None, use default sparse reward logic (for HER relabeling)
        if info is None:
            d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
            return -(d > self.distance_threshold).astype(np.float32)
        # Check both subgoals for reward shaping
        gripper_pos = info.get('gripper_pos', None)
        object_pos = info.get('object_pos', None)
        target_pos = info.get('target_pos', None)
        first_done = False
        second_done = False
        # Add a threshold for gripper touching the cube (subgoal 0)
        if gripper_pos is not None and object_pos is not None:
            dist1 = np.linalg.norm(gripper_pos - object_pos)
            if dist1 < self.subgoal_threshold:
                first_done = True
        if object_pos is not None and target_pos is not None:
            dist2 = np.linalg.norm(object_pos - target_pos)
            if dist2 < self.distance_threshold:
                second_done = True
        if first_done and not second_done:
            return 0.5  # Give 0.5 reward when gripper touches the cube but not yet at target
        elif first_done and second_done:
            return 0.0
        else:
            return -1.0

    def _get_obs(self):
        # Get original observation
        obs = super()._get_obs()
        # Sub-goal 0: gripper to object
        if self.current_subgoal == 0:
            gripper_pos = self.sim.data.get_site_xpos('robot0:grip').copy()
            object_pos = self.sim.data.get_site_xpos('object0').copy()
            obs['achieved_goal'] = gripper_pos
            # desired_goal should be the object's position at the START of the episode, not updated every step
            if not hasattr(self, '_object_initial_pos'):
                self._object_initial_pos = object_pos.copy()
            obs['desired_goal'] = self._object_initial_pos
        # Sub-goal 1: object to target (default behavior)
        # else: use the default achieved_goal and desired_goal
        return obs

    def _is_success(self, achieved_goal, desired_goal):
        # Use the same success logic as the original FetchPickAndPlace
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return (d < self.distance_threshold).astype(np.float32)

    def _sample_goal(self):
        # Use the same goal sampling as the original FetchPickAndPlace
        goal = super()._sample_goal()
        return goal

    def render(self, mode='human', width=500, height=500):
        return super().render(mode=mode, width=width, height=height)

    def seed(self, seed=None):
        return super().seed(seed)

    def close(self):
        return super().close()

    def get_env_state(self):
        # Optionally expose the full state for debugging or saving
        return self.sim.get_state()

    def set_env_state(self, state):
        # Optionally restore the full state
        self.sim.set_state(state)
        self.sim.forward()

    def _reset_sim(self):
        # Reset the simulation to the initial state, including object and robot
        # This is a direct override for custom initialization if needed
        return super()._reset_sim()

    def _set_object_initial_pos(self, pos):
        # Set the object's initial position (for curriculum or debugging)
        # pos: np.array of shape (3,)
        body_id = self.sim.model.body_name2id('object0')
        self.sim.model.body_pos[body_id] = pos.copy()
        self.sim.forward()
        self._object_initial_pos = pos.copy()

    def _get_object_initial_pos(self):
        # Return the object's initial position (as set at episode start)
        return getattr(self, '_object_initial_pos', self.sim.data.get_site_xpos('object0').copy())
