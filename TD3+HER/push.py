import os
from gym import utils
from gym.envs.robotics import fetch_env
import numpy as np


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join("fetch", "push.xml")


class FetchPushEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type="sparse"):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        self._max_episode_steps = 50  # Add this line for compatibility
        self.curriculum_manager = None  # Add curriculum manager

        fetch_env.FetchEnv.__init__(
            self,
            MODEL_XML_PATH,
            has_object=True,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.0,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
        )
        utils.EzPickle.__init__(self, reward_type=reward_type)

    def reset(self):
        self._step_count = 0
        self._successes_this_episode = []  # Track per-timestep success for curriculum
        obs = super().reset()
        if self.curriculum_manager is not None:
            if self.curriculum_manager.get_current_stage() == 0:
                # For reach_cube stage, place cube closer to the gripper
                self.sim.data.set_joint_qpos('object0:joint', 
                                           [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0])
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self._step_count += 1
        info['gripper_pos'] = self.sim.data.get_site_xpos('robot0:grip')
        info['cube_pos'] = self.sim.data.get_site_xpos('object0')
        info['original_reward'] = reward
        # If using curriculum, compute appropriate reward and track per-timestep success
        if self.curriculum_manager is not None:
            reward = self.curriculum_manager.compute_reward(
                obs['achieved_goal'], 
                obs['desired_goal'], 
                info
            )
            if self.curriculum_manager.get_current_stage() == 0:
                gripper_pos = info['gripper_pos']
                cube_pos = info['cube_pos']
                d = np.linalg.norm(gripper_pos - cube_pos)
                self._successes_this_episode.append(float(d < 0.05))
            # End episode early if push_to_target and success
            if self.curriculum_manager.get_current_stage() == 1:
                if self._is_success(obs['achieved_goal'], obs['desired_goal']):
                    done = True
            # Update curriculum based on success at the end of episode
            if done:
                if self.curriculum_manager.get_current_stage() == 0:
                    self.curriculum_manager.update(self._successes_this_episode)
                else:
                    success = self._is_success(obs['achieved_goal'], obs['desired_goal'])
                    self.curriculum_manager.update([success])
        if self._step_count >= 50:
            done = True
        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        if self.curriculum_manager is not None:
            if isinstance(info, dict):
                # Add achieved and desired goals to info for curriculum
                info['achieved_goal'] = achieved_goal
                info['desired_goal'] = desired_goal
            return self.curriculum_manager.compute_reward(achieved_goal, desired_goal, info)
        
        # Default reward logic
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(d > self.distance_threshold).astype(np.float32)

    def _is_success(self, achieved_goal, desired_goal):
        if self.curriculum_manager is not None and self.curriculum_manager.get_current_stage() == 0:
            # In reaching stage, check if gripper is close to cube
            gripper_pos = self.sim.data.get_site_xpos('robot0:grip')
            cube_pos = self.sim.data.get_site_xpos('object0')
            d = np.linalg.norm(gripper_pos - cube_pos)
            return (d < 0.05).astype(np.float32)  # Use same threshold as reward
        else:
            # Default success check (cube to target)
            d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
            return (d < self.distance_threshold).astype(np.float32)

    def _sample_goal(self):
        goal = super()._sample_goal()
        return goal

    def render(self, mode='human', width=500, height=500):
        return super().render(mode=mode, width=width, height=height)

    def seed(self, seed=None):
        return super().seed(seed)

    def close(self):
        return super().close()

    def get_env_state(self):
        return self.sim.get_state()

    def set_env_state(self, state):
        self.sim.set_state(state)
        self.sim.forward()
