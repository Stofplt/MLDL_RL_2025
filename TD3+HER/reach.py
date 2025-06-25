import os
import numpy as np
from gym import utils
from gym.envs.robotics import fetch_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join("fetch", "reach.xml")


class FetchReachEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type="sparse"):
        initial_qpos = {
            "robot0:slide0": 0.4049,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }
        self._max_episode_steps = 50  # Add this line for compatibility
        fetch_env.FetchEnv.__init__(
            self,
            MODEL_XML_PATH,
            has_object=False,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
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
        return super().reset()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self._step_count += 1
        if self._step_count >= 50:
            done = True
        return obs, reward, done, info

    def _is_success(self, achieved_goal, desired_goal):
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
