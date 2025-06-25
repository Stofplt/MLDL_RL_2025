import numpy as np
import torch
from numba import jit
import matplotlib.pyplot as plt
import random


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, use_rank=False):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
    
    def add_hindsight(self, trajectory, goal, env, k=4, fetch_env=False):
        for t in range(len(trajectory)):
            state, action, next_state, reward, done = trajectory[t]
            if fetch_env:
                # Extract components for all Fetch environments
                obs = state['observation']
                achieved = state['achieved_goal']
                desired = goal  # Use the new goal
                
                # Compute new reward using env's reward function
                new_reward = env.compute_reward(achieved, desired, None)
                
                # Concatenate observation and goal for state and next_state
                state_arr = np.concatenate([obs, desired])
                if isinstance(next_state, dict):
                    next_obs = next_state['observation']
                    next_desired = desired
                    next_state_arr = np.concatenate([next_obs, next_desired])
                else:
                    next_state_arr = next_state  # fallback
                
                # Add to replay buffer
                self.add(state_arr, action, next_state_arr, new_reward, done)
            else:
                # ...existing code for non-fetch environments...
                pass

    # refactored
    def updated_hindsight_experience(self, state, action, next_state, future_state, env, fetch_reach): 
        if fetch_reach:
            new_goal = future_state["desired_goal"]
            x = np.concatenate([np.array(state["observation"]), new_goal])
            next_x = np.concatenate([np.array(next_state["observation"]), new_goal])
            gc_reward = env.compute_reward(next_state["achieved_goal"], new_goal, {})
        else:
            new_goal = env.get_fingertip_from_state(future_state)[:2]
            x = np.concatenate([state, new_goal])
            next_x = np.concatenate([next_state, new_goal])
            gc_reward = env.goal_cond_reward(action, next_state, new_goal) 
        return x, next_x, gc_reward  


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, state_dim, action_dim, max_timesteps,
                 start_timesteps, max_size=int(1e6),
                 alpha=1.0, beta=0.0):
        super().__init__(state_dim, action_dim, max_size)
        self.priority = np.zeros(max_size)
        self.adjustment = 0
        self.start_timesteps = start_timesteps
        self.max_timesteps = max_timesteps
        self.train_timesteps = max_timesteps - start_timesteps - self.adjustment
        self.rank_ctr = 0
        self.norm_list = np.zeros(max_size)
        self.batched_ranklist = np.zeros(max_size)

        self.alpha = float(alpha)
        self.beta = beta

    def add(self, state, action, next_state, reward, done):
        self.priority[self.ptr] = max(np.max(self.priority), 1.0)

        super().add(state, action, next_state, reward, done)

    def rank_probs(self):
        if self.rank_ctr % 256 == 0:
            problist = list(enumerate(self.priority[:self.size]))
            problist.sort(key=lambda priority : priority[1])
            ranklist = [(len(problist) - new_idx, old_idx) for (new_idx, (old_idx, _)) in enumerate(problist)]
            batched_ranklist = [(1.0/rank, i) for rank, i in ranklist]
            '''
            each segment is of size self.size/batch_size S
            sample 1
            '''
            self.batched_ranklist = batched_ranklist.copy()
            self.batched_ranklist.sort(key=lambda rankidx : rankidx[0], reverse=True)
            
            batched_ranklist.sort(key=lambda rankidx : rankidx[1])
            new_list = [score for score, idx in batched_ranklist]
            norm_list = new_list / np.sum(new_list)
            self.norm_list[:self.size] = norm_list


    def sample(self, batch_size, use_rank=False):
        if use_rank:
            self.rank_probs()
            # self.prob = self.norm_list[:self.size]
            # self.prob /= np.sum(self.prob)
            # self.ind = np.random.choice(self.size, p=self.prob, size=batch_size, replace=True)
            self.ind = np.zeros(batch_size)
            self.prob = np.zeros(batch_size)

            # prelim_p = np.zeros(batch_size)
            # p = np.zeros(batch_size)
            for i in range(256):
                S = (len(self.batched_ranklist)//batch_size)
                if i==255:
                    segment = self.batched_ranklist[i * S:]
                else:
                    segment = self.batched_ranklist[i * S:(i * S) + S]
                p = np.array([rank for (rank, _) in segment])
                p /= np.sum(p)
                rand_choice = np.random.choice(len(segment), p=p)
                self.ind[i] = segment[rand_choice][1]
                self.prob[i] = segment[rand_choice][0]
            self.ind = self.ind.astype(int)
            self.rank_ctr += 1

        else:
            scaled_priorities = np.power(self.priority, self.alpha)[:self.size]
            self.prob = scaled_priorities
            self.prob /= np.sum(self.prob)
            self.ind = np.random.choice(self.size, p=self.prob, size=batch_size, replace=True)
        self.weights = self.compute_weights(use_rank)

        self.beta = min(self.beta + (1.0 / (self.train_timesteps - self.start_timesteps)), 1.0)

        return (
            torch.FloatTensor(self.state[self.ind]).to(self.device),
            torch.FloatTensor(self.action[self.ind]).to(self.device),
            torch.FloatTensor(self.next_state[self.ind]).to(self.device),
            torch.FloatTensor(self.reward[self.ind]).to(self.device),
            torch.FloatTensor(self.not_done[self.ind]).to(self.device)
        )

    def plot(self, list_to_plot):
        plt.plot(np.arange(len(list_to_plot)), list_to_plot)
        plt.show()

    def update_priority(self, td_error):
        self.priority[self.ind] = np.abs(td_error.detach().numpy())

    def compute_weights(self, use_rank):
        if use_rank:
            weights = ((1.0 / self.size) * (1.0 / self.prob))
        else:
            weights = ((1.0 / self.size) * (1.0 / np.take(self.prob, self.ind)))
        beta_weights = np.power(weights, self.beta)
        return beta_weights / np.max(beta_weights)


class GeneralUtils():
    def __init__(self, args):
        self.args = args
        self.fetch_env = "Fetch" in args.env
        self.env_utils = FetchEnvUtils(args.env) if self.fetch_env else None

    def compute_x_goal(self, state, env, sigma=0):
        if self.fetch_env:
            obs_components = self.env_utils.get_observation_components(state)
            x = obs_components['observation']
            goal = obs_components['desired_goal']
            if self.args.use_hindsight and sigma > 0:
                goal = goal + sigma * np.random.randn(self.env_utils.goal_dim)
            return x, goal



    def epsilon_calc(self, eps_upper, eps_lower, max_episode_steps):
        num_episodes = int(np.ceil(self.args.max_timesteps / max_episode_steps))    
        x = np.arange(num_episodes)
        if eps_upper == eps_lower:
            return np.full(num_episodes, eps_upper)
        if self.args.decay_type == 'linear':
            epsilon_step = (eps_upper - eps_lower) / num_episodes
            return np.arange(eps_upper, eps_lower, -epsilon_step)
        if self.args.decay_type  == 'exp':
            return eps_upper * (1 - eps_lower) ** x
class FetchEnvUtils:
    def __init__(self, env_name):
        self.env_name = env_name
        self.is_fetch = "Fetch" in env_name
        self.goal_dim = 3
        
    def get_observation_components(self, state):
        if self.is_fetch:
            return {
                'observation': state['observation'],
                'achieved_goal': state['achieved_goal'],
                'desired_goal': state['desired_goal']
            }
        return state

class HERReplayBuffer(object):
    def __init__(self, obs_dim, action_dim, goal_dim, max_size=int(1e6), her_k=4, reward_func=None):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.her_k = her_k
        self.reward_func = reward_func
        self.obs = np.zeros((max_size, obs_dim))
        self.achieved_goal = np.zeros((max_size, goal_dim))
        self.desired_goal = np.zeros((max_size, goal_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_obs = np.zeros((max_size, obs_dim))
        self.next_achieved_goal = np.zeros((max_size, goal_dim))
        self.next_desired_goal = np.zeros((max_size, goal_dim))
        self.done = np.zeros((max_size, 1))
        self.stage = np.zeros((max_size,), dtype=np.int32)  # Store stage for each transition

    def add(self, obs, achieved_goal, desired_goal, action, reward, next_obs, next_achieved_goal, next_desired_goal, done, stage=0):
        self.obs[self.ptr] = obs
        self.achieved_goal[self.ptr] = achieved_goal
        self.desired_goal[self.ptr] = desired_goal
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.next_achieved_goal[self.ptr] = next_achieved_goal
        self.next_desired_goal[self.ptr] = next_desired_goal
        self.done[self.ptr] = done
        self.stage[self.ptr] = stage
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, stage=None):
        # Only sample from the current stage if specified
        if stage is not None:
            idxs_all = np.where(self.stage[:self.size] == stage)[0]
            if len(idxs_all) < batch_size:
                idxs = np.random.choice(idxs_all, size=batch_size, replace=True)
            else:
                idxs = np.random.choice(idxs_all, size=batch_size, replace=False)
        else:
            idxs = np.random.randint(0, self.size, size=batch_size)
        # HER relabeling: for half the batch, relabel with future achieved goals from the same stage
        her_idxs = np.random.choice(batch_size, size=batch_size // 2, replace=False)
        obs = self.obs[idxs].copy()
        achieved_goal = self.achieved_goal[idxs].copy()
        desired_goal = self.desired_goal[idxs].copy()
        action = self.action[idxs].copy()
        reward = self.reward[idxs].copy()
        next_obs = self.next_obs[idxs].copy()
        next_achieved_goal = self.next_achieved_goal[idxs].copy()
        next_desired_goal = self.next_desired_goal[idxs].copy()
        done = self.done[idxs].copy()
        idxs_stage = self.stage[idxs]
        for i in her_idxs:
            # Only relabel if there are enough future steps in the same stage
            future_candidates = np.where((self.stage[:self.size] == idxs_stage[i]) & (np.arange(self.size) > idxs[i]))[0]
            if len(future_candidates) > 0:
                future_idx = np.random.choice(future_candidates)
                new_goal = self.achieved_goal[future_idx]
                desired_goal[i] = new_goal
                next_desired_goal[i] = new_goal
                reward[i] = self.reward_func(next_achieved_goal[i], new_goal, None)
        return (
            torch.FloatTensor(np.concatenate([obs, desired_goal], axis=1)),
            torch.FloatTensor(action),
            torch.FloatTensor(np.concatenate([next_obs, next_desired_goal], axis=1)),
            torch.FloatTensor(reward),
            torch.FloatTensor(1. - done)
        )
    
    def relay_experiences(self, from_stage, to_stage, relabel_func=None):
        """
        Copy all experiences from from_stage, relabel their goals for to_stage, and add as to_stage.
        relabel_func: function(obs, achieved_goal, desired_goal, info) -> new_desired_goal
        """
        idxs = np.where(self.stage[:self.size] == from_stage)[0]
        for i in idxs:
            obs = self.obs[i]
            achieved_goal = self.achieved_goal[i]
            desired_goal = self.desired_goal[i]
            action = self.action[i]
            reward = self.reward[i]
            next_obs = self.next_obs[i]
            next_achieved_goal = self.next_achieved_goal[i]
            next_desired_goal = self.next_desired_goal[i]
            done = self.done[i]
            # Relabel the goal for the new stage
            if relabel_func is not None:
                # You may need to pass more info depending on your env
                new_desired_goal = relabel_func(obs, achieved_goal, desired_goal)
            else:
                new_desired_goal = desired_goal  # fallback
            # Recompute reward for the new goal
            new_reward = self.reward_func(next_achieved_goal, new_desired_goal, None)
            self.add(
                obs, achieved_goal, new_desired_goal, action, new_reward,
                next_obs, next_achieved_goal, new_desired_goal, done, stage=to_stage
            )