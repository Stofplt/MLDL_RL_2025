"""
main.py
-------
Primary training and evaluation runner for TD3+HER with curriculum learning.
Handles environment setup, training loop, evaluation, and testing.
"""

import os
import sys
import numpy as np
import torch
import gym
import warnings
from datetime import datetime
import mujoco_py
import utils
import TD3
from curriculum import CurriculumManager
from tqdm import tqdm  # Progress bar for test_model

import plotter
from parser import parse_our_args

# Ensure results and models directories exist
os.makedirs("./results", exist_ok=True)
os.makedirs("./models", exist_ok=True)

# Mapping for supported environments
ENV_MAP = {
    "reach": "FetchReach-v1",
    "push": "FetchPush-v1",
    "pnp": "FetchPickAndPlace-v1"
}
LOCAL_ENV_MAP = {
    "FetchReach-v1": "reach.FetchReachEnv",
    "FetchPush-v1": "push.FetchPushEnv",
    "FetchPickAndPlace-v1": "pick_and_place.FetchPickAndPlaceEnv"
}

def eval_policy(policy, env_name, seed, eval_episodes=10, curriculum_manager=None):
    """
    Evaluate a policy in the specified environment for a number of episodes.
    If curriculum_manager is provided, use curriculum-specific success metrics.
    Returns average reward, std, and success rate (if available).
    """
    module_name, class_name = LOCAL_ENV_MAP[env_name].rsplit('.', 1)
    mod = __import__(module_name, fromlist=[class_name])
    env_class = getattr(mod, class_name)
    eval_env = env_class()
    eval_env.reward_type = "sparse"
    if curriculum_manager is not None:
        eval_env.curriculum_manager = curriculum_manager
    eval_env.seed(int(seed) + 100)
    rewards = np.zeros(eval_episodes)
    stage_success_count = 0
    for i in range(eval_episodes):
        returns = 0.0
        state, done = eval_env.reset(), False
        successful_steps = 0
        while not done:
            x = np.concatenate([state['observation'], state['desired_goal']])
            action = policy.select_action(x)
            state, reward, done, info = eval_env.step(action)
            if curriculum_manager is not None and curriculum_manager.get_current_stage() == 0:
                gripper_pos = eval_env.sim.data.get_site_xpos('robot0:grip')
                cube_pos = eval_env.sim.data.get_site_xpos('object0')
                d = np.linalg.norm(gripper_pos - cube_pos)
                if d < 0.05:
                    successful_steps += 1
                returns += reward
            else:
                returns += reward
        rewards[i] = returns
        if curriculum_manager is not None and curriculum_manager.get_current_stage() == 0:
            # Success if at least half the steps were successful
            stage_success = successful_steps >= (eval_env._max_episode_steps // 2)
            stage_success_count += int(stage_success)
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    if curriculum_manager is not None and curriculum_manager.get_current_stage() == 0:
        success_rate = stage_success_count / eval_episodes
    else:
        # Use environment's own success metric for non-curriculum or push_to_target
        success_rate = None
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} from {rewards}")
    print(f"Success rate: {success_rate if success_rate is not None else 'N/A'}")
    print("---------------------------------------")
    return [avg_reward, std_reward, success_rate]

def train(args):
    """
    Main training loop for TD3+HER with optional curriculum learning.
    Handles environment setup, curriculum switching, evaluation, and model saving.
    """
    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    warnings.filterwarnings("ignore")

    
    # Map simplified env names to full names
    if args.env in ENV_MAP:
        args.env = ENV_MAP[args.env]
    

    # Initialize curriculum learning for FetchPush only if the flag is set
    curriculum_manager = None
    if args.curriculum_learning and args.env == "FetchPush-v1":
        curriculum_manager = CurriculumManager(args.env)
        print("\nStarting curriculum learning:")
        print(f"Current stage: {curriculum_manager.stages[0]}")
        print("Will advance to next stage when success rate > 80% over 100 episodes\n")

    if args.env in LOCAL_ENV_MAP:
        module_name, class_name = LOCAL_ENV_MAP[args.env].rsplit('.', 1)
        mod = __import__(module_name, fromlist=[class_name])
        env_class = getattr(mod, class_name)
        env = env_class()
        # Attach curriculum manager only if enabled
        if args.curriculum_learning and args.env == "FetchPush-v1":
            env.curriculum_manager = curriculum_manager
        else:
            curriculum_manager = None
    else:
        raise ValueError("Unsupported environment")
    env.reward_type = "sparse"
    env.seed(int(args.seed))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    obs_dim = env.observation_space.spaces['observation'].shape[0]
    goal_dim = env.observation_space.spaces['desired_goal'].shape[0]
    state_dim = obs_dim + goal_dim
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "prioritized_replay": False,  # Always pass this argument
    }
    if args.policy == "TD3":
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
    # Use HERReplayBuffer for all curriculum stages
    her_k = 4  # Default value, can be made configurable
    reward_func = env.compute_reward  # Use env's reward function (handles curriculum)
    replay_buffer = utils.HERReplayBuffer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        max_size=int(1e6),
        her_k=her_k,
        reward_func=reward_func
    )
    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed, curriculum_manager=curriculum_manager)]
    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    # Track if we just switched to push stage
    push_stage_just_started = False
    push_stage_start_timestep = None
    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1
        obs = state['observation']
        achieved_goal = state['achieved_goal']
        desired_goal = state['desired_goal']
        x = np.concatenate([obs, desired_goal])
        # Boost exploration after curriculum switch
        if curriculum_manager is not None and curriculum_manager.current_stage == 1:
            if not push_stage_just_started:
                push_stage_just_started = True
                push_stage_start_timestep = t
                print("[INFO] Push stage started: boosting exploration for 50,000 timesteps.")
            # For first 50,000 timesteps in push stage, act randomly
            if t - push_stage_start_timestep < 50000:
                action = env.action_space.sample()
            else:
                action = (
                    policy.select_action(np.array(x))
                    + np.random.normal(0, max_action * 0.5, size=action_dim)  # expl_noise=0.5
                ).clip(-max_action, max_action)
        else:
            if t < args.start_timesteps:
                action = env.action_space.sample()
            else:
                action = (
                    policy.select_action(np.array(x))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)
        next_state, reward, done, _ = env.step(action)
        if args.render:
            env.render()
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
        next_obs = next_state['observation']
        next_achieved_goal = next_state['achieved_goal']
        next_desired_goal = next_state['desired_goal']
        # Add to HERReplayBuffer with stage
        replay_buffer.add(
            obs, achieved_goal, desired_goal, action, reward,
            next_obs, next_achieved_goal, next_desired_goal, done_bool,
            stage=curriculum_manager.current_stage if curriculum_manager is not None else 0
        )
        state = next_state
        episode_reward += reward
        if t >= args.start_timesteps:
            # Sample only from the current stage
            policy.train(replay_buffer, args.batch_size, stage=curriculum_manager.current_stage if curriculum_manager is not None else 0)
        # Track rolling sum of episode rewards for success rate
        if 'rolling_rewards' not in locals():
            rolling_rewards = []
        if done:
            rolling_rewards.append(episode_reward)
            if len(rolling_rewards) > 10:
                rolling_rewards.pop(0)
            # Calculate success rate every 10 episodes
            if episode_num % 10 == 0 and len(rolling_rewards) == 10:
                sum_rewards = sum(rolling_rewards)
                # -500 = 0% success, 0 = 100% success
                success_rate = (1.0 + (sum_rewards / 500.0)) * 100.0
                if curriculum_manager is not None:
                    curr_info = curriculum_manager.get_info()
                    print(f"\rStage: {curr_info['stage']}, Rolling Success Rate (last 10 episodes): {success_rate:.1f}%", end="")
                else:
                    print(f"\rRolling Success Rate (last 10 episodes): {success_rate:.1f}%", end="")
                # Advance curriculum if needed
                if curriculum_manager is not None and curriculum_manager.current_stage == 0 and success_rate >= 80.0:
                    curriculum_manager.current_stage = 1
                    curriculum_manager.success_history = []
                    # Relay experiences from previous stage to new stage
                    if hasattr(replay_buffer, 'relay_experiences'):
                        def relabel_func(obs, achieved_goal, desired_goal):
                            # For push_to_target, use the push target as the new goal
                            return desired_goal  # For FetchPush, this is correct
                        # Save current stage
                        old_stage = curriculum_manager.current_stage
                        # Temporarily set to push stage for reward computation
                        curriculum_manager.current_stage = 1
                        replay_buffer.relay_experiences(
                            from_stage=old_stage - 1,
                            to_stage=old_stage,
                            relabel_func=relabel_func
                        )
                        curriculum_manager.current_stage = old_stage  # Restore
                        if hasattr(replay_buffer, 'stage'):
                            n_new = np.sum(replay_buffer.stage[:replay_buffer.size] == curriculum_manager.current_stage)
                            print(f"Relayed {n_new} transitions to stage {curriculum_manager.current_stage} buffer.")
                            # Print a few relayed transitions for debugging
                            idxs = np.where(replay_buffer.stage[:replay_buffer.size] == curriculum_manager.current_stage)[0][:5]
                            for i in idxs:
                                print(f"[DEBUG] Relayed transition {i}: desired_goal={replay_buffer.desired_goal[i]}, reward={replay_buffer.reward[i][0]}")
                    print("\nCurriculum advanced to next stage! Experiences relayed.\n")
            print(f"\nTotal T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
        if done and curriculum_manager is not None:
            curr_info = curriculum_manager.get_info()
            print(f"Stage: {curr_info['stage']}, "
                  f"Success Rate: {curr_info['success_rate']:.3f}")
        if (t + 1) % args.eval_freq == 0:
            evaled_policy = eval_policy(policy, args.env, args.seed, curriculum_manager=curriculum_manager)
            evaluations.append(evaled_policy)
            # Convert to float32 numpy array before saving to avoid object array issues
            evals_np = np.array(evaluations, dtype=np.float32)
            np.save(f"./results/{args.env}.npy", evals_np)
            if curriculum_manager is not None:
                np.save(f"./results/{args.env}_stage{curriculum_manager.current_stage}.npy", evals_np)
            if args.plot:
                plotter.plot(args.env, False, True)
            if args.save_model:
                policy.save(f"./models/{args.env}")


def test_model(args):
    """
    Test a trained model in the specified environment.
    Loads the model, runs several episodes, and prints statistics.
    """
    print(f"\nTesting model: {args.test_model}")
    
    # Always use GLFW for interactive rendering during testing
    os.environ['MUJOCO_GL'] = 'glfw'
    

    # Map simplified env names
    if args.env in ENV_MAP:
        args.env = ENV_MAP[args.env]
    
    is_fetch = "Fetch" in args.env
    

    env = gym.make(args.env)


    # Set seeds
    env.seed(int(args.seed) + 100)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Determine state dimensions
    if is_fetch:
        obs_space = env.observation_space.spaces['observation']
        goal_space = env.observation_space.spaces['desired_goal']
        obs_dim = obs_space.shape[0]
        goal_dim = goal_space.shape[0]
        
        if args.use_hindsight:
            state_dim = obs_dim + goal_dim
        else:
            state_dim = obs_dim
    else:
        state_dim = env.observation_space.shape[0]
        if args.use_hindsight:
            if args.custom_env:
                state_dim += 2
            else:
                state_dim *= 2

    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    # Initialize policy
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "prioritized_replay": args.prioritized_replay,
    }
    
    if args.policy == "TD3":
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
    
    # Load the trained model
    policy.load(args.test_model)

    test_episodes = 10
    rewards = []
    frames = []

    for ep in tqdm(range(test_episodes), desc="Testing episodes"):
        state = env.reset()
        done = False
        ep_reward = 0

        while not done:
            # Prepare input based on environment type
            if is_fetch:
                if args.use_hindsight:
                    x = np.concatenate([state['observation'], state['desired_goal']])
                else:
                    x = state['observation']
            else:
                if args.use_hindsight:
                    x = np.concatenate([state, env.goal])
                else:
                    x = state

            action = policy.select_action(np.array(x))
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward

            # Show animation in real-time if rendering is enabled
            if args.render:
                env.render()

            state = next_state

        rewards.append(ep_reward)
        print(f"Episode {ep+1}/{test_episodes} | Reward: {ep_reward:.2f}")

    # Print final statistics
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"\nTesting complete! Average reward: {avg_reward:.2f} ± {std_reward:.2f}")

if __name__ == "__main__":
    """
    Entry point: parses arguments, sets up environment, and starts training or testing.
    """
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ["PYTHONPATH"] = parent_dir + ":" + os.environ.get("PYTHONPATH", "")

    args = parse_our_args()
    
    # Map simplified env names immediately
    if args.env in ENV_MAP:
        args.env = ENV_MAP[args.env]

    if args.test_model:
        test_model(args)
        sys.exit(0)

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")
    
    kwargs = {}

    if args.smoke_test:
        args.start_timesteps = 25
        args.max_timesteps = 75
        args.eval_freq = 5

    kwargs["args"] = args
    
    train( **kwargs)