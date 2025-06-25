import sys
import re
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import copy
import numpy as np

from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}

#train modified function in order to accept environments created in the main
def train(args, extra_args, env_train, eval_env): 
    env_type, env_id = get_env_type(args)
    env_type = 'robotics' 
    print('env_type: {}'.format(env_type)) 

    total_timesteps = int(args.num_timesteps)
    seed = args.seed 

    learn = get_learn_function(args.alg) 
    alg_kwargs = get_learn_function_defaults(args.alg, env_type) 
    alg_kwargs.update(extra_args) 

    env = env_train

    if args.save_video_interval != 0: 
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"), 
                               record_video_trigger=lambda x: x % args.save_video_interval == 0, 
                               video_length=args.save_video_length) 

    if args.network: 
        alg_kwargs['network'] = args.network 
    else: 
        if alg_kwargs.get('network') is None: 
            alg_kwargs['network'] = get_default_network(env_type) 

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs)) 

    model = learn( 
        env=env, #training environment
        seed=seed,
        total_timesteps=total_timesteps,
        eval_env=eval_env, #test environment
        **alg_kwargs
    ) 

    return model, env 


def get_env_type(args): 
    env_id = args.env 

    if args.env_type is not None: 
        return args.env_type, env_id 

    for env_spec in gym.envs.registry.all(): 
        env_type = env_spec.entry_point.split(':')[0].split('.')[-1] 
        _game_envs[env_type].add(env_spec.id) 

    if env_id in _game_envs.keys(): 
        env_type = env_id 
        env_id = [g for g in _game_envs[env_type]][0] 
    else: 
        env_type = None 
        for g, e in _game_envs.items(): 
            if env_id in e: 
                env_type = g 
                break 
        if ':' in env_id: 
            env_type = re.sub(r':.*', '', env_id) 
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys()) 

    return env_type, env_id 


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'


def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        print("rl_algs:", '.'.join(['rl_' + 'algs', alg, submodule]))
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))
    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs


def parse_cmdline_kwargs(args):
    def parse(v):
        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v
    return {k: parse(v) for k, v in parse_unknown_args(args).items()}


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


#modified main function to manage the creation of envs to train
def main(args_list):
    print("args_list (raw):", args_list)

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args_list)
    extra_args = parse_cmdline_kwargs(unknown_args)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[])

    print("Parsed args:", args)

    # Train randomization parameters
    train_mass_scales = {
        'min_mass_scale': 0.05,
        'max_mass_scale': 50
    }

    # Test randomization parameters
    eval_mass_scales = {
        'min_mass_scale': 0.1,
        'max_mass_scale': 10
    }

    env_type, env_id = get_env_type(args)
    if env_type != 'robotics':
        print(f"Warning: env_type is {env_type}, forcing to 'robotics' for Fetch environments.")
        env_type = 'robotics'

    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    get_session(config=config)

    # train env creation
    num_envs_train = args.num_env or multiprocessing.cpu_count()
    env_train = make_vec_env(
        env_id, env_type, num_envs_train, args.seed,
        reward_scale=args.reward_scale,
        flatten_dict_observations=(args.alg not in {'rher'}),
        wrapper_kwargs=train_mass_scales
    )
    print(f"Ambiente di training creato con UDR: {train_mass_scales['min_mass_scale']}-{train_mass_scales['max_mass_scale']}")

    # test env creation
    num_envs_eval = 1
    eval_env = make_vec_env(
        env_id, env_type, num_envs_eval, args.seed,
        reward_scale=args.reward_scale,
        flatten_dict_observations=(args.alg not in {'rher'}),
        wrapper_kwargs=eval_mass_scales
    )
    print(f"Ambiente di test creato con UDR: {eval_mass_scales['min_mass_scale']}-{eval_mass_scales['max_mass_scale']}")

    # calls train with both envs
    model, env_returned_from_train = train(args, extra_args, env_train, eval_env)

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    if args.play:
        logger.log("Running trained model")
        obs = env_returned_from_train.reset()

        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))

        episode_rew = np.zeros(env_returned_from_train.num_envs) if isinstance(env_returned_from_train, VecEnv) else np.zeros(1)
        while True:
            if state is not None:
                actions, _, state, _ = model.step(obs, S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)

            obs, rew, done, _ = env_returned_from_train.step(actions)
            episode_rew += rew
            env_returned_from_train.render()
            done_any = done.any() if isinstance(done, np.ndarray) else done
            if done_any:
                for i in np.nonzero(done)[0]:
                    print('episode_rew={}'.format(episode_rew[i]))
                    episode_rew[i] = 0

    env_returned_from_train.close()
    eval_env.close()

    return model


if __name__ == '__main__':
    args_list = sys.argv
    import os

    gpu_id_list = [s for s in args_list if 'gpu' in s]
    gpu_id = str(gpu_id_list[0].split('=')[-1]) if gpu_id_list else '0' # Default a '0' se non specificato
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    print(f"CUDA_VISIBLE_DEVICES impostato a: {gpu_id}")

    if '--alg=rher' not in args_list:
        args_list.append('--alg=rher')

    seed_val = None
    if '--seed' in args_list:
        try:
            seed_index = args_list.index('--seed')
            seed_val = 100 * (int(args_list[seed_index + 1]) + 1)
        except (ValueError, IndexError):
            print("Avviso: Argomento --seed non valido, usando None.")
            seed_val = None
    print("Seed (calcolato):", seed_val)

    env_name = None
    if '--env' in args_list:
        try:
            env_index = args_list.index('--env')
            env_name = args_list[env_index + 1]
        except (ValueError, IndexError):
            print("Avviso: Argomento --env non valido, usando None.")
            env_name = None
    print("Ambiente:", env_name)

    if '--num_cpu=1' not in args_list:
        args_list.append('--num_cpu=1')
    if '--num_env=1' not in args_list:
        args_list.append('--num_env=1')
    if '--num_timesteps=1.3e6' not in args_list:
        args_list.append('--num_timesteps=1.3e6')
    if '--render=False' not in args_list:
        args_list.append('--render=False')

    log_path_arg_present = any('--log_path' in s for s in args_list)
    if not log_path_arg_present and env_name and seed_val is not None:
        log_path_base = '/tmp/rher_results/rher_np1_case0_results/'
        log_path_full = f"{log_path_base}{env_name}_RHER/{env_name}_RHER_s{seed_val}"
        args_list.append(f'--log_path={log_path_full}')
    elif not log_path_arg_present:
        print("Avviso: --log_path non specificato e non posso costruirlo automaticamente. Imposterà il default.")

    main(args_list)