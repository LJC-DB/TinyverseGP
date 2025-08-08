"""
Example module to test CGP with policy search problems.
Evolves a policy for Pong from the Gymnasium Atari Learning Environment:

https://ale.farama.org/
https://ale.farama.org/environments/

https://ale.farama.org/environments/pong/

Pong has the following specifications that are adapted to
the GP mode in this example:

Action space: Discrete(6)

Observation space: Box(0, 255, (210, 160, 3), uint8)
"""

from src.benchmark.policy_search.pl_benchmark import PLBenchmark, ALEArgs
from src.gp.tiny_cgp import *
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.wrappers import TransformObservation
from src.gp.problem import PolicySearch
from src.gp.functions import *
from src.gp.tinyverse import Checkpointer
import warnings
import numpy as np
import datetime
import pathlib

if np.version.version[0] == "2":
    warnings.warn("Using NumPy version >=2 can lead to overflow.")


breakout_byte_mapping = dict(
            player_x=72,
            # blocks_hit_count=77,            
            # score=84,  # 5 for each hit
            ball_x=99,
            ball_y=101,
)
breakout_mapping_indices = sorted(list(breakout_byte_mapping.values()))
breakout_byte_mapping_mask = np.isin(np.arange(128), (breakout_mapping_indices))

breakout_bit_mapping = dict()
'''
Bitmap configuration, where each number corresponds to a 6x2 matrix representing a column of the blocks on RAM:
     X 18 17 16
    12 13 14 15
    11 10  X  X
     9  8  7  6
     2  3  4  5
     1  X  X  X
Each line of the 6x2 matriz corresponds to a line of the blocks, starting from the lowest line.
'''
temp = dict(
    block_bit_map=(
        [0,0,1,0,1,0,1,0,],
        [1,0,1,0,1,0,1,0,],
        [1,0,1,0,0,0,0,0,],
        [1,0,1,0,1,0,1,0,],
        [1,0,1,0,1,0,1,0,],
        [1,0,0,0,0,0,0,0,],
    )
)
for k, v in temp.items():
    for i, vi in enumerate(v):
        for j in range(6):
            breakout_bit_mapping["%s_%i" % (k, i*6+j)] = (i*6+j, vi)
breakout_mapping_indices, breakout_mapping_bits = list(zip(*breakout_bit_mapping.values()))
breakout_bit_mapping_mask = np.isin(np.arange(128), (breakout_mapping_indices))
breakout_mapping_bits = np.array(breakout_mapping_bits).astype(bool)

def get_ram(obs):    
    return obs[breakout_byte_mapping_mask]/255

    bit_values = obs[breakout_bit_mapping_mask]
    bit_values = np.concatenate([np.unpackbits(byte)[breakout_mapping_bits[i]] for i, byte in enumerate(bit_values)])*255

    return np.concatenate((byte_values, bit_values))

def process_obs(obs):
    player_x = obs[3][0]
    ball_1 = obs[1][1:3]
    ball_2 = obs[2][1:3]
    ball_3 = obs[3][1:3]
    speed_1 = ball_2-ball_1
    speed_2 = ball_3-ball_2
    acceleration_1 = speed_2-speed_1
    return np.concatenate(([player_x], ball_3, speed_2, acceleration_1))


ale_args = ALEArgs(
    noop_max=30,
    frame_skip=4,
    screen_size=32,
    grayscale_obs=True,
    terminal_on_life_loss=False,
    scale_obs=False,
    frame_stack=4,
)

env = gym.make('BreakoutNoFrameskip-v4', obs_type='ram', render_mode='rgb_array', frameskip = 4, max_episode_steps=18000)
output_shape = (len(breakout_byte_mapping),)
# output_shape = (len(breakout_byte_mapping)+breakout_mapping_bits.sum().sum(),)
env = TransformObservation(env, get_ram, Box(0, 1, output_shape, np.float64))
benchmark = PLBenchmark(env, ale_=False, ale_args=ale_args, flatten_obs_= False)
# benchmark.wrapped_env = TransformObservation(benchmark.wrapped_env, process_obs, Box(0, 1, (7,), np.float64))
wrapped_env = benchmark.wrapped_env
functions = [ADD, SUB, MUL, DIV, AND, OR, NAND, NOR, NOT, LT, GT, EQ, MIN, MAX, IF]
terminals = benchmark.gen_terminals()
num_inputs = benchmark.len_observation_space()
num_outputs = benchmark.len_action_space()
checkpoint_dir = 'examples/checkpointing/checkpoints'
experiment_name = 'breakout_cgp_ram_only_bytes'

config = CGPConfig(
    num_jobs=1,
    max_generations=1000,
    stopping_criteria=432,
    minimizing_fitness=False,
    ideal_fitness=432,
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_functions=len(functions),
    max_arity=3,
    num_inputs=num_inputs,
    num_outputs=num_outputs,
    num_function_nodes=100,
    report_interval=1,
    max_time=9999999,
    global_seed=42,
    checkpoint_interval=25,
    checkpoint_dir=checkpoint_dir,
    experiment_name=experiment_name,
)

hyperparameters = CGPHyperparameters(
    mu=1,
    lmbda=19,
    population_size=20,
    levels_back=100,
    mutation_rate=0.05,
    strict_selection=True,
)

print(datetime.datetime.now())
problem = PolicySearch(env=wrapped_env, ideal_=432, minimizing_=False, num_episodes_ = 5)
cgp = TinyCGP(functions, terminals, config, hyperparameters)

# checkpoint_path = f'{checkpoint_dir}/{experiment_name}/checkpoint_gen_{850}.dill'
checkpoint_path = sorted(list(pathlib.Path(f'{checkpoint_dir}/{experiment_name}').iterdir()), key=lambda p: p.stat().st_ctime)[-1]
policy = cgp.resume(checkpoint_path, problem)
# policy = cgp.evolve(problem)

env.close()
print(datetime.datetime.now())
input('Press enter to test policy.')

env = gym.make('BreakoutNoFrameskip-v4', obs_type='ram', render_mode='human', frameskip = 4, max_episode_steps=18000)
env = TransformObservation(env, get_ram, Box(0, 1, output_shape, np.float64))
benchmark = PLBenchmark(env, ale_=False, ale_args=ale_args, flatten_obs_= False)
# benchmark.wrapped_env = TransformObservation(benchmark.wrapped_env, process_obs, Box(0, 1, (7,), np.float64))
wrapped_env = benchmark.wrapped_env
problem = PolicySearch(env=wrapped_env, ideal_=432, minimizing_=False)
problem.evaluate(policy.genome, cgp, num_episodes=1, wait_key=False)
env.close()
