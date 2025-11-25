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

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from src.benchmark.policy_search.pl_benchmark import PLBenchmark, ALEArgs
from src.gp.tiny_cgp import *
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.wrappers import TransformObservation
from keras.saving import load_model
from src.gp.problem import PolicySearch
from src.gp.functions import *
from src.gp.tinyverse import Checkpointer
import warnings
import numpy as np
import datetime
import pathlib

if np.version.version[0] == "2":
    warnings.warn("Using NumPy version >=2 can lead to overflow.")

ale_args = ALEArgs(
    noop_max=30,
    frame_skip=4,
    screen_size=84,
    grayscale_obs=True,
    terminal_on_life_loss=False,
    scale_obs=False,
    frame_stack=4,
)

env = gym.make('BreakoutNoFrameskip-v4', max_episode_steps=18000)

benchmark = PLBenchmark(env, ale_=True, ale_args=ale_args, flatten_obs_= False)

cnn = load_model("examples/policy_learning/cnn.keras")
output_shape = cnn.output_shape[1:]
input_shape = (1, *cnn.input_shape[1:])
benchmark.wrapped_env = TransformObservation(benchmark.wrapped_env, lambda obs: cnn.predict(obs.reshape(input_shape), verbose=0), Box(0, np.inf, output_shape))

wrapped_env = benchmark.wrapped_env
functions = [ADD, SUB, MUL, DIV, AND, OR, NAND, NOR, NOT, LT, GT, EQ, MIN, MAX, IF]
terminals = benchmark.gen_terminals()
num_inputs = benchmark.len_observation_space()
num_outputs = benchmark.len_action_space()
checkpoint_dir = 'examples/checkpointing/checkpoints'
experiment_name = 'breakout_cgp_cnn_redo_500_100'

config = CGPConfig(
    num_jobs=1,
    max_generations=501,
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
    report_interval=1,
    max_time=9999999,
    global_seed=42,
    checkpointing=True,
    checkpoint_interval=1,
    checkpoint_dir=checkpoint_dir,
    experiment_name=experiment_name,
)

hyperparameters = CGPHyperparameters(
    mu=1,
    lmbda=99,
    population_size=100,
    num_function_nodes=50,
    levels_back=50,
    mutation_rate=0.1,
    strict_selection=False,
)

print(datetime.datetime.now())
problem = PolicySearch(env=wrapped_env, ideal_=432, minimizing_=False, num_episodes_ = 10, prob_=.01)
cgp = TinyCGP(functions, terminals, config, hyperparameters)

try:
    checkpoint_path = sorted(list(pathlib.Path(f'{checkpoint_dir}/{experiment_name}').iterdir()), key=lambda p: p.stat().st_ctime)[-1]
    checkpoint =  cgp.checkpointer.load(checkpoint_path)
    policy = cgp.resume(checkpoint, problem)
except:
    policy = cgp.evolve(problem)

env.close()
print(datetime.datetime.now())
# input('Press enter to test policy.')

env = gym.make('BreakoutNoFrameskip-v4', render_mode='rgb_array', max_episode_steps=18000)
benchmark = PLBenchmark(env, ale_=True, ale_args=ale_args, flatten_obs_= False)
benchmark.wrapped_env = TransformObservation(benchmark.wrapped_env, lambda obs: cnn.predict(obs.reshape(input_shape)), Box(0, np.inf, output_shape))
wrapped_env = benchmark.wrapped_env
problem = PolicySearch(env=wrapped_env, ideal_=432, minimizing_=False, prob_=.05)
print(problem.evaluate(policy.genome, cgp, num_episodes=100, wait_key=False))
env.close()
