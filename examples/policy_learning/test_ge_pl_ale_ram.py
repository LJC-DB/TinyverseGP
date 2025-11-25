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
from src.gp.tiny_ge import *
import gymnasium as gym
from gymnasium.core import WrapperActType, WrapperObsType
from gymnasium.spaces import Box
from gymnasium.wrappers import TransformObservation
from src.gp.problem import PolicySearch
from src.gp.functions import *
import warnings
import numpy as np
import datetime
import pathlib
from typing import Any, SupportsFloat

if np.version.version[0] == "2":
    warnings.warn("Using NumPy version >=2 can lead to overflow.")


class NoOp_Wrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Implements common preprocessing techniques for Atari environments (excluding frame stacking).

    For frame stacking use :class:`gymnasium.wrappers.FrameStackObservation`.
    No vector version of the wrapper exists

    Specifically, the following preprocess stages applies to the atari environment:
    - Noop Reset: Obtains the initial state by taking a random number of no-ops on reset, default max 30 no-ops.
    - Frame skipping: The number of frames skipped between steps, 4 by default.

    Example:
        >>> import gymnasium as gym
        >>> import ale_py
        >>> gym.register_envs(ale_py)
        >>> env = gym.make("ALE/Pong-v5", frameskip=1)
        >>> env = NoOp_Wrapper(
        ...     env,
        ...     noop_max=10, frame_skip=4,
        ... )
    """

    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        frame_skip: int = 4,
    ):
        """Wrapper for No-Op preprocessing.

        Args:
            env (Env): The environment to apply the preprocessing
            noop_max (int): For No-op reset, the max number no-ops actions are taken at reset, to turn off, set to 0.
            frame_skip (int): The number of frames between new observation the agents observations effecting the frequency at which the agent experiences the game.
        """
        gym.utils.RecordConstructorArgs.__init__(
            self,
            noop_max=noop_max,
            frame_skip=frame_skip,
        )
        gym.Wrapper.__init__(self, env)
        assert frame_skip > 0
        assert noop_max >= 0

        if frame_skip > 1 and getattr(env.unwrapped, "_frameskip", None) != 1:
            raise ValueError(
                "Disable frame-skipping in the original env. Otherwise, more than one frame-skip will happen as through this wrapper"
            )
        
        self.noop_max = noop_max
        if noop_max > 0:
            assert env.unwrapped.get_action_meanings()[0] == "NOOP"

        self.frame_skip = frame_skip

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Applies the preprocessing for an :meth:`env.step`."""
        total_reward, terminated, truncated, info = 0.0, False, False, {}

        for t in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            self.game_over = terminated

            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info
    
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Resets the environment using preprocessing."""
        # NoopReset
        obs, reset_info = self.env.reset(seed=seed, options=options)

        noops = (
            self.env.unwrapped.np_random.integers(1, self.noop_max + 1)
            if self.noop_max > 0
            else 0
        )
        for _ in range(noops):
            obs, _, terminated, truncated, step_info = self.env.step(0)
            reset_info.update(step_info)
            if terminated or truncated:
                obs, reset_info = self.env.reset(seed=seed, options=options)

        return obs, reset_info


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

env = gym.make('BreakoutNoFrameskip-v4', obs_type='ram', render_mode='rgb_array', max_episode_steps=18000)
env = NoOp_Wrapper(env)
output_shape = (len(breakout_byte_mapping),)
# output_shape = (len(breakout_byte_mapping)+breakout_mapping_bits.sum().sum(),)
env = TransformObservation(env, get_ram, Box(0, 1, output_shape, np.float64))
benchmark = PLBenchmark(env, ale_=False, ale_args=ale_args, flatten_obs_= False)
# benchmark.wrapped_env = TransformObservation(benchmark.wrapped_env, process_obs, Box(-1, 1, (7,), np.float64))
wrapped_env = benchmark.wrapped_env
num_inputs = benchmark.len_observation_space()
num_outputs = benchmark.len_action_space()
checkpoint_dir = 'examples/checkpointing/checkpoints'
experiment_name = 'breakout_ge_ram_redo_1000_100_50_50_.9_.05_30_.05'

config = GPConfig(
    num_jobs=1,
    max_generations=1001,
    stopping_criteria=432,
    minimizing_fitness=False,
    ideal_fitness=432,
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_outputs=num_outputs,
    report_interval=1,
    max_time=9999999,
    global_seed=42,
    checkpointing=True,
    checkpoint_interval=10,
    checkpoint_dir=checkpoint_dir,
    experiment_name=experiment_name,
)

hyperparameters = GEHyperparameters(
    pop_size=100,
    genome_length=50,
    codon_size=50,
    cx_rate=0.9,
    mutation_rate=0.05,
    tournament_size=30,
    penalty_value=-99999,
)

functions = [ADD, SUB, MUL, DIV, AND, OR, NAND, NOR, NOT, LT, GT, EQ, MIN, MAX, IF]
arguments = [f'x{i}' for i in range(num_inputs)]  # Inputs for the functions
grammar = {
    "<expr>": ["IF(<cond>,[0,0,0,1],IF(<cond>,[0,0,1,0],IF(<cond>,[0,1,0,0],[1,0,0,0])))"],
    "<cond>": [
        "ADD(<cond>, <cond>)",
        "SUB(<cond>, <cond>)",
        "MUL(<cond>, <cond>)",
        "DIV(<cond>, <cond>)",
        "AND(<cond>, <cond>)",
        "OR(<cond>, <cond>)",
        "NAND(<cond>, <cond>)",
        "NOR(<cond>, <cond>)",
        "NOT(<cond>)",
        "LT(<cond>, <cond>)",
        "GT(<cond>, <cond>)",        
        "EQ(<cond>, <cond>)",
        "MIN(<cond>, <cond>)",
        "MAX(<cond>, <cond>)",
        "IF(<cond>, <cond>, <cond>)",
        "<d>",
        "<d>.<d><d>",
        "<d><d>.<d><d>",
        *arguments
    ],
    "<d>": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
}

print(datetime.datetime.now())
problem = PolicySearch(env=wrapped_env, ideal_=432, minimizing_=False, num_episodes_ = 10, prob_=.05)
ge = TinyGE(functions, grammar, arguments, config, hyperparameters)

try:
    checkpoint_path = sorted(list(pathlib.Path(f'{checkpoint_dir}/{experiment_name}').iterdir()), key=lambda p: p.stat().st_ctime)[-1]
    checkpoint =  ge.checkpointer.load(checkpoint_path)
    policy = ge.resume(checkpoint, problem)
except:
    policy = ge.evolve(problem)

env.close()
print(datetime.datetime.now())
# input('Press enter to test policy.')

env = gym.make('BreakoutNoFrameskip-v4', obs_type='ram', render_mode='rgb_array', max_episode_steps=18000)
env = NoOp_Wrapper(env)
env = TransformObservation(env, get_ram, Box(0, 1, output_shape, np.float64))
benchmark = PLBenchmark(env, ale_=False, ale_args=ale_args, flatten_obs_= False)
# benchmark.wrapped_env = TransformObservation(benchmark.wrapped_env, process_obs, Box(-1, 1, (7,), np.float64))
wrapped_env = benchmark.wrapped_env
problem = PolicySearch(env=wrapped_env, ideal_=432, minimizing_=False, prob_=.05)
print(problem.evaluate(policy.genome, ge, num_episodes=100, wait_key=False))
env.close()
