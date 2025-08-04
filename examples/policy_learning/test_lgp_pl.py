"""
Example module to test TGP with policy search problems.
Evolves a policy for the Gymnasium Lunar Lander environment.

TGP is used with multiple trees.

https://gymnasium.farama.org/environments/box2d/lunar_lander/

The Lunar Lander has the following specifications that are adapted to
the GP mode in this example:

Action space: Discrete(4)

Observation space: Box([ -2.5 -2.5 -10. -10. -6.2831855 -10. -0. -0. ],
                       [ 2.5 2.5 10. 10. 6.2831855 10. 1. 1. ], (8,), float32)
"""

from math import sqrt, pi
from gymnasium.wrappers import FlattenObservation

from src.gp.tiny_lgp import *
from src.gp.functions import *
from src.gp.problem import PolicySearch
import warnings
import numpy

if numpy.version.version[0] == "2":
    warnings.warn("Using NumPy version >=2 can lead to overflow.")

env = gym.make("LunarLander-v3")
wrapped_env = FlattenObservation(env)

NUM_INPUTS = wrapped_env.observation_space.shape[0]
functions = [ADD, SUB, MUL, DIV, AND, OR, NAND, NOR, NOT, IF, LT, GT]
terminals = [Var(i) for i in range(NUM_INPUTS)] + [
    Const(1),
    Const(2),
    Const(sqrt(2)),
    Const(pi),
    Const(0.5),
]

hyperparameters = LGPHyperparameters(
        mu=30,
        probability_mutation=0.3,
        branch_probability=0.0,
        p_register=1.0,
        max_len = 30
    )
config = LGPConfig(
        num_jobs=1,
        max_generations=300 - hyperparameters.mu,
        stopping_criteria=100,
        minimizing_fitness=False,
        ideal_fitness=100,
        silent_algorithm=False,
        silent_evolver=False,
        minimalistic_output=True,
        report_interval=5,
        max_time=500,
        num_outputs=4,
        num_registers=num_outputs + 2,
        global_seed=13,
        checkpoint_interval=100,
        checkpoint_dir="checkpoints",
        experiment_name="my_experiment",
)

problem = PolicySearch(env=env, ideal_=300, minimizing_=False)
lgp = TinyLGP(functions, terminals, config, hyperparameters)
policy = lgp.evolve(problem)

env = gym.make("LunarLander-v3", render_mode="human")
problem = PolicySearch(env=env, ideal_=100, minimizing_=False)
problem.evaluate(policy.genome, lgp, num_episodes=1, wait_key=True)
