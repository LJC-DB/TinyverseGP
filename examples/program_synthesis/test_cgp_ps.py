"""
Example module to test CGP with program synthesis problems.
"""

from src.gp.tiny_cgp import *
from src.gp.problem import ProgramSynthesis
from src.benchmark.program_synthesis.ps_benchmark import PSBenchmark
from src.benchmark.program_synthesis.leetcode.power_of_two import *
from src.gp.functions import *
from src.gp.tinyverse import Var, Const

NUM_INPUTS = 1
functions = [ADD, SUB, MUL, DIV, AND, OR, NAND, NOR, NOT, IF, LT, GT]
terminals = [Var(i) for i in range(NUM_INPUTS)] + [Const(1), Const(2)]

config = CGPConfig(
    num_jobs=1,
    max_generations=10000,
    stopping_criteria=100,
    minimizing_fitness=False,
    ideal_fitness=100,
    silent_algorithm=False,
    silent_evolver=False,
    minimalistic_output=True,
    num_functions=len(functions),
    max_arity=3,
    num_inputs=NUM_INPUTS,
    num_outputs=1,
    num_function_nodes=100,
    report_interval=1,
    max_time=60
)
config.init()

hyperparameters = CGPHyperparameters(
    mu=1,
    lmbda=32,
    population_size=33,
    levels_back=len(terminals),
    mutation_rate=0.1,
    strict_selection=True
)
config.init()

generator = gen_power_of_two
n = 10
m = 100

benchmark = PSBenchmark(generator, [n,m])
problem = ProgramSynthesis(benchmark.dataset)
cgp = TinyCGP(problem, functions, terminals, config, hyperparameters)
cgp.evolve()
