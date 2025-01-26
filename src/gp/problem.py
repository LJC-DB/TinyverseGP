import gymnasium as gym
from dataclasses import dataclass
from abc import ABC
from src.benchmark.policy_search.policy_evaluation import GPAgent
from src.gp.tinyverse import GPModel

class Problem(ABC):
    '''
    Abstract class for a problem to be solved by genetic programming.
    '''
    ideal: float

    def is_ideal(self, fitness: float) -> bool:
        '''
        Check if the fitness reached an ideal state.
        This can prompt an early stop in the optimization process.
        '''
        return fitness <= self.ideal if self.minimizing \
            else fitness >= self.ideal

    def is_better(self, fitness1: float, fitness2: float) -> bool:
        '''
        Check if the first fitness is better than the second.
        It takes into consideration whether the problem is minimizing or maximizing.
        '''
        return fitness1 < fitness2 if self.minimizing \
            else fitness1 > fitness2

    def evaluate(self, genome, model:GPModel):
        '''
        This method implements how to evaluate the genome using the model.
        It is problem-specific and should be implemented by the user.
        '''
        pass

@dataclass
class BlackBox(Problem):
    '''
    A black-box problem where the fitness is calculated by a loss function
    of a set of examples of input and output.
    '''
    observations: list
    actual: list

    def __init__(self, observations_: list, actual_: list, loss_: callable,
                 ideal_: float, minimizing_: bool):
        self.observations = observations_
        self.actual = actual_
        self.loss = loss_
        self.ideal = ideal_
        self.minimizing = minimizing_
        self.unidim = True if isinstance(self.actual[0], float) or isinstance(self.actual[0], int)  else False

    def evaluate(self, genome, model:GPModel) -> float:
        predictions = []
        for observation in self.observations:
            prediction = model.predict(genome, observation)
            predictions.append(prediction)
        return self.cost(predictions)

    def cost(self, predictions: list) -> float:
        cost = 0.0
        for index, _ in enumerate(predictions[0]):
            cost += self.loss([prediction[index] for prediction in predictions], [act if self.unidim else act[index] for act in self.actual])
        return cost

class PolicySearch(Problem):
    '''
    A reinforcement learning problem where the fitness is calculated by the
    average reward of the policy.
    '''
    agent: GPAgent
    num_episodes: int

    def __init__(self, env: gym.Env, ideal_: float, minimizing_: bool, num_episodes_: int = 100):
        self.agent = GPAgent(env)
        self.ideal = ideal_
        self.minimizing = minimizing_
        self.num_episodes = num_episodes_

    def evaluate(self, genome, model:GPModel, num_episodes:int = None, wait_key:bool = False) -> float:
        if num_episodes is None:
            num_episodes = self.num_episodes
        return self.agent.evaluate_policy(genome, model, num_episodes, wait_key)
