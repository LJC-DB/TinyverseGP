import gymnasium as gym
from dataclasses import dataclass
from abc import ABC
from src.benchmark.policy_search.policy_evaluation import GPAgent
from src.gp.tinyverse import GPModel

class Problem(ABC):
    ideal: float

    def is_ideal(self, fitness: float) -> bool:
        return fitness <= self.ideal if self.minimizing \
            else fitness >= self.ideal

    def is_better(self, fitness1: float, fitness2: float) -> bool:
        return fitness1 < fitness2 if self.minimizing \
            else fitness1 > fitness2

    def evaluate(self, genome, model:GPModel):
        pass

@dataclass
class BlackBox(Problem):
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

    #def loss(self, prediction: list) -> float:
    #    return self.loss(self.actual, prediction)

class PolicySearch(Problem):
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
