from card_game import Player, Card
from abc import ABC, abstractmethod
from typing import Callable, Any, List
from numpy.random import default_rng, Generator

class Agent(Player, ABC):
	def __init__(self, alpha: float, epsilon: float, gamma: float, legal_actions_getter: Callable[[Any], List[Any]], rng: Generator =default_rng(2137)):
		self.alpha = alpha
		self.epsilon = epsilon
		self.gamma = gamma
		self.action_getter = legal_actions_getter
		self.rng = rng
  
	