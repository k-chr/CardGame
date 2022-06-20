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
		
  
	@abstractmethod
	def get_action(self, state: Any): ...
	
	@abstractmethod
	def get_best_action(self, state: Any): ...
 
	def make_move(self, game_state: dict, was_previous_move_wrong: bool) -> Card:
		action = self.get_action(game_state)
		return game_state['hand'][action]
	