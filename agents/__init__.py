from ..card_game import Player, Card
from abc import ABC, abstractmethod
from typing import Callable, Any, List, Optional
from numpy.random._generator import default_rng, Generator
from .utils import small_deck, full_deck


class Agent(Player, ABC):
	def __init__(self, full_deck: bool, alpha: float, epsilon: float, gamma: float, rng: Generator =default_rng(2137)):
		self.alpha = alpha
		self.epsilon = epsilon
		self.gamma = gamma
		self.full_deck = full_deck
		self.previous_action: int = -1
		self.invalid_actions: List[int] =[]
		self.rng = rng
  
		self.training = True

	def toggle_training(self, value: bool): 
		self.training = value
		
	@abstractmethod
	def get_action(self, state: Any, invalid_actions: Optional[List[int]] = None): ...
	
	@abstractmethod
	def get_best_action(self, state: Any, invalid_actions: Optional[List[int]] = None): ...
 
	def get_name(self) -> str:
		return "Różowe Jednorożce"
 
	def set_final_reward(self, points: dict):
		return super().set_final_reward(points)

	def set_temp_reward(self, discarded_cards: dict, point_deltas: dict):
		return super().set_temp_reward(discarded_cards, point_deltas)
 
	def make_move(self, game_state: dict, was_previous_move_wrong: bool) -> Card:
		if was_previous_move_wrong:
			self.invalid_actions.append(self.previous_action)
		else:
			self.invalid_actions.clear()
		action_getter = self.get_action if self.training else self.get_best_action
		action = action_getter(game_state, self.invalid_actions)
		self.previous_action = action
		return small_deck[action] if not self.full_deck else full_deck[action]
	