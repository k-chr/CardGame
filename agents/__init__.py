from ..card_game import Player, Card
from abc import ABC, abstractmethod
from typing import Callable, Any, List
from numpy.random import default_rng, Generator
from .utils import small_deck, full_deck


class Agent(Player, ABC):
	def __init__(self, full_deck: bool, alpha: float, epsilon: float, gamma: float, legal_actions_getter: Callable[[Any], List[Any]], rng: Generator =default_rng(2137)):
		self.alpha = alpha
		self.epsilon = epsilon
		self.gamma = gamma
		self.full_deck = full_deck
		self.action_getter = legal_actions_getter
		self.rng = rng
		
	@abstractmethod
	def get_action(self, state: Any): ...
	
	@abstractmethod
	def get_best_action(self, state: Any): ...
 
	def get_name(self) -> str:
		return "Różowe Jednorożce"
 
	def set_final_reward(self, points: dict):
		return super().set_final_reward(points)

	def set_temp_reward(self, discarded_cards: dict, point_deltas: dict):
		return super().set_temp_reward(discarded_cards, point_deltas)
 
	def make_move(self, game_state: dict, was_previous_move_wrong: bool) -> Card:
		action = self.get_action(game_state)
		return small_deck[action] if not self.full_deck else full_deck[action]
	