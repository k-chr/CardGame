from __future__ import annotations
from copy import deepcopy
from card_game import Player, Card
from abc import ABC, abstractmethod, abstractproperty
from typing import Callable, Any, List, Optional
from numpy.random._generator import default_rng, Generator
from .utils import small_deck, full_deck
import numpy as np
from collections import deque


INVALID_ACTION_PENALTY = 50
VICTORY_PENALTY = -100
POINTS_COEF = 1.1**17

class Agent(Player, ABC):
    def __init__(self, full_deck: bool, alpha: float, epsilon: float, gamma: float, rng: Generator =default_rng(2137)):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.full_deck = full_deck
        self.previous_action: int = -1
        self.current_reward: int = 0 
        self.previous_state: dict = None
        self.invalid_actions_per_episode: List[int] = []
        self.cummulative_invalid_actions: int =0
        self.invalid_actions: List[int] =[]
        self.eval_invalid_actions: List[int] =[]
        self.discarded_cards_so_far: List[Card] =[]
        self.rewards = deque(maxlen=100)
        self.rng = rng
        self.loss_callback: Callable[[float], None] = None
        self.invalid_actions_callback: Callable[[int], None] = None
        self.evaluate_callback: Callable[[], None] = None
        self.training = True
        self.eval_interval = 0
        self.max_eval_interval = 11
        

    def toggle_training(self, value: bool): 
        self.training = value
        
    def set_loss_callback(self, fn: Callable[[float], None]):
        self.loss_callback = fn

    def set_invalid_actions_callback(self, fn: Callable[[int], None]):
        self.invalid_actions_callback = fn

    def set_evaluate_callback(self, fn: Callable[[], None]):
        self.evaluate_callback = fn

    @abstractmethod
    def get_action(self, state: Any, invalid_actions: Optional[List[int]] = None): ...
    
    @abstractmethod
    def get_best_action(self, state: Any, invalid_actions: Optional[List[int]] = None): ...
 
    def get_name(self) -> str:
        return "Różowe Jednorożce"
    
    @abstractproperty
    def algorithm_name(self)->str:...
 
    def set_final_reward(self, points: dict):
        self.discarded_cards_so_far.clear()
        if self.training:
            self.eval_interval += 1
            self.invalid_actions_per_episode.append(self.cummulative_invalid_actions)
            if self.invalid_actions_callback: self.invalid_actions_callback(self.cummulative_invalid_actions)
            self.cummulative_invalid_actions = 0
            self.current_reward = VICTORY_PENALTY if min(points.values()) == points[self] else -VICTORY_PENALTY
            if self.evaluate_callback and (self.eval_interval) % self.max_eval_interval == 0:
                self.evaluate_callback()
                self.eval_interval = 0
        return super().set_final_reward(points)

    def set_temp_reward(self, discarded_cards: dict, point_deltas: dict):
     
        self.rewards.append(point_deltas[self])
        self.current_reward = point_deltas[self] * POINTS_COEF
        self.discarded_cards_so_far += list(discarded_cards.values())
        
        return super().set_temp_reward(discarded_cards, point_deltas)
 
    def make_move(self, game_state: dict, was_previous_move_wrong: bool) -> Card:
        invalid_actions = self.invalid_actions if self.training else self.eval_invalid_actions
        if was_previous_move_wrong:
            invalid_actions.append(self.previous_action)
            
        else:
            if self.training: 
                self.cummulative_invalid_actions += invalid_actions.__len__()
            invalid_actions.clear()
                
        game_state["played_cards"] = deepcopy(self.discarded_cards_so_far)
        action_getter = self.get_action if self.training else self.get_best_action
        action = action_getter(game_state, invalid_actions)
        self.previous_state = game_state
        self.previous_action = action
        return small_deck[action] if not self.full_deck else full_deck[action]


from .pg import REINFORCEAgent
from .ac import ACAgent
from .ppo import PPOAgent