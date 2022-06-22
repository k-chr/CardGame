from typing import Any, List, Callable, Dict, NamedTuple, Type, Union, Generic, TypeVar, Deque
import torch as t
import numpy as np
from collections import deque
from operator import itemgetter
from abc import abstractmethod, ABCMeta
from card_game import CardGame, Card

ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
suits = ['Clovers', 'Diamonds', 'Hearts', 'Spades']

small_deck = [Card(suits[j], ranks[i]) for i in range(7, 13) for j in range(4)]
full_deck = [Card(suits[j], ranks[i]) for i in range(0, 13) for j in range(4)]

class Trajectory(NamedTuple):
	state: Union[t.Tensor, List[t.Tensor]]
	action: Union[int, List[int]]
	reward: Union[float, List[float]]
	prob: Union[float, List[float]]
	
def get_legal_actions(game: CardGame, agent, state):
    hand = state['hand']
    legit_hand = [card for card in hand if game._CardGame_validate(agent, card)]
    return legit_hand

def cumulative_rewards(gamma, rewards):
	l = len(rewards)
	G = [0 for _ in range(l)]
	r_t_1 = 0

	T = reversed(range(0, l))
 
	for t, r_t in zip(T, rewards[::-1]):
		r_t_1 = r_t_1 * gamma + r_t
		G[t] = r_t_1
  
	return np.asarray(G)

TRAJ = TypeVar("TRAJ")
class Memory(Generic[TRAJ]):
    
	def __init__(self, size, cls: Type[TRAJ]) -> None:
		super().__init__()
		self.__queue: Deque[TRAJ] = deque(maxlen=size)
		self.__trajectory_cls = cls
		self._max_size = size
  
	def store(self, trajectory: TRAJ):
		self.__queue.append(trajectory)
	
	def __len__(self): return len(self.__queue)
	
	def sample(self, batch_size: int, random_state: np.random.Generator=None):
		if not random_state:
			batch_idx = np.random.choice(np.arange(len(self)), size=batch_size)
		else:
			batch_idx = random_state.choice(np.arange(len(self)), size=batch_size)
		getter = itemgetter(*batch_idx)
		memory = getter(self.__queue)
		batch:TRAJ = self.__trajectory_cls(*zip(*memory))
   
		return batch

	def get(self, index) -> TRAJ:
		getter = itemgetter(*index)
		memory = getter(self.__queue)
		batch:TRAJ = self.__trajectory_cls(*zip(*memory))
		return batch
	def set_items(self, queue):
		self.clear()
		self.__queue = deque(queue, self._max_size)
 
	def clear(self):
		self.__queue.clear()	
  
  
class StateParser(metaclass=ABCMeta):
	def parse(self, state) -> t.Tensor: return self._parse(state)
	
	@abstractmethod
	def _parse(self, state) -> t.Tensor:...
	
	@property
	def state_len(self): return self._state_len()
 
	@abstractmethod
	def _state_len(self)-> int:...
 
class HeartsStateParser(StateParser):
    
	def __init__(self, is_full_deck: bool =False) -> None:
		self.deck_size = 52 if is_full_deck else 24
		self.deck = full_deck if is_full_deck else small_deck
		super().__init__()
  
	def _state_len(self) -> int:
		return self.deck_size * 3

	def _parse(self, state) -> t.Tensor:
		hand: List[Card] = state['hand']
		discard: List[Card] = state['discard']
		played: List[Card] = state['played_cards']

		h_vec = t.zeros(self.deck_size)
		d_vec = t.zeros(self.deck_size)
		p_vec = t.zeros(self.deck_size)

		for h in hand:
			h_vec[self.deck.index(h)] = 1
   
		for d in discard:
			d_vec[self.deck.index(d)] = 1
   
		for p in played:
			p_vec[self.deck.index(p)] = 1
		
		return t.concat([h_vec, d_vec, p_vec], dim=0)