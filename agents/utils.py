from __future__ import annotations
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
    advantage: Union[float, List[float]] = None
    value: Union[float, List[float]] = None
        
def get_legal_actions(game: CardGame, agent, state):
    hand = state['hand']
    legit_hand = [card for card in hand if game._CardGame_validate(agent, card)]
    return legit_hand

def cumulative_rewards(gamma, rewards):
    l = len(rewards)
    G = [0 for _ in range(l)]
    r_t_1 = 0

    T = reversed(range(l))
 
    for t, r_t in zip(T, rewards[::-1]):
        r_t_1 = r_t_1 * gamma + r_t
        G[t] = r_t_1
  
    G = np.asarray(G)
    return G

def normalize(arr:np.ndarray):
    return (arr-arr.mean())/(arr.std() + 1e-8)

def cummulative_rewards_gae(gamma, gae_lambda, rewards, values):
    l = len(rewards)
    G = [0 for _ in range(l)]
    A = [0 for _ in range(l)]
    
    r_t_1 = 0
    a_t_1 = 0
    
    T = reversed(range(l))
    
    for t, r_t, v_t in zip(T, rewards[::-1], values[::-1]):
        r_t_1 = r_t_1 * gamma + r_t
        G[t] = r_t_1
        
        a_t_1 = r_t + gamma * a_t_1 - v_t
        A[t] = a_t_1
        a_t_1 = v_t + a_t_1 * gae_lambda
        
    return np.asarray(G), np.asarray(A)
        

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
    
    def cat(self, queue: Memory[TRAJ]):
        [self.__queue.append(traj) for traj in queue]
 
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
        self.__fixed_terminal_state = t.concat([t.zeros(self.deck_size), t.zeros(self.deck_size), t.ones(self.deck_size)], dim=0).unsqueeze(0)
        super().__init__()
  
    def fixed_terminal_state(self):
        return self.__fixed_terminal_state
    
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
        
        return t.concat([h_vec, d_vec, p_vec], dim=0).unsqueeze(0)