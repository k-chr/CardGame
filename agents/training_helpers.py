from typing import Any, Dict, List, Optional, Tuple, Type, Callable
import torch as t
from torch import nn, optim
from torch.distributions import Categorical

import numpy as np

from . import Agent

from .utils import HeartsStateParser, Memory, Trajectory

class Optimizers:
    _optimizers: Dict[str, Type[optim.Optimizer]] = {
        'adam': optim.Adam,
        'sgd': optim.SGD,
        'rms': optim.RMSprop
    }
    
    @classmethod
    def get(cls, optimizer): return Optimizers._optimizers.get(optimizer, optim.SGD)

class Activations:
    _modules: Dict[str, Callable[[],nn.Module]] = {
        'relu': lambda: nn.ReLU(),
        'leaky-relu':lambda: nn.LeakyReLU(),
        'tanh':lambda: nn.Tanh(),
        'selu':lambda: nn.SELU(),
        'sigmoid':lambda: nn.Sigmoid()
    }
 
    @classmethod
    def get(cls, activation): return Activations._modules.get(activation, lambda: nn.Identity())()
 
class Initializers:
    _initializers: Dict[str, Callable[[t.Tensor], t.Tensor]] = {
        'kaiming_n': nn.init.kaiming_normal_,
        'kaiming_u': nn.init.kaiming_uniform_,
        'xavier_n': nn.init.xavier_normal_,
        'xavier_u': nn.init.xavier_uniform_,
        'const': nn.init.constant_,
        'normal': nn.init.normal_,
        'uniform':nn.init.uniform_
    }
    
    @classmethod
    def get(cls, initializer): return Initializers._initializers.get(initializer, lambda x: x)
 
 
def build_model(layers: List[int], activations: List[str], initializer: str, initializer_params={}):
     
    layer_init = lambda _in, _out, activation=None: nn.Sequential(
        nn.Linear(_in, _out), Activations.get(activation)
    )
  
    qnet = nn.Sequential(*[
         layer_init(_in, _out, _activation) for _in, _out, _activation in zip(layers[:-1], layers[1:], activations)
    ])
  

    for module in qnet.modules():
        if isinstance(module, nn.Linear): 
            Initializers.get(initializer)(module.weight, **initializer_params)
            Initializers.get('const')(module.bias, val=0)

    return qnet


class Worker(nn.Module, Agent):
    def __init__(self, policy: nn.Module, critic: nn.Module, action_size, parser: HeartsStateParser, callback):
        nn.Module.__init__(self)
        Agent.__init__(self, False, 0.0, 0.0, 0.0, None)
        self.policy = policy
        self.critic = critic
        self.rollouts = Memory[Trajectory](None, Trajectory)
        self.parser = parser
        self.callback = callback
        self.training = True
        self.action_size = action_size
        self.last_prob = 0.0
        self.last_val = 0.0
        self.learning_device = "cuda" if t.cuda.is_available() else 'cpu'
        
    def get_action(self, state: Any, invalid_actions: Optional[List[int]] =None): 
        logits: t.Tensor
        state: t.Tensor =self.parser.parse(state)
        possible_actions = possible_actions = set(range(0, self.action_size))
  
        if invalid_actions: possible_actions -= set(invalid_actions)
        possible_actions = list(possible_actions)
        self.last_possible_actions = possible_actions
        with t.no_grad():
            self.eval()
            logits, value = self(state)
            logits = logits.cpu().squeeze(0)
            probs: t.Tensor = t.softmax(logits, dim=0)	
        
        self.train()
        possible_actions = list(possible_actions)
        probs_gathered: np.ndarray = probs.gather(0, t.as_tensor((possible_actions))) +1e-8
        probs_gathered = probs_gathered/probs_gathered.sum()
        dist = Categorical(probs_gathered)
        idx = dist.sample().item()
        action = possible_actions[idx]
        self.last_prob = probs_gathered[idx].item()
        self.last_val = value.item()
        return action
    
    def get_best_action(self, state, invalid_actions: Optional[List[int]] =None): return self.get_action(state, invalid_actions)
    
    @property 
    def algorithm_name(self) -> str:
        return 'PPO'
    
    def get_name(self) -> str:
        return super().get_name() + " - PPO-worker"
    
    def remember(self, state, action, reward):
        if not isinstance(state, t.Tensor): state= self.parser.parse(state)
        #Function adds information to the memory about last action and its results
        self.rollouts.store(Trajectory(state, action, reward, self.last_prob, value=self.last_val))
    
    def set_temp_reward(self, discarded_cards: dict, point_deltas: dict):	
        super().set_temp_reward(discarded_cards, point_deltas)
        if not self.training: return
  
        self.remember(self.parser.parse(self.previous_state), self.previous_action, -self.current_reward)

    def set_final_reward(self, points: dict):
        super().set_final_reward(points)
        self.callback(self)
        
    def forward(self, state:t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        return self.policy(state.to(self.learning_device)), self.critic(state.to(self.learning_device))  