from typing import Dict, Type, Callable
import torch as t
from torch import nn, optim

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