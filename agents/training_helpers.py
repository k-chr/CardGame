from typing import Dict, List, Type, Callable
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