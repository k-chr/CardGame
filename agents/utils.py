from typing import Union, List, NamedTuple
import torch as t
import numpy as np


class Trajectory(NamedTuple):
	state: Union[t.Tensor, List[t.Tensor]]
	action: Union[int, List[int]]
	reward: Union[float, List[float]]
	prob: Union[float, List[float]]
	
def cumulative_rewards(gamma, rewards):
	l = len(rewards)
	G = [0 for _ in range(l)]
	r_t_1 = 0

	T = reversed(range(0, l))
 
	for t, r_t in zip(T, rewards[::-1]):
		r_t_1 = r_t_1 * gamma + r_t
		G[t] = r_t_1
  
	return np.asarray(G)