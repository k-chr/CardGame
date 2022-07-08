from torch import nn
import torch as t
import numpy as np
from typing import List, Any, Dict, Optional, Tuple
from .utils import HeartsStateParser, Memory, Trajectory, cummulative_rewards, normalize
from .training_helpers import Optimizers, build_model, Worker
from . import INVALID_ACTION_PENALTY, Agent
from torch.distributions import Categorical
from numpy.random._generator import Generator, default_rng
from torch.nn import functional as F

class REINFORCEAgent(nn.Module, Agent):
    def __init__(self, 
                 batch_size: int,
                 full_deck,
                 learning_rate,
                 baseline_learning_rate,
                 gamma = 0.95,
                 importance_weighting = False,
                 queue_size = 2000,
                 layers: List[int] =[],
                 baseline_layers: List[int] = [],
                 rng: Generator =default_rng(2137),
                 optimizer='adam',
                 baseline_optimizer='adam',
                 optimizer_params: Dict[str, Any] = {},
                 baseline_optimizer_params: Dict[str, Any] ={},
                 activation='relu',
                 initializer='xavier_u',
                 initializer_params: Dict[str, Any] = {}):
     
        nn.Module.__init__(self)
        Agent.__init__(self, full_deck, (learning_rate, baseline_learning_rate), 0.0, gamma, rng)
        parser = HeartsStateParser(full_deck)
        self.batch_size = batch_size
        state_size = parser.state_len
        self.losses = []
        self.parser = parser
        self.state_size = state_size
        self.action_size = 13 * 4 if full_deck else 6 * 4
  
        self.rollouts = Memory[Trajectory](None, Trajectory)
        self.memory = Memory[Trajectory](queue_size, Trajectory) if importance_weighting else None 
  
        self.last_prob: float = 0.0
        self.importance_weighting = importance_weighting
  
        activations = [activation] * (len(layers))
        activations.append('')
        activations_baseline = [activation] * (len(baseline_layers))
        activations_baseline.append('')
  
        layers: List[int] = [state_size] + layers
        layers.append(self.action_size)
        baseline_layers: List[int] = [state_size] + baseline_layers
        baseline_layers.append(1)
  
        self.qnet = build_model(layers, activations, initializer, initializer_params)
        self.baseline = build_model(baseline_layers, activations_baseline, initializer, initializer_params)

        optimizer_params.update(lr=learning_rate)
        self.optimizer = Optimizers.get(optimizer)(self.qnet.parameters(), **optimizer_params)
        baseline_optimizer_params.update(lr=baseline_learning_rate)
        self.baseline_optimizer = Optimizers.get(baseline_optimizer)(self.baseline.parameters(), **baseline_optimizer_params)
  
        self.learning_device = "cuda" if t.cuda.is_available() else 'cpu'
        self.eval_device = 'cpu'
        self = self.to(self.learning_device)

        self.workers: List[Worker] = []
        self.served = False
        self.workers_served = {}
        
    def _worker_callback(self, worker):
        self.workers_served[worker] = True
        
        if self.served and all(list(self.workers_served.values())):
            a_loss, c_loss = self.replay()
            self.losses.append((a_loss, c_loss))
            if self.loss_callback:
                self.loss_callback((a_loss, c_loss))
                
    def spawn_worker(self) -> Agent:
        w = Worker(self.qnet, self.baseline, self.action_size, self.parser, self._worker_callback)
        self.workers.append(w)
        self.workers_served[w] = False
        return w 
        
    def set_temp_reward(self, discarded_cards: dict, point_deltas: dict):	
        super().set_temp_reward(discarded_cards, point_deltas)
        if not self.training: return
  
        self.remember(self.parser.parse(self.previous_state), self.previous_action, -self.current_reward)
    
    def set_final_reward(self, points: dict):
        super().set_final_reward(points)
        # TODO sth with points in total.
        if not self.training: return
        loss = self.replay()
        self.losses.append(loss)
        if self.loss_callback:
            self.loss_callback(loss)
    
    @property 
    def algorithm_name(self) -> str:
        return 'REINFORCE'
 
    def remember(self, state, action, reward):
        if not isinstance(state, t.Tensor): state= self.parser.parse(state)
        #Function adds information to the memory about last action and its results
        self.rollouts.store(Trajectory(state, action, reward, self.last_prob, self.last_val))
    
    def forward(self, state):
        return self.qnet(state.to(self.learning_device))

    def get_name(self) -> str:
        return super().get_name() + " - REINFORCE"
    
    def make_move(self, game_state: dict, was_previous_move_wrong: bool):
        last_prob = self.last_prob
        

        if was_previous_move_wrong and self.training:
            self.rollouts.store(Trajectory(self.parser.parse(game_state), self.previous_action, -INVALID_ACTION_PENALTY, last_prob, self.last_val))
   
        act = super().make_move(game_state, was_previous_move_wrong)
        return act
 
    def get_action(self, state, invalid_actions: Optional[List[int]] = None):
        """
        Compute the action to take in the current state, basing on policy returned by the network.

        Note: To pick action according to the probability generated by the network
        """

        #
        # INSERT CODE HERE to get action in a given state
        # 
        state: t.Tensor =self.parser.parse(state)
        possible_actions = set(range(0, self.action_size))
  
        if invalid_actions:
            possible_actions -= set(invalid_actions)

        with t.no_grad():
            self.eval()
            logits: t.Tensor = self(state).cpu().squeeze(0)
            self.last_val = self.baseline(state).cpu().item()
            probs: t.Tensor = t.softmax(logits, dim=0)
   
        self.train()
        possible_actions = list(possible_actions)
        probs_gathered: np.ndarray = probs.gather(0, t.as_tensor((possible_actions))) +1e-8
        probs_gathered = probs_gathered/probs_gathered.sum()
        dist = Categorical(probs_gathered)
        idx = dist.sample().item()
        action = possible_actions[idx]
        self.last_prob = probs_gathered[idx].item()
        
        return action

    def get_best_action(self, state, invalid_actions: Optional[List[int]] = None):
        state: t.Tensor =self.parser.parse(state)
        possible_actions = set(range(0, self.action_size))
  
        if invalid_actions:
            possible_actions -= set(invalid_actions)

        possible_actions = list(possible_actions)

        with t.no_grad():
            self.eval()
            logits: t.Tensor = self(state).cpu().squeeze(0).gather(0, t.as_tensor(possible_actions))
            
            m, _ = logits.max(-1)
            indices: np.ndarray = t.nonzero(logits == m).numpy().flatten()
        
        cond = indices.size == 1
        action = (possible_actions[indices[0]] if cond else possible_actions[self.rng.choice(indices)])

        return action

    def replay(self):
        """
        Function learn network using data stored in state, action and reward memory. 
        First calculates G_t for each state and train network
        """
        #
        # INSERT CODE HERE to train network
        #
        
        batch_size = self.batch_size
        trajectories = self.rollouts.get(list(range(len(self.rollouts))))

        rewards = cummulative_rewards(self.gamma, trajectories.reward[:-1] + (-self.current_reward, ))
        advantage = rewards - values
        advantage = normalize(advantage)
        rewards = normalize(rewards)
        trajectories = Trajectory(trajectories.state, trajectories.action,  tuple(rewards), trajectories.prob, tuple(advantage), trajectories.value)
        self.rollouts.set_items(list(zip(*trajectories)))
        
        for worker in self.workers:
            worker_trajectories = worker.rollouts.get(list(range(len(worker.rollouts))))
            rewards = cummulative_rewards(self.gamma, worker_trajectories.reward[:-1] + (-worker.current_reward, ))
            advantage = rewards - worker_trajectories.value
            advantage = normalize(advantage)
            rewards = normalize(rewards)
            
            trajectories = Trajectory(worker_trajectories.state, worker_trajectories.action, tuple(rewards), worker_trajectories.prob, tuple(advantage), worker_trajectories.value)
            self.rollouts.cat(list(zip(*trajectories)))   
        
        if self.importance_weighting:
            for tup in zip(trajectories.state, trajectories.action, trajectories.reward, trajectories.prob):
                self.memory.store(tup)
        mem = self.memory if self.importance_weighting else self.rollouts
        count_mem = len(mem)
        
        batch_size = min(count_mem, batch_size)
        
        batch = mem.sample(batch_size)
        states = t.stack(batch.state).to(self.learning_device).squeeze(1)
        actions = t.as_tensor(batch.action, dtype=t.int64, device=self.learning_device).unsqueeze(1)
        rewards = t.as_tensor(batch.reward, device=self.learning_device).unsqueeze(1)
        probs = t.as_tensor(batch.prob, device=self.learning_device).unsqueeze(1)
   
        with t.no_grad():
            self.eval()
            current_policy: t.Tensor = t.softmax(self(states), dim=1)
            current_policy_a = current_policy.gather(1, actions)
            importance_weight = (current_policy_a + 1e-8) / (probs + 1e-8)
            values = self.baseline(states)
   
        self.train()
  
        
        predicted_probs = t.softmax(self(states), dim=1) + 1e-8
        predicted_probs = predicted_probs/predicted_probs.sum()
        dist = Categorical(predicted_probs)
        predicted: t.Tensor = dist.log_prob(actions)
        
        loss = -predicted * importance_weight * advantage
        loss = t.mean(loss)
        baseline_loss = F.smooth_l1_loss(self.baseline(states), rewards)
  
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
  
        self.baseline_optimizer.zero_grad()
        baseline_loss.backward()
        self.baseline_optimizer.step()
  
        self.rollouts.clear()
        return -loss.cpu().item(), baseline_loss.cpu().item()
