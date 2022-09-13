from torch import nn
import torch as t
import numpy as np
from typing import List, Any, Dict, Optional
from .training_helpers import Optimizers, build_model
from .utils import HeartsStateParser, get_legal_actions
from torch.nn import functional as F
from numpy.random._generator import Generator, default_rng
from torch.distributions import Categorical
from . import Agent, INVALID_ACTION_PENALTY

class ACAgent(nn.Module, Agent):
    def __init__(self,
                 full_deck,
                 actor_learning_rate,
                 critic_learning_rate,
                 gamma=0.99,
                 loss_decay = 0.99995,
                 critic_layers: List[int]=[],
                 actor_layers: List[int]=[],
                 rng: Generator =default_rng(2137),
                 actor_optimizer='adam',
                 critic_optimizer='adam',
                 actor_optimizer_params: Dict[str, Any] = {},
                 critic_optimizer_params: Dict[str, Any] = {},
                 activation='relu',
                 initializer='xavier_u',
                 initializer_params: Dict[str, Any] = {}):
     
        nn.Module.__init__(self)
        Agent.__init__(self, full_deck, (actor_learning_rate, critic_learning_rate), 0.0, gamma, rng)
  
        self.parser = HeartsStateParser(full_deck)
        self.episode_losses_actor = []
        self.episode_losses_critic = []
        self.state_size = self.parser.state_len
        self.action_size = 13 * 4 if full_deck else 6 * 4
        self.loss_decay = loss_decay
        activations_actor = [activation] * (len(actor_layers))
        activations_actor.append('')
        actor_layers: List[int] = [self.state_size] + actor_layers
        actor_layers.append(self.action_size)
        self.losses = []
        activations_critic = [activation] * (len(critic_layers))
        activations_critic.append('')
        critic_layers: List[int] = [self.state_size] + critic_layers
        critic_layers.append(1)
  
        self.actor = build_model(actor_layers, activations_actor, initializer, initializer_params)
        self.critic = build_model(critic_layers, activations_critic, initializer, initializer_params)

        actor_optimizer_params.update(lr=actor_learning_rate)
        self.actor_optimizer = Optimizers.get(actor_optimizer)(self.actor.parameters(), **actor_optimizer_params)

        critic_optimizer_params.update(lr=critic_learning_rate)
        self.critic_optimizer = Optimizers.get(critic_optimizer)(self.critic.parameters(), **critic_optimizer_params)
  
        self.learning_device = "cuda" if t.cuda.is_available() else 'cpu'
        self.eval_device = 'cpu'
        self = self.to(self.learning_device)
        self.I =1
  
    def set_temp_reward(self, discarded_cards: dict, point_deltas: dict):	
        super().set_temp_reward(discarded_cards, point_deltas)

    def set_final_reward(self, points: dict):
        super().set_final_reward(points)
        a_loss, c_loss = self.learn(self.previous_state, self.previous_action, -self.current_reward, self.parser.fixed_terminal_state(), True)
        self.episode_losses_actor.append(a_loss)
        self.episode_losses_critic.append(c_loss)
        # TODO sth with points in total.
        actor_loss = np.mean(self.episode_losses_actor)
        critic_loss = np.mean(self.episode_losses_critic)
        self.episode_losses_actor.clear()
        self.episode_losses_critic.clear()
        self.losses.append((actor_loss, critic_loss))
        if self.loss_callback:
            self.loss_callback((actor_loss, critic_loss))
  
    def forward(self, state:t.Tensor):
        return self.actor(state.to(self.learning_device)), self.critic(state.to(self.learning_device))
    
    @property
    def algorithm_name(self) -> str:
        return 'OSAC'	
 
    def get_name(self) -> str:
        return super().get_name() + " - Actor-Critic"
 
    def get_best_action(self, state, invalid_actions: Optional[List[int]] =None):
        possible_actions = get_legal_actions(state, not self.full_deck)
     
        state: t.Tensor =self.parser.parse(state).to(self.learning_device)
        # possible_actions = set(range(0, self.action_size))
  
        # if invalid_actions: possible_actions -= set(invalid_actions)
        possible_actions = list(possible_actions)
        with t.no_grad():
            self.eval()
            logits: t.Tensor = self(state)[0].cpu().squeeze(0).gather(0, t.as_tensor(possible_actions))
            
            m, _ = logits.max(-1)
            indices: np.ndarray = t.nonzero(logits == m).numpy().flatten()
        
        cond = indices.size == 1
        action = (possible_actions[indices[0]] if cond else possible_actions[self.rng.choice(indices)])

        return action
    
    def get_action(self, state: Any, invalid_actions: Optional[List[int]] =None):
        """
        Compute the action to take in the current state, basing on policy returned by the network.

        Note: To pick action according to the probability generated by the network
        """

        #
        # INSERT CODE HERE to get action in a given state
        # 

  
        if self.previous_state:

            a_loss, c_loss = self.learn(self.previous_state, self.previous_action, -self.current_reward, state, False)
            self.episode_losses_actor.append(a_loss)
            self.episode_losses_critic.append(c_loss)
 
        logits: t.Tensor
        possible_actions = get_legal_actions(state, not self.full_deck)
        
        state: t.Tensor =self.parser.parse(state).to(self.learning_device)
        # possible_actions = possible_actions = set(range(0, self.action_size))
  
        # if invalid_actions: possible_actions -= set(invalid_actions)
        possible_actions = list(possible_actions)
        self.last_possible_actions = possible_actions
        with t.no_grad():
            self.eval()
            logits = self(state)[0].cpu().squeeze(0)
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

    def make_move(self, game_state: dict, was_previous_move_wrong: bool):		

        if was_previous_move_wrong:
            self.current_reward = INVALID_ACTION_PENALTY
   
        act = super().make_move(game_state, was_previous_move_wrong)
        return act

    def learn(self, state, action, reward, next_state, done):
        """
        Function learn networks using information about state, action, reward and next state. 
        First the values for state and next_state should be estimated based on output of critic network.
        Critic network should be trained based on target value:
        target = r + \gamma next_state_value if not done]
        target = r if done.
        Actor network shpuld be trained based on delta value:
        delta = target - state_value
        """
        if not isinstance(state, t.Tensor): state= self.parser.parse(state).to(self.learning_device)
        if not isinstance(next_state, t.Tensor): next_state= self.parser.parse(next_state).to(self.learning_device)
        #
        # INSERT CODE HERE to train network
        #
        reward = reward
        gamma = (self.gamma)
        
        value = self.critic(state.to(self.learning_device))
        next_value = self.critic(next_state.to(self.learning_device)) if not done else t.zeros(1).to(self.learning_device)
        delta = (reward + gamma * next_value - value)
        
        critic_loss: t.Tensor =  F.smooth_l1_loss (value.flatten(), reward + gamma * next_value.flatten()) * self.I
        logits = self.actor(state.to(self.learning_device)).squeeze(0)
        probs: t.Tensor = (t.softmax(logits, dim=0)) + 1e-8
        normalized_probs = probs/probs.sum()
        log_prob: t.Tensor = t.log(normalized_probs.gather(0, t.as_tensor([action], dtype=t.int64).to(self.learning_device)))
        actor_loss: t.Tensor = (-log_prob * delta) * self.I
        
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()
  
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
  
        self.I *= self.loss_decay
  
        return -actor_loss.detach().item(), critic_loss.detach().item()
