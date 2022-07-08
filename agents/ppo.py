from . import Agent, INVALID_ACTION_PENALTY
from torch import nn
from torch.nn import functional as F
import torch as t
import numpy as np
from typing import List, Any, Dict, Optional, Tuple
from .utils import HeartsStateParser, Memory, Trajectory, cummulative_rewards_gae, normalize
from .training_helpers import Optimizers, build_model
from torch.distributions import Categorical
from numpy.random._generator import Generator, default_rng

CLIP_EPSILON = 0.1

class _PPOWorker(nn.Module, Agent):
    def __init__(self, policy: nn.Module, critic: nn.Module, lambda_gae, gamma, action_size, parser: HeartsStateParser, callback):
        nn.Module.__init__(self)
        Agent.__init__(self, False, 0.0, 0.0, gamma, None)
        self.policy = policy
        self.critic = critic
        self.lambda_gae = lambda_gae
        self.gamma = gamma
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
        
class PPOAgent(nn.Module, Agent):
    def __init__(self, 
                 batch_size: int,
                 full_deck,
                 actor_learning_rate,
                 critic_learning_rate,
                 update_interval=5,
                 gamma = 0.95,
                 lambda_gae = 0.98,
                 entropy_coeff = 0.02,
                 actor_layers: List[int]=[],
                 critic_layers: List[int]=[],
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
        self.update_interval = update_interval
        self.episode_losses_actor = []
        self.episode_losses_critic = []
        self.state_size = self.parser.state_len
        self.action_size = 13 * 4 if full_deck else 6 * 4
        activations_actor = [activation] * (len(actor_layers))
        activations_actor.append('')
        actor_layers: List[int] = [self.state_size] + actor_layers
        actor_layers.append(self.action_size)
        self.losses = []
        self.workers_served = {}
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
        self.workers: List[_PPOWorker] = []
        self.served = False
        self.lambda_gae = lambda_gae
        self.entropy_coeff = entropy_coeff
        self.batch_size = batch_size
        self.rollouts = Memory[Trajectory](None, Trajectory)
        self = self.to(self.learning_device)
        self.last_prob = 0.0
        
     
    def forward(self, state:t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        return self.actor(state.to(self.learning_device)), self.critic(state.to(self.learning_device))   
        
    def get_name(self) -> str:
        return super().get_name() + " - PPO"
    
    def _worker_callback(self, worker):
        self.workers_served[worker] = True
        
        if self.served and all(list(self.workers_served.values())):
            a_loss, c_loss = self.replay()
            self.losses.append((a_loss, c_loss))
            if self.loss_callback:
                self.loss_callback((a_loss, c_loss)) 
    
    def spawn_worker(self) -> Agent:
        w = _PPOWorker(self.actor, self.critic, self.lambda_gae, self.gamma, self.action_size, self.parser, self._worker_callback)
        self.workers.append(w)
        self.workers_served[w] = False
        return w    
        
    def get_best_action(self, state, invalid_actions: Optional[List[int]] =None):
        state: t.Tensor =self.parser.parse(state)
        possible_actions = set(range(0, self.action_size))
  
        if invalid_actions: possible_actions -= set(invalid_actions)
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

    def set_temp_reward(self, discarded_cards: dict, point_deltas: dict):	
        super().set_temp_reward(discarded_cards, point_deltas)
        if not self.training: return
  
        self.remember(self.parser.parse(self.previous_state), self.previous_action, -self.current_reward)

    def set_final_reward(self, points: dict):
        super().set_final_reward(points)
        # TODO sth with points in total.
        if not self.training: return
        self.served = True
        if self.served and all(list(self.workers_served.values())):
            a_loss, c_loss = self.replay()
            self.losses.append((a_loss, c_loss))
            if self.loss_callback:
                self.loss_callback((a_loss, c_loss))

    def remember(self, state, action, reward):
        if not isinstance(state, t.Tensor): state= self.parser.parse(state)
        #Function adds information to the memory about last action and its results
        self.rollouts.store(Trajectory(state, action, reward, self.last_prob, value=self.last_val))
    
    def make_move(self, game_state: dict, was_previous_move_wrong: bool):		

        last_prob = self.last_prob
        

        if was_previous_move_wrong and self.training:
            self.rollouts.store(Trajectory(self.parser.parse(game_state), self.previous_action, -INVALID_ACTION_PENALTY, last_prob, value=self.last_val))
   
        act = super().make_move(game_state, was_previous_move_wrong)
        return act    
     
    @property 
    def algorithm_name(self) -> str:
        return 'PPO'  
        
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

        rewards, advantage = cummulative_rewards_gae(self.gamma, self.lambda_gae, trajectories.reward[:-1] + (-self.current_reward, ), trajectories.value)
        advantage = normalize(advantage)
        rewards = normalize(rewards)
        trajectories = Trajectory(trajectories.state, trajectories.action,  tuple(rewards), trajectories.prob, tuple(advantage), trajectories.value)
        self.rollouts.set_items(list(zip(*trajectories)))
        
        for worker in self.workers:
            worker_trajectories = worker.rollouts.get(list(range(len(worker.rollouts))))
            rewards, advantage = cummulative_rewards_gae(self.gamma, self.lambda_gae, worker_trajectories.reward[:-1] + (-worker.current_reward, ), worker_trajectories.value)
            advantage = normalize(advantage)
            rewards = normalize(rewards)
            
            trajectories = Trajectory(worker_trajectories.state, worker_trajectories.action, tuple(rewards), worker_trajectories.prob, tuple(advantage), worker_trajectories.value)
            self.rollouts.cat(list(zip(*trajectories)))            
                
        mem = self.rollouts
        count_mem = len(mem)
        
        batch_size = min(count_mem, batch_size)
        actor_losses = []
        critic_losses = []
        for _ in range(self.update_interval):
            
            batch = mem.sample(batch_size)
            states = t.stack(batch.state).to(self.learning_device).squeeze(1)
            actions = t.as_tensor(batch.action, dtype=t.int64, device=self.learning_device).unsqueeze(1)
            rewards = t.as_tensor(batch.reward, device=self.learning_device).unsqueeze(1)
            probs = t.as_tensor(batch.prob, device=self.learning_device).unsqueeze(1)
            advantage = t.as_tensor(batch.advantage, device=self.learning_device).unsqueeze(1)        
                
            predicted_probs = t.softmax(self.actor(states), dim=1) + 1e-8
            predicted_probs = predicted_probs/predicted_probs.sum()
            current_policy_a = predicted_probs.gather(1, actions)
            importance_weight = (t.log(current_policy_a) - t.log(probs + 1e-8)).exp()
            dist = Categorical(predicted_probs)
            entropy: t.Tensor = dist.entropy()

            clipped = advantage * importance_weight.clamp(1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
            not_clipped = advantage * importance_weight
            
            actor_loss = -t.min(clipped, not_clipped)
            actor_loss = t.mean(actor_loss) + t.mean(entropy) * self.entropy_coeff
            
            critic_loss = F.smooth_l1_loss(self.critic(states), rewards)
    
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
    
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_losses.append(actor_loss.cpu().item())
            critic_losses.append(critic_loss.cpu().item())
        self.rollouts.clear()
        self.served = False

        for worker in self.workers:
            worker.rollouts.clear()
            self.workers_served[worker] = False
        
        return np.mean(actor_losses), np.mean(critic_losses)