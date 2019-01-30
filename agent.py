import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from replay_buffer import ReplayBuffer
from noise import OUNoise
from model import Actor, Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():

    def __init__(self, state_size, action_size):
        
        # Constants
        self.buffer_size = int(1e6)
        self.batch_size = 128
        self.learning_rate = 1e-4
        self.learn_every = 2
        self.learning_rounds = 4

        self.gamma = 0.99
        self.tau = 1e-3

        self.t = 0
        self.state_size = state_size
        self.action_size = action_size
        self.eps = 5.0
        self.eps_decay = 1 / (300 * self.learning_rounds)

        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.learning_rate)

        self.critic_local = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.learning_rate)

        self.noise = OUNoise((1, action_size))
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size)

    def step(self, state, action, reward, next_state, done, agent_number):
        self.t += 1

        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > self.batch_size and self.t % self.learn_every == 0:
                for _ in range(self.learning_rounds):
                    experiences = self.memory.sample()
                    self.learn(experiences, self.gamma, agent_number)

    def act(self, states, add_noise):
        states = torch.from_numpy(states).to(device).float()
        
        # Get the actions for this agent
        with torch.no_grad():
            actions = self.actor_local(states.squeeze()).unsqueeze(0).cpu().data.numpy()

        if add_noise:
            actions += self.eps * self.noise.sample()

        actions = np.clip(actions, -1, 1)
        return actions

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, agent_number):
        states, actions, rewards, next_states, dones = experiences

        # Find the best action according to target network
        actions_next = self.actor_target(next_states)
        if agent_number == 0:
            #Get the first two actions
            actions_next = torch.cat((actions_next, actions[:,2:]), dim=1)
        else:
            #Get the second two action
            actions_next = torch.cat((actions[:,:2], actions_next), dim=1)

        # Compute Q targets for current states
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        # Compute loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Clip the gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # Find the best action according to local network
        actions_pred = self.actor_local(states)
        if agent_number == 0:
            #Get the first two actions
            actions_pred = torch.cat((actions_pred, actions[:,2:]), dim=1)
        else:
            #Get the second two actions
            actions_pred = torch.cat((actions[:,:2], actions_pred), dim=1)

        # Compute loss
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target network ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

        # Update noise param eps
        self.eps -= self.eps_decay
        self.eps = max(self.eps, 0)
        self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

