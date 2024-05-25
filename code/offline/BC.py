import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
BC implementation of https://github.com/sfujim/TD3_BC [TD3_BC paper]
"""

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class BC(object):
    def __init__(self, state_dim, action_dim, max_action, device='cpu', **kwargs):
        self.device = device
        self.state_dim = state_dim
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.total_it = 0

    def select_action(self, state, return_batch=False):
        state = torch.FloatTensor(state).to(self.device).reshape(-1, self.state_dim)
        if return_batch:
            return self.actor(state).cpu().data.numpy()
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        pi = self.actor(state)

        actor_loss = F.mse_loss(pi, action)

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
