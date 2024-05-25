import numpy as np
import torch

"""
Rely on implementation of https://github.com/sfujim/TD3_BC [TD3_BC paper]
"""
class ReplayBuffer(object):
	# in offline setting, we do not need CPU to run the environment,
	# so moving all data to GPU speeds up the training.
	# That is also why we should run evaluation separately (which requires CPU)

	def __init__(self, state_dim, action_dim, max_size=int(1e6), device='cpu'):
		self.max_size = max_size

		self.device = device
		self.ptr = 0
		self.size = 0

		self.state = torch.zeros((max_size, state_dim), device=self.device, dtype=torch.float32)
		self.action = torch.zeros((max_size, action_dim), device=self.device, dtype=torch.float32)
		self.next_state = torch.zeros((max_size, state_dim), device=self.device, dtype=torch.float32)
		self.reward = torch.zeros((max_size, 1), device=self.device, dtype=torch.float32)
		self.not_done = torch.zeros((max_size, 1), device=self.device, dtype=torch.float32)


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = torch.randint(0, self.size, size=(batch_size,), device=self.device)
		return (
			self.state[ind],
			self.action[ind],
			self.next_state[ind],
			self.reward[ind],
			self.not_done[ind]
		)



	def convert_D4RL(self, dataset):
		self.state = torch.tensor(dataset['observations'], device=self.device, dtype=torch.float32)
		self.action = torch.tensor(dataset['actions'], device=self.device, dtype=torch.float32)
		self.next_state = torch.tensor(dataset['next_observations'], device=self.device, dtype=torch.float32)
		self.reward = torch.tensor(dataset['rewards'], device=self.device, dtype=torch.float32).reshape(-1,1)
		self.not_done = 1. - torch.tensor(dataset['terminals'], device=self.device, dtype=torch.float32).reshape(-1,1)
		self.size = self.state.shape[0]


	def normalize_states(self, eps = 1e-3):
		with torch.no_grad():
			mean = torch.mean(self.state, dim=0,keepdim=True)
			std = torch.std(self.state, dim=0,keepdim=True) + eps
			self.state = (self.state - mean)/std
			self.next_state = (self.next_state - mean)/std
		return mean.cpu().numpy(), std.cpu().numpy()