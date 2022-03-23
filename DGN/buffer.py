import numpy as np
class ReplayBuffer(object):

	def __init__(self, buffer_size, obs_space, n_action, n_agent):
		self.buffer_size = buffer_size
		self.n_agent = n_agent
		self.pointer = 0
		self.len = 0
		self.actions = np.zeros((self.buffer_size,self.n_agent),dtype = np.int32)
		self.rewards = np.zeros((self.buffer_size,n_agent))
		self.dones = np.zeros((self.buffer_size,n_agent),dtype = np.int32)
		self.obs = np.zeros((self.buffer_size,self.n_agent,obs_space), dtype='float32')
		self.next_obs = np.zeros((self.buffer_size,self.n_agent,obs_space), dtype='float32')
		self.matrix = np.zeros((self.buffer_size,self.n_agent,self.n_agent), dtype='float32')
		self.next_matrix = np.zeros((self.buffer_size,self.n_agent,self.n_agent), dtype='float32')

	def getBatch(self, batch_size):

		index = np.random.choice(self.len, batch_size, replace=False)
		return self.obs[index], self.actions[index], self.rewards[index], self.next_obs[index], self.matrix[index], self.next_matrix[index], self.dones[index]

	def add(self, obs, action, reward, next_obs, matrix, next_matrix, done):

		self.obs[self.pointer] = obs
		self.actions[self.pointer] = action
		self.rewards[self.pointer] = reward
		self.next_obs[self.pointer] = next_obs
		self.matrix[self.pointer] = matrix
		self.next_matrix[self.pointer] = next_matrix
		self.dones[self.pointer] = done
		self.pointer = (self.pointer + 1) % self.buffer_size
		self.len = min(self.len + 1, self.buffer_size)
