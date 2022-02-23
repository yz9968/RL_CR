import random
import threading
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity # 经验回放的容量
        self.buffer = [] # 缓冲区
        self.position = 0 
    
    def push(self, state, action, reward, next_state, done):
        ''' 缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity 
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) # 随机采出小批量转移
        state, action, reward, next_state, done =  zip(*batch) # 解压成状态，动作等
        return state, action, reward, next_state, done
    
    def __len__(self):
        ''' 返回当前存储的量
        '''
        return len(self.buffer)

class MultiReplayBuffer:
    def __init__(self,capacity,args) -> None:
        self.capacity=capacity
        self.current_size = 0
        self.index=0
        self.agents_num=args.n_agents
        self.args=args
        self.buffer=dict()
        for i in range(self.agents_num):
            # each agent has its dictionary
            # for each agent, the dictionary consists of four np array:o,a,r,o_n
            self.buffer['o_%d' % i] = np.empty([self.capacity, self.args.obs_shape[i]], dtype='float32')
            self.buffer['u_%d' % i] = np.empty([self.capacity, self.args.action_shape[i]], dtype='float32')
            self.buffer['r_%d' % i] = np.empty([self.capacity], dtype='float32')
            self.buffer['o_next_%d' % i] = np.empty([self.capacity, self.args.obs_shape[i]], dtype='float32')

        self.lock=threading.Lock()

    def push(self,obs,action,reward,obs_next):
        idxs = self._get_storage_idx(inc=1)  # 以transition的形式存，每次只存一条经验
        for i in range(self.agents_num):
            with self.lock:
                self.buffer['o_%d' % i][idxs] = obs[i]
                self.buffer['u_%d' % i][idxs] = action[i]
                self.buffer['r_%d' % i][idxs] = reward[i]
                self.buffer['o_next_%d' % i][idxs] = obs_next[i]

     # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.capacity:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.capacity:
            overflow = inc - (self.capacity - self.current_size)
            idx_a = np.arange(self.current_size, self.capacity)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.capacity, inc)
        self.current_size = min(self.capacity, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx





