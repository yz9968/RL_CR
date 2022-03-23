import math
import random
from turtle import forward
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.distributions import Categorical


USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **
                                                     kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


class Encoder(nn.Module):
    def __init__(self, din=32, hidden_dim=128):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(din, hidden_dim)

    def forward(self, x):
        embedding = F.relu(self.fc(x))
        return embedding


class AttModel(nn.Module):
    def __init__(self, n_node, din, hidden_dim, dout):
        super(AttModel, self).__init__()
        self.fcv = nn.Linear(din, hidden_dim)
        self.fck = nn.Linear(din, hidden_dim)
        self.fcq = nn.Linear(din, hidden_dim)
        self.fcout = nn.Linear(hidden_dim, dout)

    def forward(self, x, mask):
        v = F.relu(self.fcv(x))
        q = F.relu(self.fcq(x))
        k = F.relu(self.fck(x)).permute(0, 2, 1)
        h = torch.clamp(torch.mul(torch.bmm(q, k), mask), 0, 9e13) - 9e15*(1 - mask)
        att = F.softmax(h, dim=2)

        out = torch.bmm(att, v)
        #out = torch.add(out,v)
        #out = F.relu(self.fcout(out))
        return out, h


class Q_Net(nn.Module):
    def __init__(self, hidden_dim, dout):
        super(Q_Net, self).__init__()
        self.fc = nn.Linear(hidden_dim, dout)

    def forward(self, x):
        q = self.fc(x)
        return q


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, agent_num, hidden_dim):
        # input_dim: 根据输入确定
        # output_dim: action dim
        super(Actor, self).__init__()
        self.output_dim = output_dim
        self.agent_num = agent_num

        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, output_dim*agent_num).view(self.agent_num,self.output_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        action_prob=self.actor(x)
        dist=Categorical(action_prob)
        actions=dist.sample()
        return actions


class Critic(nn.Module):
    def __init__(self, imput_dim, obs_dim, action_dim, agent_num, hidden_dim):
        self.agent_num = agent_num
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(imput_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, action_list):
        # action_list=action_list.view((1,self.agent_num,-1))
        x=torch.cat((x,action_list),dim=2)
        value=self.critic(x)        
        return value

class DGN_actor(nn.Module):
    def __init__(self, n_agent, obs_dim, hidden_dim, action_dim):
        super(DGN_actor, self).__init__()

        self.encoder = Encoder(obs_dim, hidden_dim)
        self.att = AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim)
        # self.q_net = Q_Net(hidden_dim,num_actions)
        self.actor = Actor(hidden_dim, action_dim, n_agent, hidden_dim)

    def forward(self, x, mask):  # ,obs_all_oneline,action_all_oneline
        h1 = self.encoder(x)  # x: 1,n,n_obs,    mask: 1,n_obs,b_obs,    h1: 1, n ,hidden
        h2, a_w = self.att(h1, mask)  # h2: torch.Size([1, 6, 128])
        # q = self.q_net(h2)
        # return q, a_w
        actions = self.actor(h2)  # h2:torch.Size([1, 6, 128])
        return actions, a_w

class DGN_critic(nn.Module):
    def __init__(self, n_agent, obs_dim, hidden_dim, action_dim):
        super(DGN_critic, self).__init__()

        self.encoder = Encoder(obs_dim, hidden_dim)
        self.att = AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim)
        # self.q_net = Q_Net(hidden_dim,num_actions)
        self.critic = Critic(hidden_dim + 1, obs_dim, action_dim, n_agent, hidden_dim)

    def forward(self, x, mask, action_all_oneline):  # ,obs_all_oneline,action_all_oneline
        h1 = self.encoder(x)  # x: 1,n,n_obs,    mask: 1,n_obs,b_obs,    h1: 1, n ,hidden
        h2, a_w = self.att(h1, mask)  # h2: torch.Size([1, 6, 128])
        # q = self.q_net(h2)
        # return q, a_w
        value = self.critic(h2, action_all_oneline)  # obs_all_oneline,action_all_oneline
        return value, a_w