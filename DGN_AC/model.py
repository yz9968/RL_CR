import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
        
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
        # self.fcout = nn.Linear(hidden_dim, dout)

    def forward(self, x, mask):
        v = F.relu(self.fcv(x))
        q = F.relu(self.fcq(x))
        k = F.relu(self.fck(x)).permute(0,2,1)
        # h = torch.clamp(torch.mul(torch.bmm(q,k), mask), 0 , 9e13) - 9e15*(1 - mask)
        # h = torch.mul(torch.bmm(q,k), mask)
        h = mask.view_as(torch.mul(torch.bmm(q,k), mask))
        h = h.where(h!=0,torch.full(h.shape,-9e5).cuda())
        att = F.softmax(h, dim=2)

        out = torch.bmm(att,v)
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
    def __init__(self, input_dim, output_dim, hidden_dim):
        # self.actor = Actor(hidden_dim, action_dim, n_agent, hidden_dim)
        # input_dim: 根据输入确定
        # output_dim: action dim
        super(Actor, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        action_prob=self.actor(state)
        dist=Categorical(action_prob)
        return dist


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        # self.critic = Critic(hidden_dim + 1, obs_dim, action_dim, n_agent, hidden_dim)
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x=torch.cat((state,action),dim=-1)
        value=self.critic(x)
        return value

class AttEncoder(nn.Module):
    def __init__(self, n_agent, obs_dim, hidden_dim, action_dim) -> None:
        super(AttEncoder,self).__init__()
        self.encoder = Encoder(obs_dim, hidden_dim)
        self.att = AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim)
    
    def forward(self, x, mask):
        h1 = self.encoder(x)
        h2, a_w = self.att(h1, mask)
        return h2, a_w
