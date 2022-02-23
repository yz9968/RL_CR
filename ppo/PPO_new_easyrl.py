from dis import dis
from math import dist
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.hidden_dim = 128

        self.actor = nn.Sequential(
            nn.Linear(args.obs_shape[agent_id], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, args.action_shape[agent_id]),
            nn.Softmax(dim=1)
        )

    def forward(self, state):
        action_prob = self.actor(state)
        return action_prob


class Critic(nn.Module):
    def __init__(self, args, agent_id):
        super(Critic, self).__init__()
        self.hidden_dim = 128
        self.critic = nn.Sequential(
            nn.Linear(args.obs_shape[agent_id], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, state):
        value = self.critic(state)
        return value


class PPO:
    def __init__(self, args, agent_id) -> None:
        self.clip_param = 0.2
        self.max_grad_norm = 0.5
        self.update_time=10
        
        self.args = args
        self.agent_id = agent_id
        self.agent_num = args.n_agents
        self.buffer = []
        self.training_step = 0
        self.gamma = self.args.gamma
        self.batch_size = self.args.batch_size
        self.buffer_capacity = self.args.buffer_size
        self.device = args.device

        #create actor and critic network and optimizer
        self.actor = Actor(args, agent_id).to(self.device)
        self.critic = Critic(args, agent_id).to(self.device)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=3e-3)

        # create the dict for store the model
        if not os.path.isdir(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.isdir(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.isdir(self.model_path):
            os.mkdir(self.model_path)

        self.actor_model_name = '/{}_actor_params.pkl'.format(self.agent_num)
        self.critic_model_name = '/{}_critic_params.pkl'.format(self.agent_num)

        # 加载模型
        if os.path.exists(self.model_path + self.actor_model_name) and self.training_step == 0:
            self.actor.load_state_dict(torch.load(self.model_path + self.actor_model_name))
            self.critic.load_state_dict(torch.load(self.model_path + self.critic_model_name))

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_prob = self.actor(state)
        dist = Categorical(action_prob)
        action = dist.sample()
        return action.item(), action_prob[:, action.item()].item()
    
    def get_value(self, state):
        state = torch.from_numpy(state).to(self.device)
        with torch.no_grad():
            value = self.critic(state)
        return value.item()

    def save_param(self):
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor.state_dict(), model_path + self.actor_model_name)
        torch.save(self.critic.state_dict(),  model_path + self.critic_model_name)

    def store_transition(self, transition):
        self.buffer.append(transition)

    def learn(self):
        states = torch.FloatTensor([t.state for t in self.buffer]).to(self.device)
        actions = torch.LongTensor([t.action for t in self.buffer]).view(-1,1).to(self.device)
        # values=torch.FloatTensor([t.value for t in self.buffer]).to(self.device)
        rewards = [t.reward for t in self.buffer]
        old_probs = torch.FloatTensor([t.a_log_prop for t in self.buffer]).view(-1,1).to(self.device)

        R = 0
        Gt = []
        for r in rewards[::-1]:
            R = r+self.gamma*R
            Gt.insert(0, R)
        Gt = torch.FloatTensor(Gt).to(self.device)
        for i in range(self.update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic(states[index])
                delta = Gt_index - V
                advantage = delta.detach()
                #PPO
                action_prob = self.actor(states[index]).gather(1, actions[index])  # new policy
                ratio = (action_prob / old_probs[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                self.actor_optim.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optim.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.critic_optim.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()
                self.training_step += 1

        if self.training_step > 0 and self.training_step % self.args.save_rate == 0:
            self.save_param()

        del self.buffer[:]  # clear experience














