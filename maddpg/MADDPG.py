import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import sys
import numpy as np


class Actor(nn.Module):
    def __init__(self, states_dim, action_dim, max_action, hidden_dim):
        super(Actor, self).__init__()
        # self.device=device
        self.max_action = max_action
        self.fc1 = nn.Linear(states_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.max_action * torch.tanh(self.fc4(x))
        return x


class Critic(nn.Module):
    def __init__(self, args) -> None:
        super(Critic, self).__init__()
        self.max_action = args.high_action
        self.fc1 = nn.Linear(sum(args.obs_shape) + sum(args.action_shape), args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc4 = nn.Linear(args.hidden_dim, 1)

    def forward(self, state, action):
        state = torch.cat(state, dim=1)

        action = torch.cat(action, dim=1)
        action /= self.max_action

        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        return x


class MADDPG:
    def __init__(self, args, agent_id) -> None:
        self.args = args
        self.train_step = 0

        self.agent_id = agent_id
        self.agent_num = args.n_agents

        self.obs_dim = self.args.obs_shape[self.agent_id]
        self.action_dim = self.args.action_shape[self.agent_id]

        # learning parameters
        self.device = args.device
        # self.device='cpu'
        self.hidden_dim = args.hidden_dim
        self.max_action = args.high_action
        self.soft_tau = args.tau
        self.gamma = args.gamma

        self.actor = Actor(self.obs_dim, self.action_dim, self.max_action, self.hidden_dim).to(self.device)
        self.critic = Critic(self.args).to(self.device)

        self.actor_target = Actor(self.obs_dim, self.action_dim, self.max_action, self.hidden_dim).to(self.device)
        self.critic_target = Critic(self.args).to(self.device)
        # copy the weights to target nerworks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        # optimizer
        self.actor_optim = optim.Adam(self.actor.parameters(), args.lr_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), args.lr_critic)

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        # self.model_path = self.args.save_dir + '/' + self.args.scenario_name + '/{}_agent_weights/'.format(self.agent_num)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        
        # comment the fllowing 3 lines
        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.actor_model_name = '/{}_actor_params.pkl'.format(self.agent_num)
        self.critic_model_name = '/{}_critic_params.pkl'.format(self.agent_num)
        # self.actor_model_name = '/{}_{}_actor_params.pkl'.format(self.agent_num,self.agent_id)
        # self.critic_model_name = '/{}_{}_critic_params.pkl'.format(self.agent_num,self.agent_id)

        # 加载模型
        if os.path.exists(self.model_path + self.actor_model_name):
            self.actor.load_state_dict(torch.load(self.model_path + self.actor_model_name))
            self.critic.load_state_dict(torch.load(self.model_path + self.critic_model_name))

    # train
    def train(self, transitions: dict, other_agnets: list):
        for key in transitions.keys():
            transitions[key] = torch.as_tensor(transitions[key], dtype=torch.float32)
        observation, action, next_observation = [], [], []
        reward = transitions['r_{}'.format(self.agent_id)]
        reward = torch.Tensor(reward).unsqueeze(1).to(self.device)

        for agent_id in range(self.agent_num):
            observation.append(transitions['o_{}'.format(agent_id)].to(self.device))
            action.append(transitions['u_{}'.format(agent_id)].to(self.device))
            next_observation.append(transitions['o_next_{}'.format(agent_id)].to(self.device))

        # observation = torch.FloatTensor(observation).to(self.device)
        # action = torch.FloatTensor(action).to(self.device)
        # next_observation = torch.FloatTensor(next_observation).to(self.device)
        # calculate the target Q value function
        next_action = []

        index = 0
        # 因为传入的other_agents要比总数少一个，可能中间某个agent是当前agent，不能遍历去选择动作
        for agent_id in range(self.agent_num):
            if agent_id == self.agent_id:
                next_action.append(self.actor_target(next_observation[agent_id]).detach().to(self.device))
            else:
                next_action.append(other_agnets[index].policy.actor_target(next_observation[agent_id]).detach().to(self.device))
                index += 1

        # next_action = torch.Tensor(next_action).to(self.device)

        action[self.agent_id] = self.actor(observation[self.agent_id].to(self.device)).detach()
        policy_loss = self.critic(observation, action)
        policy_loss = -policy_loss.mean()

        target_value = self.critic_target(next_observation, next_action).detach()
        expected_value = reward + self.gamma * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)

        value = self.critic(observation, action)
        value_loss = nn.MSELoss()(value, expected_value.detach())

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        value_loss.backward()
        self.critic_optim.step()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.soft_tau) + self.soft_tau * param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.soft_tau) + self.soft_tau * param.data)

        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1

    def save_model(self, train_step):
        # if self.agent_id == self.agent_num -1:
        #     logging.info('save model')
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor.state_dict(), model_path + self.actor_model_name)
        torch.save(self.critic.state_dict(), model_path + self.critic_model_name)

    def load_model(self):
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        self.actor_network.load_state_dict(torch.load(model_path + self.actor_model_name))
        self.critic_network.load_state_dict(torch.load(model_path + self.critic_model_name))
