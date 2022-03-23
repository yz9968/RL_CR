import os
import time
import numpy as np
import logging
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from DGN1.model import DGN_actor,DGN_critic
from DGN1.buffer import ReplayBuffer
from common.plot import plot_multi_cn, plot_cn_off,plot_cn,plot_multi_cn_off

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

class Runner_DGN_AC:
    def __init__(self, args, env):
        self.args = args
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info('Using device: %s', self.device)
        self.env = env
        self.epsilon = args.epsilon
        self.num_episode = args.num_episodes
        self.max_step = args.max_episode_len
        self.agents = self.env.agents
        self.agent_num = self.env.agent_num
        self.n_action = args.action_shape[0]
        self.n_obs=args.obs_shape[0]
        self.hidden_dim = 128
        self.buffer = ReplayBuffer(args.buffer_size, self.n_obs, self.n_action, self.agent_num)
        self.lr_actor = 1e-4
        self.lr_critic= 1e-3
        self.soft_tau = 1e-2 # 软更新参数
        self.batch_size = args.batch_size
        self.train_epoch = 25
        self.gamma = args.gamma
        self.observation_space = self.env.observation_space
        # self.model = DGN(self.agent_num, self.observation_space, self.hidden_dim, self.n_action).to(self.device)
        # self.model_tar = DGN(self.agent_num, self.observation_space, self.hidden_dim, self.n_action).to(self.device)
        self.actor=DGN_actor(self.agent_num, self.observation_space, self.hidden_dim, self.n_action).to(self.device)
        self.critic=DGN_critic(self.agent_num, self.observation_space, self.hidden_dim, self.n_action).to(self.device)
        self.actor_tar=DGN_actor(self.agent_num, self.observation_space, self.hidden_dim, self.n_action).to(self.device)
        self.critic_tar=DGN_critic(self.agent_num, self.observation_space, self.hidden_dim, self.n_action).to(self.device)
        
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name +'/{}_agent_att{}/'.format(self.agent_num,args.attach_info)
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
        self.actor_model_name = '{}_GRL_actor_weight.pth'.format(self.agent_num)
        self.critic_model_name = '{}_GRL_critic_weight.pth'.format(self.agent_num)
        if os.path.exists(self.save_path + self.actor_model_name):
            self.actor.load_state_dict(torch.load(self.save_path + self.actor_model_name))
        if os.path.exists(self.save_path + self.critic_model_name):
            self.critic.load_state_dict(torch.load(self.save_path + self.critic_model_name))

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optim =optim.Adam(self.critic.parameters(),lr=self.lr_critic)

        # 复制参数到目标网络
        for target_param, param in zip(self.critic_tar.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.actor_tar.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

    def run(self):
        KL = nn.KLDivLoss()
        lamb = 0.1
        tau = 0.98
        reward_total = []
        conflict_total = []
        collide_wall_total = []
        success_total = []
        nmac_total = []
        start_episode = 40
        start = time.time()
        episode = -1
        rl_actor_model_dir = self.save_path + self.actor_model_name
        rl_critic_model_dir = self.save_path + self.critic_model_name
        while episode < self.num_episode:
            if episode > start_episode:
                self.epsilon = max(0.02, self.epsilon - 0.00016)

            episode += 1
            step = 0
            obs, adj = self.env.reset()
            logging.info("current episode {}".format(episode))
            while step < self.max_step:
                if not self.env.simulation_done:
                    step += 1

                    obs_tensor=torch.FloatTensor(obs).view((1,self.agent_num,self.n_obs)).to(self.device) # torch.Size([1, 6, 9])
                    adj_tensor=torch.FloatTensor(adj).view((1,self.agent_num,self.agent_num)).to(self.device) # torch.Size([1, 6, 6])
                    # select action
                    # TODO 激活函数选择，负状态值的影响
                    actions, a_ws = self.actor(obs_tensor,adj_tensor)  # actions: (1,6) 
                    actions_line=torch.unsqueeze(actions,dim=2) # actions_line: [6]
                    actions=list(torch.squeeze(actions).cpu().numpy())

                    next_obs, next_adj, reward, done_signals, info = self.env.step(actions)

                    value, a_wt=self.critic(obs_tensor,adj_tensor,actions_line) # value: (1,6)  
                    value_line=torch.squeeze(value)  # value_line: [6] 
                    values=list(value_line.cpu().detach().numpy())

                    self.buffer.add(obs, actions,values, reward, next_obs, adj, next_adj,[info['simulation_done']]*self.agent_num)
                    obs = next_obs
                    adj = next_adj

                else:
                    if self.env.simulation_done:
                        logging.info("all agents done!")
                    break

            if episode > 0 and episode % self.args.evaluate_rate == 0:
                rew, info = self.evaluate()
                reward_total.append(rew)
                conflict_total.append(info[0])
                collide_wall_total.append(info[1])
                success_total.append(info[2])
                nmac_total.append(info[3])

            if episode < start_episode:
                continue

            for epoch in range(self.train_epoch):
                Obs, Action,Values,Reward, Next_Obs, Mat, Next_Mat, Done = self.buffer.getBatch(self.batch_size)
                # Obs, Action,Values,Reward, Next_Obs, Mat, Next_Mat, Done = self.buffer.getBatch(3)
                Obs = torch.Tensor(Obs).to(self.device)
                Mat = torch.Tensor(Mat).to(self.device)
                Action = torch.Tensor(Action).unsqueeze(2).to(self.device)
                Values = torch.Tensor(Values).unsqueeze(2).to(self.device)
                # Action=torch.unsqueeze(Action,dim=2) # actions_line: [6]
                Next_Obs = torch.Tensor(Next_Obs).to(self.device)
                Next_Mat = torch.Tensor(Next_Mat).to(self.device)
                Reward = torch.FloatTensor(Reward).unsqueeze(2).to(self.device) 
                Done = torch.FloatTensor(np.float32(Done)).unsqueeze(2).to(self.device)

                actions,actor_attention=self.actor(Obs,Mat)
                actor_attention=F.log_softmax(actor_attention,dim=2)
                # actions=torch.unsqueeze(actions,dim=2)
                actions.unsqueeze_(dim=2)
                policy_loss, policy_attention=self.critic(Obs,Mat,actions)

                next_action,target_actor_attention=self.actor_tar(Next_Obs,Next_Mat)
                target_actor_attention=F.log_softmax(target_actor_attention,dim=2)
                
                actor_loss_kl = F.kl_div(actor_attention, target_actor_attention.detach(), reduction='batchmean')
                # # 加 advantage

                # # actor critic attention 是各自计算损失，还是对应计算损失 
                policy_loss = -policy_loss.mean()  + lamb * 0.5 * actor_loss_kl

                # next_action=torch.unsqueeze(next_action,dim=2) # actions_line: [6]
                next_action.unsqueeze_(dim=2) # actions_line: [6]
                target_value,target_critic_attention = self.critic_tar(Next_Obs,Next_Mat, next_action.detach())
                target_critic_attention=F.log_softmax(target_critic_attention,dim=2)
                expected_value=Reward + (1-Done)*self.gamma*target_value
                expected_value = torch.clamp(expected_value, -np.inf, np.inf)

                value,critic_attention=self.critic(Obs,Mat,Action)
                critic_attention=F.log_softmax(critic_attention,dim=2)
                critic_loss_kl = F.kl_div(critic_attention, target_critic_attention.detach(), reduction='batchmean')

                value_loss=nn.MSELoss()(value,expected_value.detach()) + lamb * critic_loss_kl

                self.actor_optim.zero_grad()
                policy_loss.backward()
                self.actor_optim.step()
                self.critic_optim.zero_grad()
                value_loss.backward()
                self.critic_optim.step()
                # 软更新
                for target_param, param in zip(self.critic_tar.parameters(), self.critic.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)
                for target_param, param in zip(self.actor_tar.parameters(), self.actor.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

            if episode != 0 and episode % self.args.save_rate == 0:
                torch.save(self.actor.state_dict(), rl_actor_model_dir)
                torch.save(self.critic.state_dict(), rl_critic_model_dir)
                logging.info("torch save model for rl_weight")

        end = time.time()
        logging.info("time cost: {}".format(end-start))

        plot_cn_off(reward_total, 'rewards', 'evaluate num', 'average rewards', self.save_path,
                save_name='{}_train_rewards.png'.format(self.agent_num))
        np.save(self.save_path + '/{}_train_rwards'.format(self.agent_num), reward_total)

        plot_multi_cn_off(
            [conflict_total, collide_wall_total, success_total, nmac_total],
            ["总冲突数量", "出界数量", "成功数量", "nmac数量"],
            ['evaluate num']*4,
            ["confilt_num", "exit_boundary_num", "success_num", "nmac_total"],
            save_path=self.save_path,
            save_name='{}_train_metrix'.format(self.agent_num)
        )
        np.save(self.save_path + '/{}_train_conflict_total.pkl'.format(self.agent_num), conflict_total)
        np.save(self.save_path + '/{}_train_metirx.pkl'.format(self.agent_num), [conflict_total, collide_wall_total, success_total, nmac_total])

    def evaluate(self):
        logging.info("evaluate during training")
        returns = []
        deviation = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            obs, adj = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                if not self.env.simulation_done:
                    if self.args.render==True:
                        self.env.m_render(self.args.render_mode)
                    obs_tensor=torch.FloatTensor(obs).view((1,self.agent_num,self.n_obs)).to(self.device) # torch.Size([1, 6, 9])
                    adj_tensor=torch.FloatTensor(adj).view((1,self.agent_num,self.agent_num)).to(self.device) # torch.Size([1, 6, 6])
                    # select action
                    actions, a_ws = self.actor(obs_tensor,adj_tensor)  # actions: (1,6) 
                    actions=list(torch.squeeze(actions).cpu().numpy())

                    next_obs, next_adj, reward, done_signals, info = self.env.step(actions)
                    rewards += sum(reward)
                    obs = next_obs
                    adj = next_adj

                else:
                    dev = self.env.route_deviation_rate()
                    deviation.append(np.mean(dev))
                    break

            rewards = rewards / 10000
            returns.append(rewards)
            print('Returns is', rewards)
        print("conflict num :", self.env.collision_num)
        print("nmac num :", self.env.nmac_num)
        print("exit boundary num：", self.env.exit_boundary_num)
        print("success num：", self.env.success_num)
        print("路径平均偏差率：", np.mean(deviation))

        return sum(returns) / self.args.evaluate_episodes, (self.env.collision_num, self.env.exit_boundary_num, self.env.success_num, self.env.nmac_num)

    def evaluate_model(self):
        """
        对现有最新模型进行评估
        :return:
        """
        logging.info("now evaluate the model")
        time_num = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
        conflict_total = []
        collide_wall_total = []
        success_total = []
        nmac_total = []
        deviation = []
        returns = []
        eval_episode = self.args.eval_model_episode
        for episode in range(eval_episode):
            logging.info("eval model, episode {}".format(episode))
            # reset the environment
            obs, adj = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                if not self.env.simulation_done:
                    if self.args.render==True and episode % 20 ==0:
                        self.env.m_render(self.args.render_mode)
                    obs_tensor=torch.FloatTensor(obs).view((1,self.agent_num,self.n_obs)).to(self.device) # torch.Size([1, 6, 9])
                    adj_tensor=torch.FloatTensor(adj).view((1,self.agent_num,self.agent_num)).to(self.device) # torch.Size([1, 6, 6])
                    # select action
                    actions, a_ws = self.actor(obs_tensor,adj_tensor)  # actions: (1,6) 
                    actions=list(torch.squeeze(actions).cpu().numpy())

                    next_obs, next_adj, reward, done_signals, info = self.env.step(actions)
                    rewards += sum(reward)
                    obs = next_obs
                    adj = next_adj
                else:
                    dev = self.env.route_deviation_rate()
                    deviation.append(np.mean(dev))
                    break

            # if episode > 0 and episode % 50 == 0:
            #     self.env.render(mode='traj')   
         
            rewards = rewards / 10000
            returns.append(rewards)
            print('Returns is', rewards)
            print("conflict num :", self.env.collision_num)
            print("nmac num：", self.env.nmac_num)
            print("exit boundary num：", self.env.exit_boundary_num)
            print("success num：", self.env.success_num)
            conflict_total.append(self.env.collision_num)
            nmac_total.append(self.env.nmac_num)
            collide_wall_total.append(self.env.exit_boundary_num)
            success_total.append(self.env.success_num)
        
        plot_cn(
            returns[1:], "average returns", "evaluate num",
            save_path=self.save_path + '/evaluate_res/{}_agent_{}/'.format(self.agent_num, time_num),
            save_name='{}_eval_return_{}'.format(self.agent_num, time_num)
            )

        ave_conflict = np.mean(conflict_total)
        ave_nmac = np.mean(nmac_total)
        ave_success = np.mean(success_total)
        ave_exit = np.mean(collide_wall_total)
        zero_conflict = sum(np.array(conflict_total) == 0)
        print("平均冲突数", ave_conflict)
        print("平均NMAC数", ave_nmac)
        print("平均成功率", ave_success / self.agent_num)
        print("平均出界率", ave_exit / self.agent_num)
        print("0冲突占比：", zero_conflict / len(conflict_total))
        print("平均偏差率", np.mean(deviation))
        
        plot_multi_cn(
            [conflict_total, collide_wall_total, success_total, nmac_total],
            # ["总冲突数量", "出界数量", "成功数量", "nmac数量"],
            ["confilt num", "exit_boundary num", "success num", "nmac total"],
            ['evaluate num']*4,
            ['']*4,
            save_path=self.save_path + '/evaluate_res/{}_agent_{}/'.format(self.agent_num, time_num),
            save_name='{}_eval_metrix_{}'.format(self.agent_num, time_num)
        )

        np.save(
            self.save_path + '/evaluate_res/{}_agent_{}/evaluate_metrics.npy'.format(self.agent_num, time_num),
            [ave_conflict, ave_nmac, ave_success/self.agent_num, ave_exit/self.agent_num,
             zero_conflict/len(conflict_total), np.mean(deviation)]
        )