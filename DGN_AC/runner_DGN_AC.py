import os
import time
import numpy as np
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from DGN_AC.model import Actor,Critic,AttEncoder
from DGN_AC.buffer import ReplayBuffer
from common.plot import plot_multi_cn, plot_cn_off,plot_cn,plot_multi_cn_off

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
        self.hidden_dim = 64
        self.lr_actor = 1e-4
        self.lr_critic= 1e-3
        self.soft_tau = 1e-2 # 软更新参数
        self.batch_size = args.batch_size
        self.buffer = ReplayBuffer(self.batch_size)
        self.train_epoch = 25
        self.gamma = args.gamma
        self.gae_lambda=0.95
        self.policy_clip=0.2
        self.observation_space = self.env.observation_space

        self.actor=Actor(self.hidden_dim,self.n_action,self.hidden_dim).to(self.device)
        self.critic=Critic(self.hidden_dim + 1,self.hidden_dim).to(self.device)
        self.att_encoder=AttEncoder(self.agent_num,self.n_obs,self.hidden_dim,self.n_action).to(self.device)
        
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name +'{1}/{0}_agent/'.format(self.agent_num,args.attach_info)
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

    def choose_action(self, state):
        # state = torch.tensor([state], dtype=torch.float).to(self.device)
        dist = self.actor(state)
        actions=dist.sample()
        values=self.critic(state,actions.unsqueeze(dim=2).detach())
        values=list(torch.squeeze(values).cpu().detach().numpy())
        probs=dist.log_prob(actions)
        actions=list(torch.squeeze(actions).cpu().numpy())
        probs=list(torch.squeeze(probs).cpu().detach().numpy())
        return actions, probs, values

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
            # if episode > start_episode:
            #     self.epsilon = max(0.05, self.epsilon - 0.00016)
            episode += 1
            step = 0
            rewards=0
            obs, adj = self.env.reset()
            logging.info("current episode {}".format(episode))
            while step < self.max_step:
                if not self.env.simulation_done:
                    if self.args.train_render == True and episode % self.args.train_render_rate == 0:
                        self.env.m_render(self.args.render_mode)
                    step += 1

                    obs_tensor=torch.FloatTensor(obs).view((1,self.agent_num,self.n_obs)).to(self.device) # torch.Size([1, 6, 9])
                    adj_tensor=torch.FloatTensor(adj).view((1,self.agent_num,self.agent_num)).to(self.device) # torch.Size([1, 6, 6])

                    obs_hybrid, obs_attention=self.att_encoder(obs_tensor,adj_tensor)
                    actions, probs, values=self.choose_action(obs_hybrid)
                    next_obs, next_adj, reward, done_signals, info = self.env.step(actions)
                    
                    rewards += sum(reward)
                    state=obs_hybrid.squeeze(dim=0).detach().cpu().numpy()
                    done=[0 if (done_i == 0 or done_i == 3) else 1 for done_i in done_signals]
                    # state, action, prob, val, reward, done
                    self.buffer.push(state, actions, probs, values, reward, done)
                    obs = next_obs
                    adj = next_adj
                else:
                    logging.info("all agents done, reward {:.2f}, update now!".format(rewards))
                    self.update()
                    break

            if episode > 0 and episode % self.args.evaluate_rate == 0:
                rew, info = self.evaluate()
                reward_total.append(rew)
                conflict_total.append(info[0])
                collide_wall_total.append(info[1])
                success_total.append(info[2])
                nmac_total.append(info[3])

            if episode != 0 and episode % self.args.save_rate == 0:
                torch.save(self.actor.state_dict(), rl_actor_model_dir)
                torch.save(self.critic.state_dict(), rl_critic_model_dir)
                logging.info("torch save model for rl_weight")

        end = time.time()
        logging.info("time cost: {}".format(end-start))

        plot_cn_off(reward_total, 'rewards', 'evaluate num', 'average rewards',save_path=self.save_path,
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
    

    def update(self):
        # state, actions, probs, values, reward, done
        state_batch, actions_batch, probs_batch, values_batch, reward_batch, done_batch, batches= self.buffer.sample()
        values_batch = torch.Tensor(values_batch).to(self.device)
        reward_batch = torch.Tensor(reward_batch).to(self.device)
        done_batch = torch.Tensor(done_batch).to(self.device)

        # 计算所有的 advantage
        advantage = torch.zeros((len(reward_batch),self.agent_num), dtype=torch.float32).to(self.device)
        for t in range(len(reward_batch)):
            discount = 1
            a_t = torch.zeros((1, self.agent_num)).to(self.device)
            for k in range(t, len(reward_batch) - 1):
                a_t += discount*(reward_batch[k] + self.gamma*values_batch[k+1]*(1-done_batch[k]) - values_batch[k])
                discount *= self.gamma*self.gae_lambda
            advantage[t] = a_t
        advantage = advantage.detach()
        values_batch = values_batch.detach()

        for batch in batches:
            states = torch.Tensor(state_batch[batch]).to(self.device)
            actions = torch.Tensor(actions_batch[batch]).to(self.device)
            old_probs = torch.Tensor(probs_batch[batch]).to(self.device)
            ### SGD ###
            dist=self.actor(states)
            entropy = dist.entropy().mean()
            new_probs=dist.log_prob(actions)
            prob_ratio=new_probs.exp()/old_probs.exp()
            weighted_probs=advantage[batch] * prob_ratio
            weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]
            actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
            returns = advantage[batch] + values_batch[batch]
            critic_value = self.critic(states,actions.unsqueeze(dim=2))
            # unsqueeze return? or squeeze critic_value?
            critic_loss = (returns.unsqueeze(dim=1)-critic_value).pow(2).mean()

            total_loss = actor_loss + 0.5*critic_loss - 0.1 * entropy
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            total_loss.backward()
            self.actor_optim.step()
            self.critic_optim.step()       
        self.buffer.clear() 

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
                    obs_hybrid, obs_attention=self.att_encoder(obs_tensor,adj_tensor)
                    actions,probs,values=self.choose_action(obs_hybrid)
                    next_obs, next_adj, reward, done_signals, info = self.env.step(actions)
                    obs = next_obs
                    adj = next_adj

                else:
                    dev = self.env.route_deviation_rate()
                    deviation.append(np.mean(dev))
                    break

            rewards = rewards / 10000
            returns.append(rewards)
            logging.info("=====eval res (training)=====:")
            logging.info('Returns is {}'.format(rewards))
            logging.info("conflict num : {}".format(self.env.collision_num))
            logging.info("nmac num : {}".format(self.env.nmac_num))
            logging.info("exit boundary num: {}".format(self.env.exit_boundary_num))
            logging.info("success num: {}".format(self.env.success_num))
        logging.info("路径平均偏差率：{}".format(np.mean(deviation)))

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
            logging.info("=====eval model, episode {}=====".format(episode))
            # reset the environment
            obs, adj = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                if not self.env.simulation_done:
                    if self.args.render==True and episode % self.args.eval_model_render_rate ==0:
                        self.env.m_render(self.args.render_mode)
                    obs_tensor=torch.FloatTensor(obs).view((1,self.agent_num,self.n_obs)).to(self.device) # torch.Size([1, 6, 9])
                    adj_tensor=torch.FloatTensor(adj).view((1,self.agent_num,self.agent_num)).to(self.device) # torch.Size([1, 6, 6])

                    obs_hybrid, obs_attention=self.att_encoder(obs_tensor,adj_tensor)
                    actions,probs,values=self.choose_action(obs_hybrid)
                    next_obs, next_adj, reward, done_signals, info = self.env.step(actions)
                    # if(sum(reward)<0):
                    #     print(actions)
                    
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
            # logging.info("eval res (evaluating the model):")
            logging.info('Returns is {}'.format(rewards))
            logging.info("conflict num :{}".format(self.env.collision_num))
            logging.info("nmac num: {}".format(self.env.nmac_num))
            logging.info("exit boundary num: {}".format(self.env.exit_boundary_num))
            logging.info("success num: {}".format(self.env.success_num))
            conflict_total.append(self.env.collision_num)
            nmac_total.append(self.env.nmac_num)
            collide_wall_total.append(self.env.exit_boundary_num)
            success_total.append(self.env.success_num)
        
        plot_cn_off(
            returns[1:], "average returns", "evaluate num",
            save_path=self.save_path + '/evaluate_res/{}_agent_{}/'.format(self.agent_num, time_num),
            save_name='{}_eval_return_{}'.format(self.agent_num, time_num)
            )
        ave_conflict = np.mean(conflict_total)
        ave_nmac = np.mean(nmac_total)
        ave_success = np.mean(success_total)
        ave_exit = np.mean(collide_wall_total)
        zero_conflict = sum(np.array(conflict_total) == 0)
        logging.info("===总轮数：{}===".format(len(conflict_total)))
        logging.info("平均冲突数：{}".format(ave_conflict))
        logging.info("平均NMAC数: {}".format(ave_nmac))
        logging.info("平均成功率：{}".format(ave_success / self.agent_num))
        logging.info("平均出界率：{}".format(ave_exit / self.agent_num))
        logging.info("0冲突占比: {}".format(zero_conflict / len(conflict_total)))
        logging.info("平均偏差率：{}".format(np.mean(deviation)))
        np.save(
            self.save_path + '/evaluate_res/{}_agent_{}/evaluate_metrics_origin.npy'.format(self.agent_num, time_num),
            [ave_conflict, ave_nmac, ave_success/self.agent_num, ave_exit/self.agent_num,
             zero_conflict/len(conflict_total), np.mean(deviation)]
        )

        # conflict num process
        conflict_total_1 = []
        nmac_total_1 = []
        for i in range(len(conflict_total)):
            if success_total[i] + collide_wall_total[i] == self.agent_num:
                conflict_total_1.append(conflict_total[i])
                nmac_total_1.append(nmac_total[i])

        conflict_total = conflict_total_1
        nmac_total = nmac_total_1
        logging.info("===有效轮数：{}===".format(len(conflict_total)))

        # 去除冲突数极大值
        conflict_total[conflict_total.index(max(conflict_total))] = 0
        conflict_total[conflict_total.index(max(conflict_total))] = 0
        ave_conflict = np.mean(conflict_total)
        ave_nmac = np.mean(nmac_total)
        ave_success = np.mean(success_total)
        ave_exit = np.mean(collide_wall_total)
        zero_conflict = sum(np.array(conflict_total) == 0) - 2
        logging.info("平均冲突数：{}".format(ave_conflict))
        logging.info("平均NMAC数: {}".format(ave_nmac))
        logging.info("平均成功率：{}".format(ave_success / self.agent_num))
        logging.info("平均出界率：{}".format(ave_exit / self.agent_num))
        logging.info("0冲突占比: {}".format(zero_conflict / len(conflict_total)))
        logging.info("平均偏差率：{}".format(np.mean(deviation)))

        plot_multi_cn_off(
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