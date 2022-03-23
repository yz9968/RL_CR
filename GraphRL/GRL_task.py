import os
import time
import numpy as np
import logging
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from GraphRL.GRL import ReplayBuffer,GRL

class Task_GRL:
    def __init__(self,args, env) -> None:
        self.args = args
        device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
        logging.info('Using device: %s', device)
        USE_CUDA = torch.cuda.is_available()
        self.env = env
        self.epsilon = args.epsilon
        self.num_episode = args.num_episodes
        self.max_step = args.max_episode_len
        self.agents = self.env.agents
        self.agent_num = self.env.agent_num
        self.n_action = args.action_shape[0]
        self.n_observation=args.obs_shape[0]
        self.hidden_dim = 64
        self.buffer = ReplayBuffer(args.buffer_size, self.n_observation, self.n_action, self.agent_num)
        self.lr = 1e-4
        self.batch_size = args.batch_size
        self.train_epoch = 25
        self.gamma = args.gamma
        self.observation_space = self.env.observation_space
        self.model = DGN(self.agent_num, self.observation_space, self.hidden_dim, self.n_action)
        self.model_tar = DGN(self.agent_num, self.observation_space, self.hidden_dim, self.n_action)
        self.model = self.model.cuda()
        self.model_tar = self.model_tar.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.model_name = '/{0}_agent/{0}_graph_rl_weight.pth'.format(self.agent_num)
        if os.path.exists(self.save_path + self.model_name):
            self.model.load_state_dict(torch.load(self.save_path + self.model_name))
            print("successfully load model: {}".format(self.model_name))

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
        rl_model_dir = self.save_path + self.model_name
        while episode < self.num_episode:
            if episode > start_episode:
                self.epsilon = max(0.05, self.epsilon - 0.00016)

            episode += 1
            step = 0
            obs, adj = self.env.reset()
            print("current episode {}".format(episode))
            while step < self.max_step:
                if not self.env.simulation_done:
                    # print(" {} episode {} step ".format(i_episode, steps))
                    step += 1
                    action = []
                    obs1 = np.expand_dims(obs, 0)  # shape （1, 6, 9(observation_space)）
                    adj1 = np.expand_dims(adj, 0)
                    q, a_w = self.model(torch.Tensor(obs1).cuda(), torch.Tensor(adj1).cuda())  # shape (100, 3)
                    q = q[0]
                    # 待改
                    for i, agent in enumerate(self.agents):
                        if np.random.rand() < self.epsilon:
                            a = np.random.randint(self.n_action)
                        else:
                            a = q[i].argmax().item()
                        action.append(a)

                    next_obs, next_adj, reward, done_signals, info = self.env.step(action)

                    self.buffer.add(obs, action, reward, next_obs, adj, next_adj, info['simulation_done'])
                    obs = next_obs
                    adj = next_adj

                else:
                    # print(" agent_terminated_times:", self.env.agent_times)
                    if self.env.simulation_done:
                        print("all agents done!")
                    break

            if episode > 0 and episode % self.args.evaluate_rate == 0:
                rew, info = self.evaluate()
                if episode % (5 * self.args.evaluate_rate) == 0:
                    self.env.render(mode='traj')
                reward_total.append(rew)
                conflict_total.append(info[0])
                collide_wall_total.append(info[1])
                success_total.append(info[2])
                nmac_total.append(info[3])
            self.env.conflict_num_episode = 0
            self.env.nmac_num_episode = 0

            if episode < start_episode:
                continue

            for epoch in range(self.train_epoch):
                Obs, Act, R, Next_Obs, Mat, Next_Mat, D = self.buffer.getBatch(self.batch_size)
                Obs = torch.Tensor(Obs).cuda()
                Mat = torch.Tensor(Mat).cuda()
                Next_Obs = torch.Tensor(Next_Obs).cuda()
                Next_Mat = torch.Tensor(Next_Mat).cuda()

                q_values, attention = self.model(Obs, Mat)  # shape (128, 6, 3)
                target_q_values, target_attention = self.model_tar(Next_Obs, Next_Mat)  # shape  (128, 6)
                target_q_values = target_q_values.max(dim=2)[0]
                target_q_values = np.array(target_q_values.cpu().data)  # shape  (128, 6)
                expected_q = np.array(q_values.cpu().data)  # (batch_size, agent_num, action_num)

                for j in range(self.batch_size):
                    for i in range(self.agent_num):
                        # sample[1]: action selection list ; sample[2]: reward size-agent_num ; sample[6]: terminated
                        expected_q[j][i][Act[j][i]] = R[j][i] + (1 - D[j]) * self.gamma * target_q_values[j][i]
                        # if sample[6][i] != 1:
                        #     expected_q[j][i][sample[1][i]] = sample[2][i] + self.gamma * target_q_values[j][i]
                        # else:
                        #     expected_q[j][i][sample[1][i]] = sample[2][i]

                attention = F.log_softmax(attention,dim=2)
                target_attention = F.softmax(target_attention,dim=2)
                target_attention = target_attention.detach()
                loss_kl = F.kl_div(attention, target_attention, reduction='mean')
                loss = (q_values - torch.Tensor(expected_q).cuda()).pow(2).mean() + lamb * loss_kl
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    for p, p_targ in zip(self.model.parameters(), self.model_tar.parameters()):
                        p_targ.data.mul_(tau)
                        p_targ.data.add_((1 - tau) * p.data)

            # if episode % 5 == 0:
            #     self.model_tar.load_state_dict(self.model.state_dict())

            if episode != 0 and episode % 200 == 0:
                torch.save(self.model.state_dict(), rl_model_dir)
                print("torch save model for rl_weight")

        end = time.time()
        print("花费时间:", end - start)

        fig, a = plt.subplots(2, 2)
        plt.title('GRL_train')
        x = range(len(conflict_total))
        a[0][0].plot(x, conflict_total, 'b')
        a[0][0].set_title('conflict_num')
        a[0][1].plot(x, collide_wall_total, 'y')
        a[0][1].set_title('exit_boundary_num')
        a[1][0].plot(x, success_total, 'r')
        a[1][0].set_title('success_num')
        a[1][1].plot(x, nmac_total)
        a[1][1].set_title('nmac_num')
        plt.savefig(self.save_path + '/{}_agent/train_metric.png'.format(self.agent_num), format='png')
        plt.show()

    def evaluate(self):
        print("now is evaluate!")
        self.env.collision_num = 0
        self.env.exit_boundary_num = 0
        self.env.success_num = 0
        self.env.nmac_num = 0
        returns = []
        deviation = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            obs, adj = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                if not self.env.simulation_done:
                    actions = []
                    obs1 = np.expand_dims(obs, 0)  # shape （1, 6, 9(observation_space)）
                    adj1 = np.expand_dims(adj, 0)
                    q, a_w = self.model(torch.Tensor(obs1).cuda(), torch.Tensor(adj1).cuda())  # shape (100, 5)
                    q = q[0]
                    for i, agent in enumerate(self.agents):
                        a = q[i].argmax().item()
                        actions.append(a)

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
        print("now evaluate the model")
        conflict_total = []
        collide_wall_total = []
        success_total = []
        nmac_total = []
        deviation = []
        self.env.collision_num = 0
        self.env.nmac_num = 0
        self.env.exit_boundary_num = 0
        self.env.success_num = 0
        returns = []
        eval_episode = 180
        for episode in range(eval_episode):
            # reset the environment
            obs, adj = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                if not self.env.simulation_done:
                    actions = []
                    obs1 = np.expand_dims(obs, 0)  # shape （1, 6, 9(observation_space)）
                    adj1 = np.expand_dims(adj, 0)
                    q, a_w = self.model(torch.Tensor(obs1).cuda(), torch.Tensor(adj1).cuda())  # shape (100, 5)
                    q = q[0]
                    for i, agent in enumerate(self.agents):
                        a = q[i].argmax().item()
                        actions.append(a)
                        # print("agent {} action {}".format(i, a))

                    next_obs, next_adj, reward, done_signals, info = self.env.step(actions)
                    rewards += sum(reward)
                    obs = next_obs
                    adj = next_adj
                else:
                    dev = self.env.route_deviation_rate()
                    deviation.append(np.mean(dev))
                    break
            # np.save(self.save_path + '/20_agent/actions/' + str(episode) + 'actions.npy',
            #         np.array(self.env.actions_total))

            if episode > 0 and episode % 50 == 0:
                self.env.render(mode='traj')
            # if episode > 0:
            #     self.env.render(mode='traj')

            # plt.figure()
            # plt.title('collision_value——time')
            # x = range(len(self.env.collision_value))
            # plt.plot(x, self.env.collision_value)
            # plt.xlabel('timestep')
            # plt.ylabel('collision_value')
            # plt.savefig(self.save_path + '/30_agent/collision_value/' + str(episode) + 'collision_value.png', format='png')
            # np.save(self.save_path + '/30_agent/collision_value/' + str(episode) + 'collision_value.npy', self.env.collision_value)
            # plt.close()

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
            self.env.collision_num = 0
            self.env.nmac_num = 0
            self.env.exit_boundary_num = 0
            self.env.success_num = 0

        plt.figure()
        plt.plot(range(1, len(returns)), returns[1:])
        plt.xlabel('evaluate num')
        plt.ylabel('average returns')
        # plt.savefig(self.save_path + '/50_agent/eval_return_2.png', format='png')

        # conflict num process
        conflict_total_1 = []
        nmac_total_1 = []
        for i in range(len(conflict_total)):
            if success_total[i] + collide_wall_total[i] == self.agent_num:
                conflict_total_1.append(conflict_total[i])
                nmac_total_1.append(nmac_total[i])

        y = range(len(conflict_total))
        conflict_total = conflict_total_1
        nmac_total = nmac_total_1
        x = range(len(conflict_total))
        print("有效轮数：", len(x))
        fig, a = plt.subplots(2, 2)
        # 去除冲突数极大值
        conflict_total[conflict_total.index(max(conflict_total))] = 0
        conflict_total[conflict_total.index(max(conflict_total))] = 0
        ave_conflict = np.mean(conflict_total)
        ave_nmac = np.mean(nmac_total)
        ave_success = np.mean(success_total)
        ave_exit = np.mean(collide_wall_total)
        zero_conflict = sum(np.array(conflict_total) == 0) - 2
        print("平均冲突数", ave_conflict)
        print("平均NMAC数", ave_nmac)
        print("平均成功率", ave_success / self.agent_num)
        print("平均出界率", ave_exit / self.agent_num)
        print("0冲突占比：", zero_conflict / len(conflict_total))
        print("平均偏差率", np.mean(deviation))
        a[0][0].plot(x, conflict_total, 'b')
        a[0][0].set_title('conflict_num')
        a[0][1].plot(y, collide_wall_total, 'y')
        a[0][1].set_title('exit_boundary_num')
        a[1][0].plot(y, success_total, 'r')
        a[1][0].set_title('success_num')
        a[1][1].plot(x, nmac_total)
        a[1][1].set_title('nmac_num')
        # plt.savefig(self.save_path + '/50_agent/eval_metric2.png', format='png')

        plt.show()
