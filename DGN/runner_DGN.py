import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
import logging
import time
from common.plot import plot_multi_cn, plot_cn_off,plot_cn,plot_multi_cn_off
from DGN.model import DGN
from DGN.buffer import ReplayBuffer

class Runner_DGN:
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
        self.buffer = ReplayBuffer(args.buffer_size, self.n_obs, self.n_action, self.agent_num)
        self.lr = 1e-4
        self.batch_size = args.batch_size
        self.train_epoch = 25
        self.gamma = args.gamma
        self.observation_space = self.env.observation_space
        self.model = DGN(self.agent_num, self.observation_space, self.hidden_dim, self.n_action).to(self.device)
        self.model_tar = DGN(self.agent_num, self.observation_space, self.hidden_dim, self.n_action).to(self.device)
        self.model = self.model.to(self.device)
        self.model_tar = self.model_tar.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name  +'{1}/{0}_agent/'.format(self.agent_num,args.attach_info)
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
        self.model_name = '/{}_graph_rl_weight.pth'.format(self.agent_num)
        if os.path.exists(self.save_path + self.model_name):
            self.model.load_state_dict(torch.load(self.save_path + self.model_name))
            logging.info("successfully load model: {}".format(self.model_name))

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
            logging.info("current episode {}".format(episode))
            while step < self.max_step:
                if not self.env.simulation_done:
                    if self.args.train_render == True:
                        self.env.m_render(self.args.render_mode)
                    step += 1
                    action = []
                    obs1 = np.expand_dims(obs, 0)  # shape （1, 6, 9(observation_space)）
                    adj1 = np.expand_dims(adj, 0)
                    q, a_w = self.model(torch.Tensor(obs1).to(self.device), torch.Tensor(adj1).to(self.device))  # shape (100, 3)
                    q = q[0]
                    # 待改
                    for i, agent in enumerate(self.agents):
                        if np.random.rand() < self.epsilon:
                            a = np.random.randint(self.n_action)
                        else:
                            a = q[i].argmax().item()
                        action.append(a)

                    next_obs, next_adj, reward, done_signals, info = self.env.step(action)
                    # TODO buffer: done shape; update: done signal
                    done=[0 if (done_i == 0 or done_i == 3) else 1 for done_i in done_signals]

                    # self.buffer.add(obs, action, reward, next_obs, adj, next_adj, info['simulation_done'])
                    self.buffer.add(obs, action, reward, next_obs, adj, next_adj, done)
                    obs = next_obs
                    adj = next_adj

                else:
                    logging.info("all agents done!")
                    break

            if episode > 0 and episode % self.args.evaluate_rate == 0:
                rew, info = self.evaluate()
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
                Obs = torch.Tensor(Obs).to(self.device)
                Mat = torch.Tensor(Mat).to(self.device)
                Next_Obs = torch.Tensor(Next_Obs).to(self.device)
                Next_Mat = torch.Tensor(Next_Mat).to(self.device)

                q_values, attention = self.model(Obs, Mat)  # shape (128, 6, 3)
                target_q_values, target_attention = self.model_tar(Next_Obs, Next_Mat)  # shape  (128, 6)
                target_q_values = target_q_values.max(dim=2)[0]
                target_q_values = np.array(target_q_values.cpu().data)  # shape  (128, 6)
                expected_q = np.array(q_values.cpu().data)  # (batch_size, agent_num, action_num)

                for j in range(self.batch_size):
                    for i in range(self.agent_num):
                        # expected_q[j][i][Act[j][i]] = R[j][i] + (1 - D[j]) * self.gamma * target_q_values[j][i]
                        expected_q[j][i][Act[j][i]] = R[j][i] + (1 - D[j][i]) * self.gamma * target_q_values[j][i]

                attention = F.log_softmax(attention,dim=2)
                target_attention = F.softmax(target_attention,dim=2)
                target_attention = target_attention.detach()
                loss_kl = F.kl_div(attention, target_attention, reduction='batchmean')
                loss = (q_values - torch.Tensor(expected_q).to(self.device)).pow(2).mean() + lamb * loss_kl
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    for p, p_targ in zip(self.model.parameters(), self.model_tar.parameters()):
                        p_targ.data.mul_(tau)
                        p_targ.data.add_((1 - tau) * p.data)

            if episode != 0 and episode % 200 == 0:
                torch.save(self.model.state_dict(), rl_model_dir)
                logging.info("torch save model for rl_weight")

        end = time.time()
        logging.info("花费时间:{}".format(end - start))

        plot_cn_off(reward_total, 'rewards', 'evaluate num', 'average rewards', save_path=self.save_path,
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
        logging.info("now is evaluate!")
        returns = []
        deviation = []
        for episode in range(self.args.evaluate_episodes):
            # logging.info("evaluate episode {}".format(episode))
            # reset the environment
            obs, adj = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                if not self.env.simulation_done:
                    if self.args.render==True:
                        self.env.m_render(self.args.render_mode)
                    actions = []
                    obs1 = np.expand_dims(obs, 0)  # shape （1, 6, 9(observation_space)）
                    adj1 = np.expand_dims(adj, 0)
                    q, a_w = self.model(torch.Tensor(obs1).to(self.device), torch.Tensor(adj1).to(self.device))  # shape (100, 5)
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
            logging.info("=====eval res (training)=====:")
            logging.info('Returns is {}'.format(rewards))
            logging.info("conflict num :".format(self.env.collision_num))
            logging.info("nmac num :".format(self.env.nmac_num))
            logging.info("exit boundary num：".format(self.env.exit_boundary_num))
            logging.info("success num：".format(self.env.success_num))
            logging.info("路径平均偏差率：".format(np.mean(deviation)))

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
                    if self.args.render==True and episode % 20 ==0:
                        self.env.m_render(self.args.render_mode)
                    actions = []
                    obs1 = np.expand_dims(obs, 0)  # shape （1, 6, 9(observation_space)）
                    adj1 = np.expand_dims(adj, 0)
                    q, a_w = self.model(torch.Tensor(obs1).to(self.device), torch.Tensor(adj1).to(self.device))  # shape (100, 5)
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