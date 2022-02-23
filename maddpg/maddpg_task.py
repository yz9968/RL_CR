from common.plot import plot_multi_cn, plot_cn_off, plot_cn
from common.arguments import get_args
from common.memory import MultiReplayBuffer
from curses import noraw
import sys
import os
import datetime
import torch
import numpy as np
import math
import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
# logging.basicConfig(filename='my.log', level=logging.DEBUG, format=LOG_FORMAT)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class Task_maddpg:
    def __init__(self, args, env) -> None:
        self.env_name = 'maddpg_conflict_resolution'
        self.args = args

        self.env = env
        self.agents = self.env.agents
        self.noise = args.noise_rate
        self.noise_end = 0.05
        self.noise_start = args.noise_rate
        self.epsilon = args.epsilon
        self.epsilon_end = 0.05
        self.epsilon_start = args.epsilon
        self.epsilon_decay = args.epsilon_decay

        self.max_step = args.max_episode_len
        self.episodes_num = args.num_episodes
        self.agent_num = self.env.agent_num
        self.buffer = MultiReplayBuffer(args.buffer_size, self.args)
        self.batch_size = args.batch_size

        self.curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU

        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

    def train(self):
        logging.info("start training...")
        start_time = datetime.datetime.now()

        # Evaluate the effectiveness of the training
        rewards = []
        ma_rewards = []
        # Key index
        returns = []
        conflict_total = []
        collide_wall_total = []
        success_total = []
        nmac_total = []

        for episode in range(self.episodes_num):
            state = self.env.reset()
            ep_reward_agents = []
            # epsilon是会递减的，这里选择指数递减
            self.epsilon = lambda episode: self.epsilon_end + \
                (self.epsilon_start-self.epsilon_end) * math.exp(-1. * episode / self.epsilon_decay)

            logging.info("episode: {}".format(episode))
            for each_step in range(self.max_step):
                # self.env.render(mode='traj')
                # self.env.m_render()
                self.noise = lambda each_step: self.noise_end + \
                    (self.noise_start-self.noise_end)*math.exp(-1.*each_step)
                if not self.env.simulation_done:
                    actions = []
                    for i, agent in enumerate(self.agents):
                        state_i = torch.FloatTensor(state[i]).to(self.device)
                        action = agent.select_action(state_i, self.noise(each_step), self.epsilon(episode))
                        actions.append(action)

                    next_state, reward, done, info = self.env.step(actions)
                    # self.buffer:MultiReplayBuffer
                    self.buffer.push(state, actions, reward, next_state)
                    state = next_state

                    # update
                    if self.buffer.current_size >= self.batch_size:
                        transitions = self.buffer.sample(self.batch_size)
                        for agent in self.agents:
                            other_agents = self.agents.copy()
                            other_agents: list
                            other_agents.remove(agent)
                            agent.learn(transitions, other_agents)
                    ep_reward_agents.append(sum(reward)/1000)  # why deivded by 1000
                else:
                    logging.info("episode {} done! Takes {} steps.".format(episode, each_step))
                    break

            rewards.append(sum(ep_reward_agents))
            ma_rewards.append(sum(ep_reward_agents) if len(ma_rewards) ==
                              0 else 0.9*ma_rewards[-1]+0.1*sum(ep_reward_agents))

            if episode > 0 and episode % self.args.evaluate_rate == 0:
                rew, info = self.evaluate()
                # if episode % (5 * self.args.evaluate_rate) == 0:
                #     self.env.render(mode='traj')
                returns.append(rew)
                conflict_total.append(info[0])
                collide_wall_total.append(info[1])
                success_total.append(info[2])
                nmac_total.append(info[3])

        end_time = datetime.datetime.now()
        logging.info('完成{}个episode，共花费时间 {}'.format(self.episodes_num, end_time-start_time))

        # plot_rewards(rewards,ma_rewards,"train","Conflict resolution","MADDPG",path=self.save_path)
        plot_cn(returns, 'returns', 'evaluate num', 'average returns', self.save_path,
                save_name='{}_train_return.png'.format(self.agent_num))
        np.save(self.save_path + '/{}_train_returns'.format(self.agent_num), returns)

        plot_multi_cn(
            [conflict_total, collide_wall_total, success_total, nmac_total],
            ["总冲突数量", "出界数量", "成功数量", "nmac数量"],
            ['evaluate num']*4,
            ["confilt_num", "exit_boundary_num", "success_num", "nmac_total"],
            save_path=self.save_path,
            save_name='{}_train_metrix'.format(self.agent_num)
        )
        np.save(self.save_path + '/{}_train_returns.pkl'.format(self.args.n_agents), conflict_total)

    def evaluate(self):
        logging.info('evaluate')
        self.env.collision_num = 0
        self.env.exit_boundary_num = 0
        self.env.success_num = 0
        self.env.nmac_num = 0
        returns = []
        deviation = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            state = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                # self.env.render()
                if self.args.render:
                    self.env.m_render()
                if not self.env.simulation_done:
                    actions = []
                    for i, agent in enumerate(self.agents):
                        state_i = torch.FloatTensor(state[i]).to(self.device)
                        action = agent.select_action(state_i, 0, 0)
                        actions.append(action)

                    next_state, reward, done, info = self.env.step(actions)
                    # self.buffer:MultiReplayBuffer
                    # self.buffer.push(state,actions,reward,next_state)
                    rewards += sum(reward)
                    state = next_state

                    # with torch.no_grad():
                    #     for agent_id, agent in enumerate(self.agents):
                    #         action = agent.select_action(s[agent_id], 0, 0)
                    #         actions.append(action)
                    # s_next, r, done, info = self.env.step(actions)
                    # rewards += sum(r)
                    # s = s_next
                else:
                    dev = self.env.route_deviation_rate()
                    deviation.append(np.mean(dev))
                    break
            rewards = rewards / 10000
            returns.append(rewards)
            print('Returns is', rewards)
        print("conflict num :", self.env.collision_num)
        print("nmac num", self.env.nmac_num)
        print("exit boundary num：", self.env.exit_boundary_num)
        print("success num：", self.env.success_num)
        print("路径平均偏差率：", np.mean(deviation))

        return sum(returns) / self.args.evaluate_episodes, (self.env.collision_num, self.env.exit_boundary_num, self.env.success_num, self.env.nmac_num)

    def evaluate_model(self):
        import matplotlib.pyplot as plt
        import time
        """
        对现有最新模型进行评估
        :return:
        """
        print("now evaluate the model")
        conflict_total = []
        collide_wall_total = []
        success_total = []
        deviation = []
        nmac_total = []
        self.env.collision_num = 0
        self.env.exit_boundary_num = 0
        self.env.success_num = 0
        self.env.nmac_num = 0
        returns = []
        eval_episode = 100
        time_num = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))

        for episode in range(eval_episode):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                if episode % 10 == 0 and self.args.render:
                    self.env.m_render()
                if not self.env.simulation_done:
                    actions = []
                    with torch.no_grad():
                        for agent_id, agent in enumerate(self.agents):
                            state_i = torch.FloatTensor(s[agent_id]).to(self.device)
                            action = agent.select_action(state_i, 0, 0)
                            actions.append(action)
                    s_next, r, done, info = self.env.step(actions)
                    rewards += sum(r)
                    s = s_next
                else:
                    dev = self.env.route_deviation_rate()
                    deviation.append(np.mean(dev))
                    break

            rewards = rewards / 1000
            returns.append(rewards)

            plot_cn_off(
                self.env.collision_value, "collision value", "step", "collision_value",
                save_path=self.save_path + '/evaluate_res/{}_agent_{}/'.format(self.agent_num, time_num),
                save_name='{}_collision_value'.format(episode)
            )

            np.save(
                self.save_path + '/evaluate_res/{}_agent_{}/collision_value.npy'.format(self.agent_num, time_num),
                self.env.collision_value
            )
            logging.info('episode {} eval res:'.format(episode))
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
            self.env.exit_boundary_num = 0
            self.env.success_num = 0
            self.env.nmac_num = 0

        plot_cn(
            returns[1:], "average returns", "evaluate num",
            save_path=self.save_path+'/evaluate_res/{}_agent_{}/'.format(self.agent_num, time_num), 
            save_name='{}_eval_return_{}'.format(self.agent_num, time_num)
        )

        ave_conflict = np.mean(conflict_total)
        ave_nmac = np.mean(nmac_total)
        ave_success = np.mean(success_total)
        ave_exit = np.mean(collide_wall_total)
        zero_conflict = sum(np.array(conflict_total) == 0)
        logging.info('eval res:'.format(episode))
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
