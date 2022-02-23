import os
import numpy as np
import logging
import time
import matplotlib.pyplot as plt
from collections import namedtuple

from common.plot import plot_multi_cn, plot_cn_off,plot_cn

Transition = namedtuple("Transition", ['state', 'action', 'a_log_prop', 'reward', 'next_state'])
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class Task_PPO:
    def __init__(self, args, env) -> None:
        self.args = args
        self.env = env

        self.epsilon = args.epsilon
        self.num_episodes = self.args.num_episodes
        self.max_step = args.max_episode_len
        self.agent_num = self.env.agent_num
        self.agents = self.env.agents

        self.save_path = self.args.save_dir+'/'+self.args.scenario_name
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

    def train(self):
        returns = []
        reward_total = []
        conflict_total = []
        collide_wall_total = []
        nmac_total = []
        success_total = []
        start = time.time()

        for episode in range(self.num_episodes):
            reward_episode = []
            state = self.env.reset()
            self.epsilon = max(0.05, self.epsilon-0.00016)
            logging.info("episode:{}".format(episode))

            for step in range(self.max_step):
                if not self.env.simulation_done:
                    actions = []
                    action_probs = []
                    for i, agent in enumerate(self.agents):
                        action, action_prob = agent.policy.select_action(state[i])  # TODO
                        actions.append(action)
                        action_probs.append(action_prob)
                    next_state, reward, done, indo = self.env.step(actions)
                    for i, agent in enumerate(self.agents):
                        transition = Transition(state[i], actions[i], action_probs[i], reward[i], next_state[i])
                        agent.policy.store_transition(transition)  # TODO
                    state = next_state
                    reward_episode.append(sum(reward)/1000)

                else:
                    logging.info("episode {} done, takes {} steps, all agent done.".format(episode, step))
                    for i, agent in enumerate(self.agents):
                        if len(agent.policy.buffer) > self.args.batch_size:
                            agent.policy.learn()  # TODO
                    break
            reward_total.append(sum(reward_episode))

            if episode > 0 and episode % self.args.evaluate_rate == 0:
                rew, info = self.evaluate()
                returns.append(rew)
                conflict_total.append(info[0])
                collide_wall_total.append(info[1])
                success_total.append(info[2])
                nmac_total.append(info[3])
        end = time.time()
        logging.info("time cost: {}".format(end-start))
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
        # plot_cn(conflict_total, "总冲突数量", label="confilt_num", save_name="confilct_curve_ppo", save_path="./ppo/")
        # plot_cn(collide_wall_total, "出界数量", label="exit_boundary_num",save_name="exit_boundary_num_ppo", save_path="./ppo/")
        # plot_cn(success_total, "成功数量", label="success_num", save_name="success_num_ppo", save_path="./ppo/")
        # plot_cn(nmac_total, "nmac数量", label="nmac_total", save_name="nmac_total_ppo", save_path="./ppo/")

    def evaluate(self):
        logging.info("evaluate")
        self.env.collision_num = 0
        self.env.exit_boundary_num = 0
        self.env.success_num = 0
        self.env.nmac_num = 0
        returns = []
        deviation = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                if self.args.render:
                    self.env.m_render()
                if not self.env.simulation_done:
                    actions = []
                    for agent_id, agent in enumerate(self.agents):
                        action, action_prob = agent.policy.select_action(s[agent_id])
                        actions.append(action)
                    s_next, r, done, info = self.env.step(actions)
                    rewards += sum(r)
                    s = s_next
                else:
                    dev = self.env.route_deviation_rate()
                    deviation.append(np.mean(dev))
                    break
            rewards = rewards / 10000
            returns.append(rewards)
            print('Returns is', rewards)
        print("\rconflict num :", self.env.collision_num)
        print("nmac num :", self.env.nmac_num)
        print("exit boundary num：", self.env.exit_boundary_num)
        print("success num：", self.env.success_num)
        print("路径平均偏差率：", np.mean(deviation), '\r')
        return sum(returns) / self.args.evaluate_episodes, (self.env.collision_num, self.env.exit_boundary_num, self.env.success_num, self.env.nmac_num)

    def evaluate_model(self):
        conflict_total = []
        collide_wall_total = []
        success_total = []
        deviation = []
        nmac_total = []
        returns = []
        eval_episode = 100
        time_num = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))

        for episode in range(eval_episode):
            state = self.env.reset()
            rewards = 0
            for step in range(self.args.evaluate_episode_len):
                if not self.env.simulation_done:
                    if episode % 50 == 0 and self.args.render: 
                        self.env.m_render()
                    actions = []
                    for i, agent in enumerate(self.agents):
                        action, prop = agent.policy.select_action(state[i])
                        actions.append(action)
                    next_state, reward, done, info = self.env.step(actions)
                    rewards += sum(reward)
                    state = next_state
                else:
                    dev = self.env.route_deviation_rate()
                    deviation.append(np.mean(dev))
                    break

            plot_cn_off(
                self.env.collision_value, "collision value", "step", "collision_value",
                save_path=self.save_path + '/evaluate_res/{}_agent_{}/'.format(self.agent_num, time_num),
                save_name='{}_collision_value'.format(episode)
            )

            np.save(
                self.save_path + '/evaluate_res/{}_agent_{}/collision_value.npy'.format(self.agent_num, time_num),
                self.env.collision_value
            )

            rewards = rewards / 1000
            returns.append(rewards)
            logging.info("episode {} eval res:".format(episode))
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

        logging.info("eval res:")
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