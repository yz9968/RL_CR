from ast import arg
import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
import cv2
from math import pi, sin, cos, tan, sqrt, inf
from multiagent_particle_envs.multiagent.multi_discrete import MultiDiscrete
from multiagent_particle_envs.multiagent.core import Collision_Detection, Collision_Network
from maddpg.MADDPG import MADDPG
from ppo.PPO import PPO
# from ppo.ppo_cnn import PPO_CNN
# from ppo.ppo_lstm import PPO_LSTM

from gym.envs.classic_control import rendering


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def step(self, action_n):
        # action_n [numpy(action_shape) ]
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            print("in")
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            action = [action]
        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    agent.action.u = action[0]
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent:
                        continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from multiagent_particle_envs.multiagent import rendering
                self.viewers[i] = rendering.Viewer(700,700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from multiagent_particle_envs.multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent_particle_envs.multiagent import rendering
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))
        return dx


# environment for all agents in the multiagent world of maddpg
class MultiAgentEnv_maddpg(MultiAgentEnv):
    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, args = None, shared_viewer=True):
        super(MultiAgentEnv, self).__init__()
        self.world = world
        self.agents = self.world.agents
        self.agent_num = len(self.agents)
        self.args = args
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment setting
        self.simulation_done = None
        # state list
        self.states = None
        # action list
        self.action_list = [10, 5, 0, -5, -10] # 角度
        self.agent_times = None
        # goal setting
        self.goal_size = 2.0 # 海里
        # nmac
        self.nmac_size = 0.082
        # evaluate metric
        self.success_num = 0
        self.collision_num = 0
        self.nmac_num = 0
        self.exit_boundary_num = 0
        self.conflict_num_episode = None
        self.nmac_num_episode = None
        self.collision_value = None

        if args.render:
            self.viewer= rendering.Viewer(700,700)   # 画板的长和宽
            # self.viewer.close()

    def route_deviation_rate(self):
        deviation_rates = []
        for i, agent in enumerate(self.agents):
            d0 = agent.get_dist_togoal()
            d1 = agent.route_len
            deviation_rate = (d1 - d0) / d0
            deviation_rates.append(deviation_rate)
        return deviation_rates

    def reset(self):
        # reset world
        self.conflict_num_episode=0
        self.nmac_num_episode=0
        self.reset_callback(self.world)
        for i, agent in enumerate(self.agents):
            policy = MADDPG(self.args, agent.agent_id)
            agent.set_policy(policy)
        self.agents = self.world.agents
        self.agent_num = len(self.agents)
        self.simulation_done = False
        self.conflict_num_episode = 0
        self.nmac_num_episode = 0
        # record observations for each agent
        obs = [agent.get_full_state() for agent in self.agents]
        self.agent_times = [0 for _ in range(self.agent_num)]
        self.collision_value = []
        self.states = []
        self.states.append(obs)
        obs = self.get_obs(obs)

        self.collision_num = 0
        self.nmac_num = 0
        self.exit_boundary_num = 0
        self.success_num = 0

        return obs

    def step(self, actions):
        """
        缺一个判断何时检测到潜在冲突，以及何时回归intention_v
        :param actions: [0, 1, 2, 1, 2]
        :return:
        """
        done_signals = []
        for i in range(self.agent_num):
            done_signals.append(0)

        # update agents pos
        for i, agent in enumerate(self.agents):
            # no potential collision
            if agent.done == 0:
                v_intent = agent.get_intention_v()
                agent.step((v_intent[0], v_intent[1]))
                self.agent_times[i] += 1
            elif agent.done == 3:
                delta_theta = self.action_list[actions[i]]
                a = agent.theta * (180 / pi)
                a += delta_theta
                a %= 360
                agent.theta = a * (pi / 180)
                vx = agent.v_pref * cos(agent.theta)
                vy = agent.v_pref * sin(agent.theta)
                agent.step((vx, vy))
                self.agent_times[i] += 1
            else:
                continue

        # update agents status
        Exit_Boundary = []
        for i, agent in enumerate(self.agents):
            px = agent.px
            py = agent.py
            if agent.done == 1 or agent.done == 2:
                Exit_Boundary.append(False)
                continue
            elif px < self.world.boundary[0] or px > self.world.boundary[1] or py < self.world.boundary[2] or py > self.world.boundary[3]:
                Exit_Boundary.append(True)
                self.exit_boundary_num += 1
                agent.done = 2
                done_signals[i] = 2
            else:
                Exit_Boundary.append(False)

        dist_to_goal = np.array([agent.get_dist_to_goal() for agent in self.agents])
        Reach_Goal = []
        for i, agent in enumerate(self.agents):
            if agent.done == 1 or agent.done == 2:
                Reach_Goal.append(False)
                continue
            elif dist_to_goal[i] <= self.goal_size:
                Reach_Goal.append(True)
                agent.px, agent.py = agent.gx + 0.0001, agent.gy + 0.0001
                self.success_num += 1
                agent.done = 1
                done_signals[i] = 1
            else:
                Reach_Goal.append(False)

        next_obs = [agent.get_full_state() for agent in self.agents]
        self.states.append(next_obs)
        # generate collision network
        cnet = Collision_Network(next_obs)
        collision_mat = cnet.collision_matrix()
        dist_mat = cnet.dist_matrix()
        row, col = np.diag_indices_from(collision_mat)
        collision_mat[row, col] = 0
        dist_mat[row, col] = inf
        reach_goals_idx = np.where(Reach_Goal)[0]
        collision_mat[reach_goals_idx, :] = 0
        collision_mat[:, reach_goals_idx] = 0
        dist_mat[reach_goals_idx, :] = inf
        dist_mat[:, reach_goals_idx] = inf
        exit_idx = np.where(Exit_Boundary)[0]
        collision_mat[exit_idx, :] = 0
        collision_mat[:, exit_idx] = 0
        dist_mat[exit_idx, :] = inf
        dist_mat[:, exit_idx] = inf

        # update agent done --- 0 or 3
        for i, agent in enumerate(self.agents):
            if agent.done != 1 and agent.done != 2:
                collision_value_row = collision_mat[i]
                d = dist_mat[i]
                self.collision_num += sum(collision_value_row == 1) / 2
                self.conflict_num_episode += sum(collision_value_row == 1) / 2
                self.nmac_num += sum(d <= self.nmac_size) / 2
                self.nmac_num_episode += sum(d <= self.nmac_size) / 2
                # # no collision resolution
                # agent.done = 0
                if np.sum(collision_value_row) <= 0.2:
                    agent.done = 0
                else:
                    agent.done = 3


        # compute reward
        reward, c_v = self.reward_callback(self.world, collision_mat, Reach_Goal, Exit_Boundary, dist_to_goal)
        self.collision_value.append(c_v)

        terminal = True
        # 判断是否到达终态
        for i, agent in enumerate(self.agents):
            if agent.done == 0 or agent.done == 3:
                terminal = False
                break

        self.simulation_done = terminal
        info = {'current_time_reach_goals': Reach_Goal, 'current_time_exit_boundary': Exit_Boundary, 'simulation_done': terminal}

        return self.get_obs(next_obs), reward, done_signals, info

    def get_obs(self, obs):
        """
        input obs list of FullState transform to 2-dimension numpy
        :param obs:
        :return: obs matrix with shape (agent_num, action_num)
        """
        agent_num = len(obs)
        obs_mat = np.zeros((agent_num, 9))
        for i in range(len(obs)):
            ob = obs[i]
            obs_mat[i][0] = ob.px
            obs_mat[i][1] = ob.py
            obs_mat[i][2] = ob.vx
            obs_mat[i][3] = ob.vy
            obs_mat[i][4] = ob.radius
            obs_mat[i][5] = ob.gx
            obs_mat[i][6] = ob.gy
            obs_mat[i][7] = ob.v_pref
            obs_mat[i][8] = ob.theta

        return obs_mat

    def render(self, mode='human'):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        from matplotlib import patches
        import matplotlib.lines as mlines
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        x_offset = 0.11
        y_offset = 0.11
        cmap = plt.cm.get_cmap('hsv', 10)
        agent_color = 'yellow'
        goal_color = 'blue'
        start_color = 'black'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        if mode == 'human':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(-20, 20)
            ax.set_ylim(-20, 20)
            for agent in self.agents:
                agent_circle = plt.Circle(agent.get_position(), agent.radius, fill=False, color='r')
                ax.add_artist(agent_circle)
            plt.show()
        elif mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            boundary = self.world.boundary[1]
            ax.tick_params(labelsize=16)
            ax.set_xlim(-(boundary + 1), boundary + 1)
            ax.set_ylim(-(boundary + 1), boundary + 1)
            ax.set_xlabel('x(nautical miles)', fontsize=16)
            ax.set_ylabel('y(nautical miles)', fontsize=16)

            agents_positions = [[self.states[i][j].position for j in range(self.agent_num)]
                                for i in range(len(self.states))]

            goal_x = [self.agents[i].gx for i in range(self.agent_num)]
            goal_y = [self.agents[i].gy for i in range(self.agent_num)]
            start_x = [self.agents[i].sx for i in range(self.agent_num)]
            start_y = [self.agents[i].sy for i in range(self.agent_num)]
            goal = mlines.Line2D(goal_x, goal_y, color=goal_color, marker='*', linestyle='None', markersize=15,
                                 label='Goal')
            start = mlines.Line2D(start_x, start_y, color=start_color, marker='*', linestyle='None', markersize=15,
                                  label='Goal')
            conflict_num = plt.text(-boundary, boundary, 'conflict_num %d' % (self.conflict_num_episode), color='red', fontsize=18)
            label = plt.text(boundary, boundary, 'maddpg', color='blue', fontsize=18)
            ax.add_artist(goal)
            ax.add_artist(start)
            ax.add_artist(conflict_num)
            ax.add_artist(label)

            for k in range(len(self.states)):
                if k % 5 == 0 or k == len(self.states) - 1:
                    agents = [plt.Circle(agents_positions[k][i], self.agents[i].radius, fill=False, color=cmap(i % 10))
                              for i in range(self.agent_num)]
                    # agent_numbers = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset, str(i),
                    #                           color='black', fontsize=12) for i in range(self.agent_num)]
                    for i in range(self.agent_num):
                        agent = agents[i]
                        ax.add_artist(agent)

                if k != 0:
                    nav_directions = [plt.Line2D((self.states[k - 1][i].px, self.states[k][i].px),
                                               (self.states[k - 1][i].py, self.states[k][i].py),
                                               color=cmap(i), ls='solid')
                                     for i in range(self.agent_num)]
                    # agent_directions = [plt.Line2D((self.states[k - 1][i].px, self.states[k][i].px),
                    #                                (self.states[k - 1][i].py, self.states[k][i].py),
                    #                                color=cmap(i), ls='solid')
                    #                     for i in range(self.agent_num)]
                    #
                    # for agent_direction in agent_directions:
                    #     ax.add_artist(agent_direction)
                    for nav_direction in nav_directions:
                        ax.add_artist(nav_direction)
            plt.show()

        elif mode == 'video':
            pass

    def m_render(self,mode='human'):
        self.viewer.geoms.clear()
        self.viewer.onetime_geoms.clear()
        
        if mode=='human':
            for agent in self.agents:
                circle=rendering.make_circle(agent.radius)
                graph_transform=rendering.Transform(translation=tuple([i*2 +self.viewer.width/2 for i in  agent.get_position()]))
                circle.add_attr(graph_transform)
                self.viewer.add_geom(circle)
            return self.viewer.render(return_rgb_array=mode == 'rgb_array')
        elif mode=='traj':
            agents_positions = [[self.states[i][j].position for j in range(self.agent_num)] for i in range(len(self.states))]
            goal_x = [self.agents[i].gx for i in range(self.agent_num)]
            goal_y = [self.agents[i].gy for i in range(self.agent_num)]
            start_x = [self.agents[i].sx for i in range(self.agent_num)]
            start_y = [self.agents[i].sy for i in range(self.agent_num)]


        pass

# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n

# environment for all agents in the multiagent world of ppo
class MultiAgentEnv_ppo(MultiAgentEnv):
    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, args = None, shared_viewer=True):
        super(MultiAgentEnv, self).__init__()
        self.world = world
        self.agents = self.world.agents
        self.agent_num = len(self.agents)
        self.args = args
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment setting
        self.simulation_done = None
        # state list
        self.states = None
        # action list
        self.action_list = [10, 5, 0, -5, -10] # 角度
        self.agent_times = None
        # goal setting
        self.goal_size = 2.0 # 海里
        # nmac
        self.nmac_size = 0.082
        # evaluate metric
        self.success_num = 0
        self.collision_num = 0
        self.nmac_num = 0
        self.exit_boundary_num = 0
        self.conflict_num_episode = None
        self.nmac_num_episode = None
        self.collision_value = None
        
        if args.render:
            self.viewer= rendering.Viewer(700,700)   # 画板的长和宽
            # self.viewer.close()

    def route_deviation_rate(self):
        deviation_rates = []
        for i, agent in enumerate(self.agents):
            d0 = agent.get_dist_togoal()
            d1 = agent.route_len
            deviation_rate = (d1 - d0) / d0
            deviation_rates.append(deviation_rate)
        return deviation_rates

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        for i, agent in enumerate(self.agents):
            policy = PPO(self.args, agent.agent_id)
            agent.set_policy(policy)
            agent.route_len = 0
        self.agents = self.world.agents
        self.agent_num = len(self.agents)
        self.simulation_done = False
        self.conflict_num_episode = 0
        self.nmac_num_episode = 0
        # record observations for each agent
        obs = [agent.get_full_state() for agent in self.agents]
        self.agent_times = [0 for _ in range(self.agent_num)]
        self.collision_value = []
        self.states = []
        self.states.append(obs)
        obs = self.get_obs(obs)

        self.collision_num = 0
        self.nmac_num = 0
        self.exit_boundary_num = 0
        self.success_num = 0

        return obs

    def step(self, actions):
        """
        缺一个判断何时检测到潜在冲突, 以及何时回归intention_v
        :param actions: [0, 1, 2, 1, 2]
        :return:
        """
        done_signals = []
        for i in range(self.agent_num):
            done_signals.append(0)
        # update agents pos
        for i, agent in enumerate(self.agents):
            # no potential collision
            if agent.done == 0:
                v_intent = agent.get_intention_v()
                agent.step((v_intent[0], v_intent[1]))
                self.agent_times[i] += 1
            elif agent.done == 3:
                delta_theta = self.action_list[actions[i]]
                a = agent.theta * (180 / pi)
                a += delta_theta
                a %= 360
                agent.theta = a * (pi / 180)
                vx = agent.v_pref * cos(agent.theta)
                vy = agent.v_pref * sin(agent.theta)
                agent.step((vx, vy))
                self.agent_times[i] += 1
            else:
                continue

        # update agents status
        Exit_Boundary = []
        for i, agent in enumerate(self.agents):
            px = agent.px
            py = agent.py
            if agent.done == 1 or agent.done == 2:
                Exit_Boundary.append(False)
                continue
            elif px < self.world.boundary[0] or px > self.world.boundary[1] or py < self.world.boundary[2] or py > self.world.boundary[3]:
                Exit_Boundary.append(True)
                self.exit_boundary_num += 1
                agent.done = 2
                done_signals[i] = 2
            else:
                Exit_Boundary.append(False)

        dist_to_goal = np.array([agent.get_dist_to_goal() for agent in self.agents])
        Reach_Goal = []
        for i, agent in enumerate(self.agents):
            if agent.done == 1 or agent.done == 2:
                Reach_Goal.append(False)
                continue
            elif dist_to_goal[i] <= self.goal_size:
                Reach_Goal.append(True)
                agent.px, agent.py = agent.gx + 0.0001, agent.gy + 0.0001
                self.success_num += 1
                agent.done = 1
                done_signals[i] = 1
            else:
                Reach_Goal.append(False)

        next_obs = [agent.get_full_state() for agent in self.agents]
        self.states.append(next_obs)
        # generate collision network
        cnet = Collision_Network(next_obs)
        collision_mat = cnet.collision_matrix()
        dist_mat = cnet.dist_matrix()
        row, col = np.diag_indices_from(collision_mat)
        collision_mat[row, col] = 0
        dist_mat[row, col] = inf
        reach_goals_idx = np.where(Reach_Goal)[0]
        collision_mat[reach_goals_idx, :] = 0
        collision_mat[:, reach_goals_idx] = 0
        dist_mat[reach_goals_idx, :] = inf
        dist_mat[:, reach_goals_idx] = inf
        exit_idx = np.where(Exit_Boundary)[0]
        collision_mat[exit_idx, :] = 0
        collision_mat[:, exit_idx] = 0
        dist_mat[exit_idx, :] = inf
        dist_mat[:, exit_idx] = inf

        # update agent done --- 0 or 3
        for i, agent in enumerate(self.agents):
            if agent.done != 1 and agent.done != 2:
                collision_value_row = collision_mat[i]
                d = dist_mat[i]
                self.collision_num += sum(collision_value_row == 1) / 2
                self.conflict_num_episode += sum(collision_value_row == 1) / 2
                self.nmac_num += sum(d <= self.nmac_size) / 2
                self.nmac_num_episode += sum(d <= self.nmac_size) / 2
                # # no collision resolution
                # agent.done = 0
                if np.sum(collision_value_row) <= 0.01:
                    agent.done = 0
                else:
                    agent.done = 3

        # compute reward
        reward, c_v = self.reward_callback(self.world, collision_mat, Reach_Goal, Exit_Boundary, dist_to_goal)
        self.collision_value.append(c_v)

        terminal = True
        # 判断是否到达终态
        for i, agent in enumerate(self.agents):
            if agent.done == 0 or agent.done == 3:
                terminal = False
                break

        self.simulation_done = terminal
        info = {'current_time_reach_goals': Reach_Goal, 'current_time_exit_boundary': Exit_Boundary, 'simulation_done': terminal}

        return self.get_obs(next_obs), reward, done_signals, info

    def get_obs(self, obs):
        """
        input obs list of FullState transform to 2-dimension numpy
        :param obs:
        :return: obs matrix with shape (agent_num, action_num)
        """
        agent_num = len(obs)
        obs_mat = np.zeros((agent_num, 9))
        for i in range(len(obs)):
            ob = obs[i]
            obs_mat[i][0] = ob.px
            obs_mat[i][1] = ob.py
            obs_mat[i][2] = ob.vx
            obs_mat[i][3] = ob.vy
            obs_mat[i][4] = ob.radius
            obs_mat[i][5] = ob.gx
            obs_mat[i][6] = ob.gy
            obs_mat[i][7] = ob.v_pref
            obs_mat[i][8] = ob.theta

        return obs_mat

    def render(self, mode='human'):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        from matplotlib import patches
        import matplotlib.lines as mlines
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        x_offset = 0.11
        y_offset = 0.11
        cmap = plt.cm.get_cmap('hsv', 10)
        agent_color = 'yellow'
        goal_color = 'blue'
        start_color = 'black'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        if mode == 'human':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(-20, 20)
            ax.set_ylim(-20, 20)
            for agent in self.agents:
                agent_circle = plt.Circle(agent.get_position(), agent.radius, fill=False, color='r')
                ax.add_artist(agent_circle)
            plt.show()
        elif mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            boundary = self.world.boundary[1]
            ax.tick_params(labelsize=16)
            ax.set_xlim(-(boundary + 1), boundary + 1)
            ax.set_ylim(-(boundary + 1), boundary + 1)
            ax.set_xlabel('x(nautical miles)', fontsize=16)
            ax.set_ylabel('y(nautical miles)', fontsize=16)

            agents_positions = [[self.states[i][j].position for j in range(self.agent_num)]
                                for i in range(len(self.states))]

            goal_x = [self.agents[i].gx for i in range(self.agent_num)]
            goal_y = [self.agents[i].gy for i in range(self.agent_num)]
            start_x = [self.agents[i].sx for i in range(self.agent_num)]
            start_y = [self.agents[i].sy for i in range(self.agent_num)]
            goal = mlines.Line2D(goal_x, goal_y, color=goal_color, marker='*', linestyle='None', markersize=15,
                                 label='Goal')
            start = mlines.Line2D(start_x, start_y, color=start_color, marker='*', linestyle='None', markersize=15,
                                  label='Goal')
            conflict_num = plt.text(-boundary, boundary, 'conflict_num %d' % (self.conflict_num_episode), color='red', fontsize=18)
            label = plt.text(boundary, boundary, 'ppo', color='blue', fontsize=18)
            ax.add_artist(goal)
            ax.add_artist(start)
            ax.add_artist(conflict_num)
            ax.add_artist(label)

            for k in range(len(self.states)):
                if k % 5 == 0 or k == len(self.states) - 1:
                    agents = [plt.Circle(agents_positions[k][i], self.agents[i].radius, fill=False, color=cmap(i % 10))
                              for i in range(self.agent_num)]
                    # agent_numbers = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset, str(i),
                    #                           color='black', fontsize=12) for i in range(self.agent_num)]
                    for i in range(self.agent_num):
                        agent = agents[i]
                        ax.add_artist(agent)

                if k != 0:
                    nav_directions = [plt.Line2D((self.states[k - 1][i].px, self.states[k][i].px),
                                               (self.states[k - 1][i].py, self.states[k][i].py),
                                               color=cmap(i), ls='solid')
                                     for i in range(self.agent_num)]
                    # agent_directions = [plt.Line2D((self.states[k - 1][i].px, self.states[k][i].px),
                    #                                (self.states[k - 1][i].py, self.states[k][i].py),
                    #                                color=cmap(i), ls='solid')
                    #                     for i in range(self.agent_num)]
                    #
                    # for agent_direction in agent_directions:
                    #     ax.add_artist(agent_direction)
                    for nav_direction in nav_directions:
                        ax.add_artist(nav_direction)
            plt.show()

        elif mode == 'video':
            pass

    def m_render(self,mode='human'):
            self.viewer.geoms.clear()
            self.viewer.onetime_geoms.clear()
            
            if mode=='human':
                for agent in self.agents:
                    circle=rendering.make_circle(agent.radius)
                    graph_transform=rendering.Transform(translation=tuple([i*2 +self.viewer.width/2 for i in  agent.get_position()]))
                    circle.add_attr(graph_transform)
                    self.viewer.add_geom(circle)
                return self.viewer.render(return_rgb_array=mode == 'rgb_array')
            elif mode=='traj':
                agents_positions = [[self.states[i][j].position for j in range(self.agent_num)] for i in range(len(self.states))]
                goal_x = [self.agents[i].gx for i in range(self.agent_num)]
                goal_y = [self.agents[i].gy for i in range(self.agent_num)]
                start_x = [self.agents[i].sx for i in range(self.agent_num)]
                start_y = [self.agents[i].sy for i in range(self.agent_num)]
