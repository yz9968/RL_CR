import numpy as np
from numpy.linalg import norm
from multiagent_particle_envs.multiagent.core import World, Agent, Landmark
from multiagent_particle_envs.multiagent.scenario import BaseScenario
from multiagent_particle_envs.multiagent.core import *
import math

# 图强化学习环境

class Scenario(BaseScenario):
    def __init__(self):
        # penalty reward setting
        self.collision_level1 = -0.25
        self.collision_level2 = -1.0
        self.collision_penalty = -10.0
        self.dist_to_goal_penalty = -1.0
        self.time_penalty = -0.5
        self.angle_dev = 2.0
        self.exit_boundary = -10.0

    def make_world(self):
        world = World()
        # set size of the world
        world.set_world(-160, 160, -160, 160)
        # set any world properties first
        self.num_agents = 50
        self.num_landmarks = self.num_agents
        world.collaborative = True
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # add agents
        world.agents = [Agent() for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.set_agent_id(i)
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks ----> goal
        world.landmarks = [Landmark() for i in range(self.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for i, agent in enumerate(world.agents):
            px, py, gx, gy, vx, vy, theta = self.generate_random_agent_attribute(world, i)
            # px, py, gx, gy, vx, vy, theta = self.generate_circle_agent_attribute(world, i)
            world.agents[i].set(px, py, gx, gy, vx, vy, theta)
            world.agents[i].set_time_step(world.dt)

    def generate_random_agent_attribute(self, world, agent_id):
        square_width = world.boundary[1]
        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1
        # generate start pos
        while True:
            px = np.random.random() * square_width * sign
            py = (np.random.random() - 0.5) * square_width * 2
            collide = False
            for agent in world.agents:
                if not agent.px or not agent.py:
                    continue
                elif norm((px - agent.px, py - agent.py)) < 2 * agent.radius:
                    collide = True
                    break
            if not collide:
                break
        # generate goal pos
        while True:
            gx = np.random.random() * square_width * -sign
            gy = (np.random.random() - 0.5) * square_width * 2
            collide = False
            for agent in world.agents:
                if not agent.gx or not agent.gy:
                    continue
                elif norm((gx - agent.gx, gy - agent.gy)) < 2 * agent.radius:
                    collide = True
                    break
            if not collide:
                break

        v = vec_normlization(sub(Vector2(gx, gy), Vector2(px, py)))
        v_norm = vec_multipli(v, world.agents[agent_id].v_pref)
        theta = vec_theta(v_norm)
        vx, vy = v_norm.x, v_norm.y
        world.agents[agent_id].set_intent_v((vx, vy, theta))

        return px, py, gx, gy, vx, vy, theta

    def generate_circle_agent_attribute(self, world, agent_id):
        circle_radius = world.boundary[1] - 10
        angle = agent_id * 2 * np.pi / self.num_agents
        px = circle_radius * np.cos(angle)
        py = circle_radius * np.sin(angle)
        gx = -px
        gy = -py
        v = vec_normlization(sub(Vector2(gx, gy), Vector2(px, py)))
        v_norm = vec_multipli(v, world.agents[agent_id].v_pref)
        theta = vec_theta(v_norm)
        vx, vy = v_norm.x, v_norm.y
        world.agents[agent_id].set_intent_v((vx, vy, theta))

        return px, py, gx, gy, vx, vy, theta

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, world, collision_mat, reach_goals, exit_boundary, dist_to_goal):
        reward = []
        c_v = 0
        # row, col = np.diag_indices_from(collision_mat)
        # collision_mat[row, col] = 0
        # reach_goals_idx = np.where(reach_goals)[0]
        # collision_mat[reach_goals_idx, :] = 0
        # collision_mat[:, reach_goals_idx] = 0
        for i, agent in enumerate(world.agents):
            if reach_goals[i]:
                reward.append(0)
            elif exit_boundary[i]:
                reward.append(self.exit_boundary)
            elif agent.done != 3:
                reward.append(0)
            else:
                collision_value_row = collision_mat[i]
                angle = agent.angle_intent_current_v()
                r1 = sum(np.logical_and(collision_value_row > 0, collision_value_row <= 0.5)) * self.collision_level1
                r2 = sum(np.logical_and(collision_value_row > 0.5, collision_value_row < 1)) * self.collision_level2
                r3 = sum(collision_value_row == 1) * self.collision_penalty
                r4 = (math.cos(angle) - 1) * self.angle_dev
                # r4 = delta_dist_to_goal[i] * self.head_goal_reward
                # r4 = dist_to_goal[i] * self.dist_to_goal_penalty
                r = r1 + r2 + r3 + r4
                # collision_value compute
                c1 = np.logical_and(collision_value_row > 0, collision_value_row <= 0.5)
                c1 = np.where(c1)[0]
                c2 = np.logical_and(collision_value_row > 0.5, collision_value_row < 1)
                c2 = np.where(c2)[0]
                v1 = sum(np.take(collision_value_row, c1)) * -self.collision_level1
                v2 = sum(np.take(collision_value_row, c2)) * -self.collision_level2
                v3 = sum(collision_value_row == 1) * 2.0
                c_v = v1 + v2 + v3
                reward.append(r)
        return reward, c_v

    # def reward(self, agent, world):
    #     # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
    #     rew = 0
    #     for l in world.landmarks:
    #         dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
    #         rew -= min(dists)
    #     if agent.collide:
    #         for a in world.agents:
    #             if self.is_collision(a, agent):
    #                 rew -= 1
    #     return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
