import numpy as np
from math import sin, cos, tan, pi, asin, acos, atan, sqrt, exp, inf
from numpy.linalg import norm
import torch

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self, px, py, vx, vy, radius, gx, gy, v_pref, theta, done):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None
        # 全局状态相关属性
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        # 安全距离
        self.radius = radius
        self.gx = gx
        self.gy = gy
        self.v_pref = v_pref
        # 航向角
        self.theta = theta
        # 状态
        self.done = done

        self.position = (self.px, self.py)
        self.goal_position = (self.gx, self.gy)
        self.velocity = (self.vx, self.vy)

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

class ActionXY(Action):
    def __init__(self, vx, vy):
        super(ActionXY, self).__init__()
        self.vx = vx
        self.vy = vy

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()
        self.px = None
        self.py = None

    def set_position(self, px, py):
        self.px = px
        self.py = py


# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # 智能体相关属性
        self.v_pref = 7.5  # 海里 / minute
        self.radius = 5.0  # 海里
        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.sx = None
        self.sy = None
        self.theta = None
        self.agent_id = None
        self.v_intent = None
        self.time_step = 0.4 # minute
        self.policy = None
        self.done = 0  # 0：exist but no potential collision；1：reach_goal；2：exit_boundary ；3：exist but detect potential collision
        self.dist_to_goal = None
        self.route_len = 0
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta, self.done)
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

    def set_policy(self, policy):
        self.policy = policy

    def set_agent_id(self, agent_id):
        self.agent_id = agent_id

    def set_time_step(self, dt):
        self.time_step = dt

    def sample_random_attributes(self):
        self.v_pref = np.random.uniform(0.5, 1.5)
        self.radius = np.random.uniform(0.3, 0.5)

    def set(self, px, py, gx, gy, vx, vy, theta):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta
        self.sx = px
        self.sy = py

    def get_dist_togoal(self):
        return dist(Vector2(self.sx, self.sy), Vector2(self.gx, self.gy))

    def compute_position(self, action, delta_t):
        px = self.px + action.vx * delta_t
        py = self.py + action.vy * delta_t
        d = dist(Vector2(self.px, self.py), Vector2(px, py))
        self.route_len += d
        return px, py

    def step(self, action):
        """
        perform an action and update the state
        :param action: tuple (vx, vy)
        :return:
        """
        action = ActionXY(action[0], action[1])
        pos = self.compute_position(action, self.time_step)
        self.px, self.py = pos

    def select_action(self, o, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            # u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
            u = np.random.randint(5)
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            pi = self.policy.actor(inputs).squeeze(0)
            # print('{} : {}'.format(self.name, pi))
            u = pi.cpu().detach().numpy()
            noise = noise_rate * 1 * np.random.randn(*u.shape)  # gaussian noise
            u += noise
            u = np.clip(u, -1, 1)
            u = np.argmax(u)
        return u

    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)

    # def step(self, action):
    #     """
    #     perform an action and update the state
    #     :param action:
    #     :return:
    #     """
    #     a = self.theta * (180 / pi)
    #     a += action
    #     a %= 360
    #     self.theta = a * (pi / 180)
    #     self.vx = self.v_pref * cos(self.theta)
    #     self.vy = self.v_pref * sin(self.theta)
    #     action = ActionXY(self.vx, self.vy)
    #     pos = self.compute_position(action, self.time_step)
    #     self.px, self.py = pos

    def get_position(self):
        return self.px, self.py

    def get_goal_position(self):
        return self.gx, self.gy

    def get_full_state(self):
        return AgentState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta, self.done)

    def get_intention_v(self):
        v = vec_normlization(sub(Vector2(self.gx, self.gy), Vector2(self.px, self.py)))
        v_norm = vec_multipli(v, self.v_pref)
        theta = vec_theta(v_norm)
        vx, vy = v_norm.x, v_norm.y
        self.v_intent = (vx, vy, theta)
        return self.v_intent

    def angle_intent_current_v(self):
        """
        :return: angle between current v and intention v
        """
        intent = self.get_intention_v()
        angle = compute_angle(Vector2(intent[0], intent[1]), Vector2(self.vx, self.vy))
        return angle

    def set_intent_v(self, velocity):
        """
        :param velocity: (vx, vy, theta)
        :return: tuple
        """
        self.v_intent = (velocity[0], velocity[1], velocity[2])


    def get_dist_to_goal(self):
        self.dist_to_goal = norm(np.array(self.get_position()) - np.array(self.get_goal_position()))
        return self.dist_to_goal

    def reach_goal(self):
        return self.dist_to_goal <= self.radius

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # size
        self.shape = None
        self.boundary = None
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.4 # minute
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

    def set_world(self, min_x, max_x, min_y, max_y):
        self.boundary = (min_x, max_x, min_y, max_y)

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise                
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]        
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]

class Vector2():
    """
    Defines a two-dimensional vector.
    """

    def __init__(self, x=0.0, y=0.0):
        """
        Constructs and initializes a two-dimensional vector from the specified xy-coordinates.

        Args:
            x (float): The x-coordinate of the two-dimensional vector.
            y (float):The y-coordinate of the two-dimensional vector.
        """
        self.x = x
        self.y = y

    def __str__(self):
        return "Vector2(x={}, y={})".format(self.x, self.y)


def vec_norm(vector):
    """
    :param vector: Vector2
    :return: norm of vector
    """
    return sqrt(vector.x * vector.x + vector.y * vector.y)

def vec_normlization(vector):
    mod = vec_norm(vector)
    return Vector2(vector.x / mod, vector.y / mod)

def vec_theta(vector):
    x = vector.y
    y = vector.x
    angle = 0
    if x == 0 and y > 0:
        angle = 0
    if x == 0 and y < 0:
        angle = 180
    if y == 0 and x > 0:
        angle = 90
    if y == 0 and x < 0:
        angle = 270
    if x > 0 and y > 0:
        angle = atan(x / y) * 180 / pi
    elif x < 0 and y > 0:
        angle = 360 + atan(x / y) * 180 / pi
    elif x < 0 and y < 0:
        angle = 180 + atan(x / y) * 180 / pi
    elif x > 0 and y < 0:
        angle = 180 + atan(x / y) * 180 / pi
    return angle * pi / 180


def sub(vector1, vector2):
    """
    compute relative vector

    Args:
        vector1 (Vector2): The top row of the two-dimensional square matrix.
        vector2 (Vector2): The bottom row of the two-dimensional square matrix.

    Returns:
        Vector2: relative 1-2 : vector1 - vector2
    """
    return Vector2(vector1.x - vector2.x, vector1.y - vector2.y)


def dist(p1, p2):
    """
    compute the distance of p1 to p2
    :param p1: Vector2
    :param p2: Vector2
    :return: 两点之间距离
    """
    x = p1.x - p2.x
    y = p1.y - p2.y
    return sqrt(x * x + y * y)


def matmul(vector1, vector2):
    """
    Computes the neiji of the specified two-dimensional vectors.

    Args:
        vector1 (Vector2): The top row of the two-dimensional square matrix.
        vector2 (Vector2): The bottom row of the two-dimensional square matrix.

    Returns:
        float: The neiji of the two-dimensional square matrix.
    """
    return vector1.x * vector2.x + vector1.y * vector2.y

def compute_angle(vector1, vector2):
    """
    计算夹角
    :param vector1:
    :param vector2:
    :return:
    """
    mid = matmul(vector1, vector2) / (vec_norm(vector1) * vec_norm(vector2))
    if mid > 1:
        mid = 1
    if mid < -1:
        mid = -1
    angle = acos(mid)

    return angle

def vec_multipli(vector, a):
    return Vector2(vector.x * a, vector.y * a)

class Collision_Detection():

    def __init__(self, p1, p2, va, vb, safe_r):
        assert isinstance(p1, Vector2) and isinstance(p2, Vector2) and isinstance(va, Vector2) and isinstance(vb, Vector2)
        self.p1 = p1
        self.p2 = p2
        self.va = va
        self.vb = vb
        self.safe_radius = safe_r
        self.AB = sub(p2, p1)
        self.vr = sub(va, vb)
        self.AB_mod = dist(self.p1, self.p2)
        self.vr_mod = vec_norm(self.vr)
        mid = matmul(self.vr, self.AB) / (self.vr_mod * self.AB_mod)
        if mid > 1:
            mid = 1
        if mid < -1:
            mid = -1
        self.gamma = acos(mid)
        self.L_low = None

    def calc_angle_Vr_Pab(self):
        """
        gamma
        :return:
        """
        return self.gamma

    def calc_angle_cc(self):
        """
        angle of cc
        :return:
        """
        a = self.safe_radius / self.AB_mod
        if a > 1:
            a = 1
        if a < -1:
            a = -1
        self.alpha = asin(a)
        return self.alpha

    def collision_detect(self):
        """
        判断是否存在潜在冲突
        """
        # 存在潜在冲突
        if self.AB_mod <= self.safe_radius:
            return True
        elif self.calc_angle_Vr_Pab() <= self.calc_angle_cc():
            return True
        else:
            return False

    def calc_collision_time(self):
        """
        计算预计冲突时间t
        :return: float t
        """
        t = inf
        if self.AB_mod <= self.safe_radius:
            t = 0
            return t
        elif self.collision_detect():
            #计算冲突时间
            self.L_low = self.AB_mod * cos(self.gamma) - sqrt(self.AB_mod * self.AB_mod * cos(self.gamma) * cos(self.gamma) - (self.AB_mod * self.AB_mod - self.safe_radius * self.safe_radius))
            t = self.L_low / self.vr_mod
        return t

    def calc_collision_value(self):
        """
        计算冲突程度值，范围[0,1] 没有潜在冲突为0,存在潜在冲突为0-1，当前时刻已经冲突为1
        :return: float
        """
        return exp(-self.calc_collision_time())

"""
根据冲突检测模块和当前观测向量建立冲突网络
"""
class Collision_Network():
    """
    1.输入所有智能体当前观测向量，输出冲突网络矩阵，智能体数量为n,矩阵维度为(n, n)
    2.对冲突网络进行调控
    """
    def __init__(self, obs):
        self.obs = obs
        self.agent_num = len(self.obs)
        self.cm = np.zeros((self.agent_num, self.agent_num), dtype='float32')
        self.dm = np.zeros((self.agent_num, self.agent_num), dtype='float32')

    def collision_matrix(self):
        """
        矩阵中0代表没有潜在冲突，1代表当前时刻存在飞行冲突，0-1表示存在潜在冲突，值越大冲突程度越高
        :return: collision_matrix
        """
        for i in range(self.agent_num):
            if self.obs[i].done == 1 or self.obs[i] == 2:
                self.cm[i, :] = 0
                self.cm[:, i] = 0
                continue
            own_id = i
            own_pos = Vector2(self.obs[i].px, self.obs[i].py)
            own_v = Vector2(self.obs[i].vx, self.obs[i].vy)
            own_safe_r = self.obs[i].radius
            for j in range(own_id + 1, self.agent_num):
                other_id = j
                other_pos = Vector2(self.obs[j].px, self.obs[j].py)
                other_v = Vector2(self.obs[j].vx, self.obs[j].vy)
                other_safe_r = self.obs[j].radius
                cd = Collision_Detection(own_pos, other_pos, own_v, other_v, max(own_safe_r, other_safe_r))
                collision = cd.collision_detect()  # True/False
                # if collision:
                #     print("agent {} exist potential collide with agent {}".format(i, j))
                collision_value = cd.calc_collision_value()
                self.cm[i][j] = collision_value
                self.cm[j][i] = collision_value

        return self.cm

    def dist_matrix(self):
        for i in range(self.agent_num):
            if self.obs[i].done == 1 or self.obs[i].done == 2:
                self.dm[i, :] = inf
                self.dm[:, i] = inf
                continue
            own_id = i
            own_pos = Vector2(self.obs[i].px, self.obs[i].py)
            for j in range(own_id + 1, self.agent_num):
                other_id = j
                other_pos = Vector2(self.obs[j].px, self.obs[j].py)
                d = dist(own_pos, other_pos)
                self.dm[i][j] = d
                self.dm[j][i] = d

        return self.dm
