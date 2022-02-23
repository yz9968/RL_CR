#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-12 16:02:24
LastEditor: John
LastEditTime: 2021-11-30 18:39:19
Discription: 
Environment: 
'''
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.font_manager import FontProperties  # 导入字体模块

import inspect
import functools
import torch
import logging


def chinese_font():
    ''' 设置中文字体，注意需要根据自己电脑情况更改字体路径，否则还是默认的字体
    '''
    try:
        font = FontProperties(
        fname=r'C:\Windows\Fonts\Deng.ttf', size=15) # fname系统字体路径，此处是mac的
    except:
        font = None
    return font

def plot_rewards_cn(rewards, ma_rewards, plot_cfg, tag='train'):
    ''' 中文画图
    '''
    sns.set()
    plt.figure()
    plt.title(u"{}环境下{}算法的学习曲线".format(plot_cfg.env_name,
              plot_cfg.algo_name), fontproperties=chinese_font())
    plt.xlabel(u'回合数', fontproperties=chinese_font())
    plt.plot(rewards)
    plt.plot(ma_rewards)
    plt.legend((u'奖励', u'滑动平均奖励',), loc="best", prop=chinese_font())
    if plot_cfg.save:
        plt.savefig(plot_cfg.result_path+f"{tag}_rewards_curve_cn")
    plt.show()


def plot_rewards(rewards, ma_rewards, plot_cfg, tag='train'):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve on {} of {} for {}".format(
        plot_cfg.device, plot_cfg.algo_name, plot_cfg.env_name))
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(ma_rewards, label='ma rewards')
    plt.legend()
    if plot_cfg.save:
        plt.savefig(plot_cfg.result_path+"{}_rewards_curve".format(tag))
    plt.show()


def plot_losses(losses, algo="DQN", save=True, path='./'):
    sns.set()
    plt.figure()
    plt.title("loss curve of {}".format(algo))
    plt.xlabel('epsiodes')
    plt.plot(losses, label='rewards')
    plt.legend()
    if save:
        plt.savefig(path+"losses_curve")
    plt.show()


def save_results(rewards, ma_rewards, tag='train', path='./results'):
    ''' 保存奖励
    '''
    np.save(path+'{}_rewards.npy'.format(tag), rewards)
    np.save(path+'{}_ma_rewards.npy'.format(tag), ma_rewards)
    print('结果保存完毕！')


def make_dir(*paths):
    ''' 创建文件夹
    '''
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def del_empty_dir(*paths):
    ''' 删除目录下所有空文件夹
    '''
    for path in paths:
        dirs = os.listdir(path)
        for dir in dirs:
            if not os.listdir(os.path.join(path, dir)):
                os.removedirs(os.path.join(path, dir))


def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def make_env(args):
    from multiagent_particle_envs.multiagent.environment import MultiAgentEnv,  MultiAgentEnv_maddpg,MultiAgentEnv_ppo
    import multiagent_particle_envs.multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(args.scenario_name + ".py").Scenario()

    # create world
    world = scenario.make_world(args.n_agents)

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)
    USE_CUDA = torch.cuda.is_available()
    args.device = device
    # create multiagent environment
    if args.scenario_name == 'cr_maddpg':
        env = MultiAgentEnv_maddpg(world, scenario.reset_world, scenario.reward, args=args)
    elif args.scenario_name == 'cr_ppo':
        env = MultiAgentEnv_ppo(world, scenario.reset_world, scenario.reward, args=args)
    else:
        logging.error("no such scenario")
    # env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    # 以下部分添加到MultiAgentEnv中
    # args.n_agents = env.agent_num
    args.obs_shape = [9 for _ in range(args.n_agents)]
    args.action_shape = [5 for _ in range(args.n_agents)]
    # args.action_shape = [1 for _ in range(args.n_agents)]
    args.high_action = 1
    args.low_action = -1
    return env, args
