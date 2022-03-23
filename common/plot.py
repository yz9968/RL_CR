#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-10-07 20:57:11
LastEditor: John
LastEditTime: 2021-09-19 23:00:36
Discription: 
Environment: 
'''
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import os


def chinese_font():
    return FontProperties(fname=r'C:\Windows\Fonts\Deng.ttf', size=15)  # 系统字体路径，此处是mac的


def plot_multi_cn(data_lists, titles=["环境-算法-数据"], xlabels=["episodes"], labels=["rewards"], save=True, save_path="./", save_name="rewards_curve"):
    from math import ceil, sqrt
    sns.set()
    plt.clf()
    subfig_num = len(data_lists)
    row_num = int(sqrt(subfig_num))
    col_num = ceil(subfig_num/row_num)

    for index in range(1, subfig_num+1):
        ax = plt.subplot(row_num, col_num, index)
        ax.plot(data_lists[index-1])
        ax.set_title(titles[index-1] if index <= len(titles) else '', fontproperties=chinese_font())
        ax.set_xlabel(xlabels[index-1] if index <= len(xlabels) else '')   # 为x轴添加标签
        ax.set_ylabel(labels[index-1] if index <= len(labels) else '')   # 为y轴添加标签
    plt.tight_layout()   # 自动调整各子图间距
    if save:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path+save_name)
    plt.show()

def plot_multi_cn_off(data_lists, titles=["环境-算法-数据"], xlabels=["episodes"], labels=["rewards"], save=True, save_path="./", save_name="rewards_curve"):
    from math import ceil, sqrt
    sns.set()
    plt.clf()
    subfig_num = len(data_lists)
    row_num = int(sqrt(subfig_num))
    col_num = ceil(subfig_num/row_num)

    for index in range(1, subfig_num+1):
        ax = plt.subplot(row_num, col_num, index)
        ax.plot(data_lists[index-1])
        ax.set_title(titles[index-1] if index <= len(titles) else '', fontproperties=chinese_font())
        ax.set_xlabel(xlabels[index-1] if index <= len(xlabels) else '')   # 为x轴添加标签
        ax.set_ylabel(labels[index-1] if index <= len(labels) else '')   # 为y轴添加标签
    plt.tight_layout()   # 自动调整各子图间距
    if save:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path+save_name)


def plot_cn(data_list, title="环境-算法-数据", xlabel="episodes", label="rewards", save=True,  save_path="./",save_name="rewards_curve"):
    sns.set()
    plt.clf()
    plt.title(title, fontproperties=chinese_font())
    plt.xlabel(xlabel, fontproperties=chinese_font())
    plt.ylabel(label,fontproperties=chinese_font())
    plt.plot(data_list)

    if save:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path+save_name)
    plt.show()

def plot_cn_off(data_list, title="环境-算法-数据", xlabel="episodes", label="rewards", save=True,  save_path="./",save_name="rewards_curve"):
    sns.set()
    plt.clf()
    plt.title(title, fontproperties=chinese_font())
    plt.xlabel(xlabel, fontproperties=chinese_font())
    plt.ylabel(label,fontproperties=chinese_font())
    plt.plot(data_list)
    if save:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path+save_name)
    # plt.show()


def plot_ioncn(data_list, title="环境-算法-数据", xlabel="episodes", label="rewards", save=True, save_name="rewards_curve", save_path="./"):
    sns.set()
    plt.ion()
    plt.title(title, fontproperties=chinese_font())
    plt.xlabel(xlabel, fontproperties=chinese_font())
    plt.ylabel(label, fontproperties=chinese_font())
    # plt.legend((label,),loc="best",prop=chinese_font())
    plt.plot(data_list)
    if save:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path+save_name)
    # plt.show()
    plt.clf()
    plt.ioff()


def plot_rewards(rewards, ma_rewards, tag="train", env='CartPole-v0', algo="DQN", save=True, path='./'):
    sns.set()
    plt.title("average learning curve of {} for {}".format(algo, env))
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(ma_rewards, label='ma rewards')
    plt.legend()
    if save:
        plt.savefig(path+"{}_rewards_curve".format(tag))
    plt.show()


def plot_rewards_cn(rewards, ma_rewards, tag="train", env='CartPole-v0', algo="DQN", save=True, path='./'):
    ''' 中文画图
    '''
    sns.set()
    plt.figure()
    plt.title(u"{}环境下{}算法的学习曲线".format(env, algo), fontproperties=chinese_font())
    plt.xlabel(u'回合数', fontproperties=chinese_font())
    plt.plot(rewards)
    plt.plot(ma_rewards)
    plt.legend((u'奖励', u'滑动平均奖励',), loc="best", prop=chinese_font())
    if save:
        plt.savefig(path+f"{tag}_rewards_curve_cn")
    # plt.show()


def plot_losses(losses, algo="DQN", save=True, path='./'):
    sns.set()
    plt.title("loss curve of {}".format(algo))
    plt.xlabel('epsiodes')
    plt.plot(losses, label='rewards')
    plt.legend()
    if save:
        plt.savefig(path+"losses_curve")
    plt.show()
