import logging
import time
import os

from common.arguments import get_args
from common.utils import make_env

from DGN_AC.runner_DGN_AC import Runner_DGN_AC

if __name__ == '__main__':
    # get the params
    args = get_args()
    args.scenario_name = "cr_grl"
    args.viewer_step = 5
    # args.render_mode='human'

    # # parameters for debug
    # args.num_episodes = 10 # default 5001
    # args.evaluate_rate = 2 # default 50
    # args.evaluate_episodes = 2 # default 10
    # args.eval_model_episode = 10
    # args.save_rate = 1 # default 500
    # args.n_agents = 6 # default 30

    # args.num_episodes = 500 # default 5001
    # args.evaluate_rate = 5 # default 50
    # args.evaluate_episodes = 5 # default 10
    # args.save_rate = 5 # default 500

    # need to try: 8 15 20 30 50
    # args.num_episodes = 200 # default 5001
    args.neighbor_dist = 25

    args.eval_model_episode = 100
    args.batch_size = 64
    args.render = False
    args.train_render = False
    args.train_render_rate = 5
    args.eval_model_render_rate = 10
    args.render_mode = 'human'
    args.attach_info = '_ppo'
    args.save_rate= 25
    
    args.n_agents = 30  # default 30
    bool_train = False
    bool_eval = True





    args.log_info = '_{}_{}'.format(args.n_agents,"train" if bool_train else ("eval" if bool_eval else "null"))
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    log_dir=args.save_dir + '/' + args.scenario_name +'{1}/{0}_agent/'.format(args.n_agents,args.attach_info)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    time_num = time.strftime("%m%d%H%M", time.localtime(time.time()))
    logging.basicConfig(
        level=logging.INFO, format=LOG_FORMAT,
        filename= log_dir + 'run{}_{}.log'.format(args.log_info,time_num),
        filemode='a',  # 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志，a是追加模式，默认如果不写的话，就是追加模式
    )

    env, args = make_env(args)
    runner = Runner_DGN_AC(args, env)

    if bool_train:
        runner.run()
    if bool_eval:
        runner.evaluate_model()
