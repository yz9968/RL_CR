import logging
import time
import os

from DGN.runner_DGN import Runner_DGN

from common.arguments import get_args
from common.utils import make_env


if __name__ == '__main__':
    # get the params
    args = get_args()
    args.scenario_name="cr_grl"
    args.viewer_step=5
    # args.render_mode='human'


    args.neighbor_dist = 20

    args.eval_model_episode = 100
    args.render = False
    args.train_render = False
    args.train_render_rate = 5
    args.eval_model_render_rate = 10
    args.render_mode = 'human'
    args.attach_info='_mask_done'
    args.save_rate= 25
    
    args.n_agents = 50  # default 30
    bool_train = True
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
    runner = Runner_DGN(args, env)

    if bool_train:
        runner.run()
    if bool_eval:
        runner.evaluate_model()

