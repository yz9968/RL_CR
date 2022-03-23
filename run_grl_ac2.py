from DGN.DGN2.runner_DGN_AC import Runner_DGN_AC

from common.arguments import get_args
from common.utils import make_env



if __name__ == '__main__':
    # get the params
    args = get_args()
    args.scenario_name="cr_grl"
    args.eval_model_episode=180
    args.viewer_step=5
    args.render_mode='traj'

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
    args.n_agents = 4 # default 30 
    args.render=False
    args.attach_info='_single'

    env, args = make_env(args)
    runner = Runner_DGN_AC(args, env)

    runner.run()
    runner.evaluate_model()
