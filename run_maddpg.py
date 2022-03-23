from ast import arg
from common.arguments import get_args
from maddpg.maddpg_task import Task_maddpg

if __name__ == "__main__":
    args = get_args()

    args = get_args()
    args.scenario_name = "cr_maddpg"
    # args.num_episodes = 10 # default 5001
    # args.evaluate_rate = 1 # default 50
    # args.evaluate_episodes = 2 # default 10
    # args.save_rate = 1 # default 500

    # args.n_agents = 6

    # args.num_episodes = 500 # default 5001
    # args.evaluate_rate = 5 # default 50
    # args.evaluate_episodes = 5 # default 10
    # args.save_rate = 5 # default 500

    # need to try: 8 15 20 30 50
    args.n_agents = 50 # default 30 
    args.render=False

    from common.utils import make_env
    # env=None
    env, args = make_env(args)

    maddpg_task = Task_maddpg(args, env)
    maddpg_task.train()
    maddpg_task.evaluate_model()
