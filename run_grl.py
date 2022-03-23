from common.arguments import get_args
from common.utils import make_env
from GraphRL.GRL_task import Task_GRL
if __name__ =="__main__":
    args=get_args()
    args.scenario_name="cr_grl"

    args.num_episodes = 10 # default 5001
    args.evaluate_rate = 2 # default 50
    args.evaluate_episodes = 2 # default 10
    args.save_rate = 1 # default 500
    args.n_agents = 6 # default 30
    args.render=True


    # args.num_episodes = 500 # default 5001
    # args.evaluate_rate = 5 # default 50
    # args.evaluate_episodes = 5 # default 10
    # args.save_rate = 5 # default 500
    
    # need to try: 8 15 20 30 50
    # args.n_agents = 8 # default 30 
    # args.render=False

    # env=None
    env, args = make_env(args)

    GRL_task = Task_GRL(args, env)
    # PPO_task.train()
    GRL_task.evaluate_model()

    # TODO: 这里应该是要把agent拆开，按照每个agent获取到更大视野信息后去计算，
    # 但是这样没有集中式critic，可能效果不好，先只改Q-Net部分