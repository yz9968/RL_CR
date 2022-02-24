from common.arguments import get_args
from common.utils import make_env
from ppo.PPO_task import Task_PPO

if __name__ =="__main__":
    args=get_args()
    args.scenario_name="cr_ppo"

    # args.num_episodes = 10 # default 5001
    # args.evaluate_rate = 2 # default 50
    # args.evaluate_episodes = 2 # default 10
    # args.save_rate = 1 # default 500
    # args.n_agents = 6 # default 30

    # args.num_episodes = 500 # default 5001
    # args.evaluate_rate = 5 # default 50
    # args.evaluate_episodes = 5 # default 10
    # args.save_rate = 5 # default 500
    
    # need to try: 8 15 20 30 50
    args.n_agents = 8 # default 30 
    args.render=False

    from common.utils import make_env
    # env=None
    env, args = make_env(args)

    PPO_task = Task_PPO(args, env)
    # PPO_task.train()
    PPO_task.evaluate_model()

    # TODO: generate fixed random seed list for scenarios
    # TODO: generate fixed random seed list for scenarios