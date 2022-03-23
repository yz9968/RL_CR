import argparse

def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # device setting
    parser.add_argument('--gpu', default=True, action='store_true')
    # Environment
    parser.add_argument("--n-agents", type=int, default=30, help="number of agents")
    parser.add_argument("--scenario-name", type=str, default="cr_maddpg", help="name of the scenario script")
    # parser.add_argument("--scenario-name", type=str, default="collision_avoidance", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=200, help="maximum time steps in an episode")
    parser.add_argument("--num-episodes", type=int, default=5001, help="number of train episodes")
    parser.add_argument("--time-steps", type=int, default=2000000, help="number of time steps")
    # Core training parameters
    parser.add_argument("--lr-actor", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--hidden-dim", type=int, default=128, help="hidden layers dimension")
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--epsilon", type=float, default=0.85, help="epsilon greedy")
    parser.add_argument("--epsilon-decay", type=float, default=1, help="epsilon greedy decay")
    parser.add_argument("--noise-rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--buffer-size", type=int, default=int(5e5), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=128, help="number of episodes to optimize at the same time")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=200, help="save model once every time this many timesteps are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")

    # Evaluate
    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=200, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=True, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=50, help="how often to evaluate model")
    args = parser.parse_args()

    return args
