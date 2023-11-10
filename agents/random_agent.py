import argparse
import os
import random
import time
import gymnasium as gym
import dengue_envs
import numpy as np
import tqdm


GAMMA = 0.9
ALPHA = 0.2
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
                        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="the K epochs to update the policy")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on

    return args

def make_env(env_id, seed, idx, capture_video, run_name, params):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, **params)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, **params)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk
def best_value_and_action(state):
    best_v = -np.inf
    best_a = None
    state = tuple(state.items())
    for action in range(4):
        if (state, action) not in values:
            values[(state, action)] = 0
        if values[(state, action)] > best_v:
            best_v = values[(state, action)]
            best_a = action
    return best_v, best_a
def value_update(values, state, action, reward, next_state):
    best_v, _ = best_value_and_action(next_state)
    new_v = reward + GAMMA * best_v
    if isinstance(state, tuple):
        state = state[0]
    old_v = values[(tuple(state.items()), action)]
    values[(tuple(state.items()), action)] = old_v * (1 - ALPHA) + new_v * ALPHA

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # env setup
    params = {
        "size": 400,
    "episize":  150,
    "epilength":  60,
    "dengue_center": (100, 100),
    "chik_center": (300, 300),
    "dengue_radius": 90,
    "chik_radius": 90,
    "clinical_specificity": 0.8,
    "render_mode": None,
    }
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, params) for i in range(args.num_envs)]
    )

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    obs, info = envs.reset()
    next_done = args.num_envs * [False]
    num_updates = args.total_timesteps // args.batch_size
    action  = envs.action_space.sample()
    values = {(tuple(obs.items()), action): 0}
    for step in tqdm.tqdm(range(0, args.num_steps)):
        next_obs, reward, done,_, info = envs.step(action)
        value_update(values, obs, action, reward, next_obs)
        obs = next_obs
        action = envs.action_space.sample() # random action
        values[(tuple(obs.items()), action)] = 0
