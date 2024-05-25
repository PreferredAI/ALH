import os

import gym
import d4rl

envs = (
    "halfcheetah-random-v2",
    "hopper-random-v2",
    "walker2d-random-v2",
    "halfcheetah-medium-v2",
    "hopper-medium-v2",
    "walker2d-medium-v2",
    "halfcheetah-expert-v2",
    "hopper-expert-v2",
    "walker2d-expert-v2",
    "halfcheetah-medium-expert-v2",
    "hopper-medium-expert-v2",
    "walker2d-medium-expert-v2",
    "halfcheetah-medium-replay-v2",
    "hopper-medium-replay-v2",
    "walker2d-medium-replay-v2",
)
for env_ in envs:
    env_ = env_.strip()
    print(env_.strip())
    env = gym.make(env_.strip())
    try:
        dataset = env.get_dataset()
    except:
        os.remove(env.dataset_filepath)
        dataset = env.get_dataset()
    print(f"Downloaded {env_}")
