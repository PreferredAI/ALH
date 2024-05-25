import argparse
import os

import d4rl
import gym
import numpy as np
import torch

import TD3_BC
import memTD3
import BC
import utils
from tqdm.auto import tqdm
"""
Rely on implementation of https://github.com/sfujim/TD3_BC [TD3_BC paper]
"""

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--logdir", default="runs", type=str)
    parser.add_argument("--env", default="hopper-medium-v0")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--seeds", default=10, type=int)
    parser.add_argument("--eval_freq", default=5e3, type=int)
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--no_bc", action="store_true")
    parser.add_argument("--hypo_dim", default=64, type=int)
    parser.add_argument("--expl_noise", default=0.1)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--mini_batch_size", default=None, type=int)
    parser.add_argument("--discount", default=0.99)
    parser.add_argument("--tau", default=0.005)
    parser.add_argument("--policy_noise", default=0.2)
    parser.add_argument("--noise_clip", default=0.5)
    parser.add_argument("--policy_freq", default=2, type=int)
    parser.add_argument("--alpha", default=2.5)
    parser.add_argument("--normalize", default=True)
    parser.add_argument("--device", default=None, type=str)
    args = parser.parse_args()
    device = args.device
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    policy = args.policy
    if args.no_bc:
        policy += f'_no_bc'
    output_file_name = policy
    file_name = f"{policy}_{args.env}_offline"
    output_file_name = f"{output_file_name}_{args.env}_offline"
    return args, policy, file_name, device, output_file_name


if __name__ == "__main__":
    args, policy, file_name, device, _ = parse()
    print("---------------------------------------")
    print(f"Train: Policy: {policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("../results/"):
        os.makedirs("../results/")

    if args.save_model and not os.path.exists("../models"):
        os.makedirs("../models")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    _ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
    LOG_DIR = os.path.join(_ROOT_DIR, f"../../{args.logdir}")

    kwargs = {"state_dim": state_dim, "action_dim": action_dim,
              "max_action": max_action, "discount": args.discount,
              "tau": args.tau,  # TD3
              "device": device, "policy_noise": args.policy_noise * max_action,
              "noise_clip": args.noise_clip * max_action, "policy_freq": args.policy_freq,  # TD3 + BC
              "alpha": args.alpha}
    if 'memTD3' in args.policy:
        kwargs['mini_batch_size'] = args.mini_batch_size
        kwargs['early_stop_mem'] = args.early_stop_mem
        kwargs['no_bc'] = args.no_bc
        kwargs['hypo_dim'] = args.hypo_dim
    policies = {
        'TD3_BC': TD3_BC.TD3_BC,
        'BC': BC.BC,
        'memTD3': memTD3.memTD3,
    }
    assert args.policy in policies, f"Not found policy: {args.policy}!"
    policy = policies[args.policy](**kwargs)


    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device=device)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    if args.normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0, 1
    if args.save_model:
        policy.save(filename=f'../models/{file_name}_0')

    for t in tqdm(range(int(args.max_timesteps)), desc=f'Training {file_name}...'):
        policy.train(replay_buffer, args.batch_size)
        if (t + 1) % args.eval_freq == 0:
            if args.save_model:
                policy.save(filename=f'../models/{file_name}_{str(t + 1)}')
