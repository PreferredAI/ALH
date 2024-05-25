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
from main import parse


def eval_policy(policy, env_name, seed, mean, std, seed_offset=100,
                eval_episodes=10, args=None) -> list:
    if hasattr(policy, 'forget'):
        policy.forget()

    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        state = (np.array(state).reshape(1, -1) - mean) / std
        while not done:
            p_state = np.array(state)
            action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action)
            state = (np.array(state).reshape(1, -1) - mean) / std
            avg_reward += reward
            if hasattr(policy, 'watch'):
                # adaptive rollout
                policy.watch(p_state, action, reward)

    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100
    return d4rl_score


if __name__ == "__main__":
    args, policy, file_name, device, output_file_name = parse()
    output_file_name = output_file_name.replace('_offline', f'_{args.seed}')
    print("---------------------------------------")
    print(f"Evaluate: Policy: {policy}, Env: {args.env}, Seed: {args.seed}, Device: {device}")
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

    policy.load(filename=f'../models/{file_name}_0')
    with torch.no_grad():
        Rs = eval_policy(policy, args.env, args.seed, mean, std, args=args)
        evaluations = [Rs]
        for t in tqdm(range(int(args.max_timesteps)), desc=f'Evaluating {output_file_name}...'):
            if (t + 1) % args.eval_freq == 0:
                policy.load(filename=f'../models/{file_name}_{str(t + 1)}')
                Rs = eval_policy(policy, args.env, args.seed, mean, std, args=args)
                evaluations.append(Rs)
                np.save(f"../results//{output_file_name}", evaluations)
