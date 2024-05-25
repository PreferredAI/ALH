import numpy as np
import torch
import gym
import argparse
import os

import tqdm.auto

import utils
import TD3
import DDPG
import memTD3
import toy_env
import datetime
from typing import Union

"""
Monitor agents in MultiNormEnv
"""

def log_the_value(policy: Union[TD3.TD3, memTD3.memTD3], *args, **kwargs):
    # range: -1, 601 (603)
    env = gym.make('MultiNormEnv')
    start, end = env.observation_space.low[0], env.observation_space.high[0]
    states = torch.arange(start - 2, end + 3, step=1, dtype=torch.float32, device=policy.device)
    # print(len(states))
    to_left_action = torch.ones(size=(len(states) - 2,), dtype=torch.float32, device=policy.device) * -1
    to_right_action = torch.ones(size=(len(states) - 2,), dtype=torch.float32, device=policy.device)

    # assert len(to_left_action) == 603, f"len(to_left_action) == {len(to_left_action)}"
    with torch.no_grad():
        _states = torch.concatenate((states[2:], states[:-2]), dim=0).reshape(-1, 1)
        _actions = torch.concatenate((to_left_action, to_right_action), dim=0).reshape(-1, 1)
        if isinstance(policy, TD3.TD3):
            q1, q2 = policy.critic(_states, _actions)
        else:
            q1, q2 = policy.critic(_states, _actions, policy.prev_state)
        q = torch.min(q1, q2)
        q = (q - q.min()) / (q.max() - q.min() + 0.003)
    to_left_values = q[:len(q) // 2]
    to_right_values = q[-len(q) // 2:]
    state_values = (to_left_values + to_right_values) / 2

    return state_values.detach().cpu().numpy()


def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)

    start, end = eval_env.observation_space.low[0], eval_env.observation_space.high[0]
    states = []
    bins = np.linspace(start-10, end+10, 100)

    eval_env.seed(seed + 100)
    options = {}
    last_state = None
    if env_name == 'MultiNormEnv':
        options['is_hard'] = False
        last_state = []

    avg_reward = 0.

    if hasattr(policy, 'forget'):
        policy.forget()

    for _ in range(eval_episodes):
        state, done = eval_env.reset(options=options), False
        while not done:
            states.append(state[0])
            if hasattr(policy, 'select_action_detach'):
                action = policy.select_action_detach(np.array(state))
            else:
                action = policy.select_action(np.array(state))
            p_state = np.array(state)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            if hasattr(policy, 'watch'):
                # adaptive rollout
                policy.watch(p_state, action, reward)
        if last_state is not None:
            last_state.append(state)

    avg_reward /= eval_episodes

    hist, edges = np.histogram(states, bins=bins, density=True)

    # print("---------------------------------------")
    # print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    # print("---------------------------------------")
    return avg_reward, last_state, hist

def run_ddg(args):
    device = args.device
    start_timesteps = args.start_timesteps
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    if args.is_not_hard:
        file_name = f"{args.policy}_not_hard_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("../results/"):
        os.makedirs("../results/")

    if args.save_model and not os.path.exists("../models"):
        os.makedirs("../models")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = gym.make(args.env)
    options = {}
    if args.env == 'MultiNormEnv':
        options['is_hard'] = not args.is_not_hard

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {"state_dim": state_dim, "action_dim": action_dim, "max_action": max_action, "discount": args.discount,
              "tau": args.tau, "device": device, "hidden_dim": args.hidden_dim,
              "hypo_dim": args.hypo_dim}

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)
    elif "memTD3" in args.policy:
        kwargs["mini_batch_size"] = args.mini_batch_size
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = memTD3.memTD3(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"../models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device=device)

    # Evaluate untrained policy
    avg_R, last_state, hist = eval_policy(policy, args.env, args.seed)
    evaluations = [avg_R]
    last_states = [last_state]
    states = [hist]
    state_values = [log_the_value(policy)]
    state, done = env.reset(options=options), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    if args.save_model:
        policy.save(f"../models/{file_name}_0")

    for t in tqdm.auto.tqdm(range(int(args.max_timesteps)), f"Training {file_name}..."):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            action = (policy.select_action(np.array(state)) + np.random.normal(0, max_action * args.expl_noise,
                                                                               size=action_dim)).clip(-max_action,
                                                                                                      max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        if args.policy == 'memTD32':
            # if is ALH-a
            policy.watch(state, action, reward)
        done_bool = float(done)

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            # Reset environment
            state, done = env.reset(options=options), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            if args.policy == 'memTD32':
                # if is ALH-a
                policy.forget()

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            avg_R, last_state, hist = eval_policy(policy, args.env, args.seed)
            evaluations.append(avg_R)
            np.save(f"../results/{file_name}", evaluations)
            if args.env == 'MultiNormEnv':
                last_states.append(last_state)
                np.save(f"../results/{file_name}_last_states", last_states)
                state_values.append(log_the_value(policy))
                np.save(f"../results/{file_name}_state_values", state_values)
                np.save(f"../results/{file_name}_replay_buffer", replay_buffer.state)
                states.append(hist)
                np.save(f"../results/{file_name}_state_histogram", states)
            if args.save_model:
                policy.save(f"../models/{file_name}_{str(t+1)}")
    env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--env", default="MultiNormEnv")
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--hypo_dim", default=64, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start_timesteps", default=25e3, type=int)
    parser.add_argument("--eval_freq", default=5e3, type=int)
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    parser.add_argument("--expl_noise", default=0.1, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--mini_batch_size", default=128, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--policy_noise", default=0.2)
    parser.add_argument("--noise_clip", default=0.5)
    parser.add_argument("--policy_freq", default=2, type=int)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--load_model", default="")
    parser.add_argument("-is_not_hard", action="store_true")
    parser.add_argument("--device", default=None, type=str)
    args = parser.parse_args()

    run_ddg(args)

