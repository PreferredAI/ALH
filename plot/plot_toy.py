import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy

sys.path.append('../code/online')
import toy_env

from policy_map import policy_map

original_dict = copy.deepcopy(policy_map._map)
policy_map = policy_map.update({
    'TD3': {
        'label': 'TD3 (bias)',
        'color': 'red',
    },
    'TD3_not_hard': {
        'label': 'TD3 (non bias)',
        'color': 'orange',
    },
    'memTD3': {
        'label': 'ALH-g (bias)',
        'color': 'blue',
    },
    'memTD32': {
        'label': 'ALH-a (bias)',
        'color': 'navy',
    },
    'memTD3_not_hard': {
        'label': 'ALH-g (non bias)',
        'color': 'blue',
    },
    'memTD32_not_hard': {
        'label': 'ALH-a (non bias)',
        'color': 'navy',
    },
})


def load_data(env, policy, seeds):
    all_data = []
    min_length = float('inf')

    for seed in seeds:
        filename = f"../results/{policy}_{env}_{seed}.npy"

        # Check if the file exists before loading
        if os.path.exists(filename):
            data = np.load(filename)
            min_length = min(min_length, len(data))
            all_data.append(data)  # Truncate to the minimum length
    lengths = [len(d) for d in all_data]
    all_data = [data[:min_length] for data in all_data]
    return np.array(all_data), min_length


def plot_results(env, policies, seeds, file=None, show=True, figsize=(7, 4)):
    plt.figure(figsize=figsize)
    all_datas = []
    max_plot = 0
    is_ppo = False
    for policy in policies:
        is_ppo = 'PPO' in policy or is_ppo
        all_data, min_length = load_data(env, policy, seeds)
        max_plot = max(max_plot, min_length)
        all_datas.append((all_data, min_length))
    _min_run = min(*[d[0].shape[0] for d in all_datas])
    for ((all_data, min_length), policy) in zip(all_datas, policies):

        if len(all_data) > 0:  # Check if any valid runs were found
            # Take mean along the truncated length
            mean_data = np.mean(all_data[:_min_run, :min_length], axis=0)
            std_data = np.std(all_data[:_min_run, :min_length], axis=0)
            window_size = 10
            mean_data_smooth = np.convolve(mean_data.reshape(-1), np.ones(window_size) / window_size,
                                           mode='valid').reshape(-1)
            std_data_smooth = np.convolve(std_data.reshape(-1), np.ones(window_size) / window_size,
                                          mode='valid').reshape(-1)

            # Adjust step array for a smoother appearance
            step = np.linspace(0, min_length / max_plot, len(mean_data_smooth))

            # Plot the smoothed data
            plt.plot(step, mean_data_smooth, label=policy_map[f"{policy}"]['label'],
                     color=policy_map[f"{policy}"]['color'])
            plt.fill_between(step, (mean_data_smooth - std_data_smooth), (mean_data_smooth + std_data_smooth),
                             alpha=0.1, color=policy_map[f"{policy}"]['color'])

            # step = np.linspace(0, min_length / max_plot, min_length)  #  # plt.plot(step, mean_data, label=policy_map[f"{policy}"]['label'], color=policy_map[f"{policy}"]['color'])  # plt.fill_between(step,  #                  (mean_data - std_data).flatten(), (mean_data + std_data).flatten(),  #                  alpha=0.1, color=policy_map[f"{policy}"]['color'])

    # plt.title(f"{env}")
    plt.xlabel(f"Time Steps ({(max_plot - 1) * 5000 / 1000000:.0f}e6)")
    plt.ylabel(f"Average Return")
    plt.legend(loc='lower right')
    plt.grid(True)

    if file is not None:
        plt.savefig(file, bbox_inches='tight')
    if show:
        plt.show()


def plot_double_states(env, policies, seeds, file=None, show=True, figsize=(6.5, 4)):
    fig, axes = plt.subplots(1, len(policies), figsize=figsize, sharex=True, sharey=True)
    # plt.figure(figsize=(8, 5))
    all_datas = []
    max_plot = 0

    _env = toy_env.MultiNormEnv()
    for policy in policies:
        all_data, min_length = load_all_state_data(env, policy, seeds)
        max_plot = max(max_plot, min_length)
        all_datas.append((all_data, min_length))
    _min_run = all_datas[0][0].shape[0]
    if len(all_datas) > 1:
        _min_run = min(*[d[0].shape[0] for d in all_datas])

    start, end = _env.observation_space.low[0], _env.observation_space.high[0]
    bins = np.linspace(start - 10, end + 10, 100)
    _, edges = np.histogram(np.linspace(start - 10, end + 10, 1000), bins=bins, density=True)
    y = edges[:-1].reshape(1, -1)
    subplot_index = 1
    for ((all_data, min_length), policy) in zip(all_datas, policies):
        ax = plt.subplot(1, len(policies), subplot_index)
        bsz = all_data.shape[-1]
        step = np.linspace([0, ] * bsz, [min_length / max_plot, ] * bsz, min_length)
        ys = np.repeat(y, min_length, axis=0)
        all_data = (all_data - all_data.min()) / (all_data.max() - all_data.min() + 0.003)
        all_data = np.exp(all_data * 6)
        all_data = (all_data - all_data.min()) / (all_data.max() - all_data.min() + 0.003)
        ax.scatter(step, ys,
                   label=policy_map[f"{policy}"]['label'],
                   color=policy_map[f"{policy}"]['color'],
                   s=all_data * 2)

        scale = 5000
        subplot_index += 1

        plt.xlabel(f"Time Steps ({(max_plot - 1) * scale / 1000000:.0f}e6)")
        if subplot_index == 2:
            plt.ylabel(f"State")
    fig.subplots_adjust(hspace=0.0)
    fig.legend(loc='lower center', ncol=len(policies), bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.grid
    if file is not None:
        fig.savefig(file, bbox_inches='tight')
    if show:
        fig.show()


def load_state_data(env, policy, seeds):
    all_data = []
    min_length = float('inf')

    for seed in seeds:
        filename = f"../results/{policy}_{env}_{seed}_last_states.npy"

        # Check if the file exists before loading
        if os.path.exists(filename):
            data = np.load(filename)
            min_length = min(min_length, len(data))
            all_data.append(data)  # Truncate to the minimum length
    # print(len(all_data))
    all_data = [data[:min_length].reshape(min_length, -1) for data in all_data]
    all_data = np.concatenate(all_data, axis=1)
    return all_data, min_length


def plot_state_results(env, policies, seeds, file=None, show=True,
                       legend_loc="best",
                       visual_all=False,
                       sparse=1, figsize=(7, 4)):
    plt.figure(figsize=figsize)
    all_datas = []
    max_plot = 0

    _env = toy_env.MultiNormEnv()

    # Generate x values
    _y_values = np.linspace(_env.observation_space.low[0], _env.observation_space.high[0], 1000)
    _x_values = _env._reward_obs(_y_values)
    im = plt.imshow(np.flip(_x_values).reshape(-1, 1), aspect='auto', cmap='gray',
                    extent=[-0.05, 0, _y_values.min(), _y_values.max()], alpha=1)
    plt.colorbar(im, label='Immediate Reward')

    for policy in policies:
        all_data, min_length = load_state_data(env, policy, seeds)
        # print(all_data.shape)
        max_plot = max(max_plot, min_length)
        all_datas.append((all_data, min_length))
    _min_run = all_datas[0][0].shape[0]
    if len(all_datas) > 1:
        _min_run = min(*[d[0].shape[0] for d in all_datas])
    space = 0
    for ((all_data, min_length), policy) in zip(all_datas, policies):
        # bins = np.linspace(start-10, end+10, 100)
        bsz = all_data.shape[-1]
        step = np.linspace([0, ] * bsz, [min_length / max_plot, ] * bsz, min_length)

        if visual_all:
            step_in_range = step.reshape(-1)
            state_in_range = all_data.reshape(-1)
        else:
            # don't care the out-of-range last state
            in_range_last_state = np.logical_and(_env.observation_space.low[0] <= all_data,
                                                 all_data <= _env.observation_space.high[0])
            step_in_range = step[in_range_last_state].reshape(-1)
            state_in_range = all_data[in_range_last_state].reshape(-1)
        idx = np.arange(step_in_range.shape[0])
        idx = (idx % sparse == 0)
        step_in_range = step_in_range[idx]
        state_in_range = state_in_range[idx]
        plt.scatter(step_in_range + space, state_in_range,
                    alpha=0.3,
                    label=policy_map[f"{policy}"]['label'],
                    color=policy_map[f"{policy}"]['color'],
                    s=0.1)
        # space += 0.001

    scale = 5000
    plt.xlabel(f"Time Steps ({(max_plot - 1) * scale / 1000000:.0f}e6)")
    plt.ylabel(f"Terminal State")
    # plt.legend(loc="best")
    legend_labels = [policy_map[f"{policy}"]['label'] for policy in policies]  # Your size categories
    legend_handles = [plt.Line2D([0], [0], linestyle='none', marker='o',
                                 markersize=2, color=policy_map[f"{policy}"]['color']) for policy in policies]

    # Create the legend with proxy artists
    plt.legend(legend_handles, legend_labels, loc=legend_loc)
    if file is not None:
        plt.savefig(file, bbox_inches='tight')
    if show:
        plt.show()


def load_replay_buffer(env, policy, seeds):
    all_data = []
    min_length = float('inf')

    for seed in seeds:
        filename = f"../results/{policy}_{env}_{seed}_replay_buffer.npy"

        # Check if the file exists before loading
        if os.path.exists(filename):
            data = np.load(filename).reshape(-1)
            min_length = min(min_length, len(data))
            all_data.append(data)  # Truncate to the minimum length
    if len(all_data) == 0:
        return None, None
    all_data = [data[:min_length].reshape(data.shape[0], -1) for data in all_data]
    all_data = np.concatenate(all_data, axis=1)
    all_data = all_data[all_data != 0]
    return all_data, min_length


def plot_replay_buffer(env, policies, seeds, file=None, show=True, figsize=(7, 4)):
    plt.figure(figsize=figsize)
    all_datas = []
    _env = toy_env.MultiNormEnv()

    for policy in policies:
        all_data, min_length = load_replay_buffer(env, policy, seeds)
        all_datas.append((all_data, min_length))

    start, end = _env.observation_space.low[0], _env.observation_space.high[0]

    for ((all_data, min_length), policy) in zip(all_datas, policies):
        if all_data is not None:
            bins = np.linspace(start - 10, end + 10, 500)

            data = all_data.reshape(-1)
            # is_in_range = _env.observation_space.contains(data)
            is_in_range = np.logical_and(data > _env.observation_space.low[0], data < _env.observation_space.high[0])
            data = data[is_in_range]
            hist, edges = np.histogram(data, bins=bins, density=True)
            plt.plot(edges[:-1], hist, alpha=1,
                     label=policy_map[f"{policy}"]['label'],
                     color=policy_map[f"{policy}"]['color'])
    # plt.yscale('log')

    plt.axvspan(_env._hard_range[0], _env._hard_range[1],
                alpha=0.3, color='red', label=f'train range (bias)')
    plt.xlabel(f"State in replay buffer")
    plt.ylabel(f"Frequency (approximately)")
    plt.legend()
    plt.grid(True)
    if file is not None:
        plt.savefig(file, bbox_inches='tight')
    if show:
        plt.show()


def load_state_values(env, policy, seeds):
    all_data = []
    min_length = float('inf')

    for seed in seeds:
        filename = f"../results/{policy}_{env}_{seed}_state_values.npy"

        # Check if the file exists before loading
        if os.path.exists(filename):
            data = np.load(filename)
            min_length = min(min_length, len(data))
            all_data.append(data)  # Truncate to the minimum length
    if len(all_data) == 0:
        return None, None
    all_data = [data[:min_length] for data in all_data]
    all_data = np.stack(all_data, axis=-1)
    all_data = np.mean(all_data, axis=-1, keepdims=False)

    return all_data, min_length


def plot_state_values(env, policies, seeds, file=None, show=True, figsize=(7, 4)):
    plt.figure(figsize=figsize)
    all_datas = []
    max_plot = 0

    _env = toy_env.MultiNormEnv()
    _x_values = np.linspace(_env.observation_space.low[0], _env.observation_space.high[0], 1000)
    _y_values = _env._reward_obs(_x_values)
    _y_values = (_y_values - _y_values.min()) / (_y_values.max() - _y_values.min())
    # im = plt.imshow(_y_values.reshape(1, -1), aspect='auto', cmap='gray',
    #                 extent=[_x_values.min(), _x_values.max(), 1, 1.05, ], alpha=1)
    # plt.colorbar(im, label='Immediate Reward')
    plt.plot(_x_values, _y_values,
             label='Expected shape',
             color='green', )
    for policy in policies:
        all_data, min_length = load_state_values(env, policy, seeds)
        max_plot = max(max_plot, min_length)
        all_datas.append((all_data, min_length))
    _min_run = all_datas[0][0].shape[0]
    if len(all_datas) > 1:
        _min_run = min(*[d[0].shape[0] for d in all_datas])

    start, end = _env.observation_space.low[0], _env.observation_space.high[0]

    to_sample = 60
    for ((all_data, min_length), policy) in zip(all_datas, policies):
        # y = all_data.mean(axis=0).reshape(-1)
        y = all_data[-1, :].reshape(-1)
        x = np.linspace(start - 1, end + 2, len(y))
        plt.plot(x, y,
                 label=policy_map[f"{policy}"]['label'],
                 color=policy_map[f"{policy}"]['color'], )

    # scale = 5000
    plt.xlabel("State")
    plt.ylabel(f"State value (normalized)")
    plt.legend(loc='lower right')
    plt.grid(True)
    if file is not None:
        plt.savefig(file, bbox_inches='tight')
    if show:
        plt.show()


def load_all_state_data(env, policy, seeds):
    all_data = []
    min_length = float('inf')

    for seed in seeds:
        filename = f"../results/{policy}_{env}_{seed}_state_histogram.npy"

        # Check if the file exists before loading
        if os.path.exists(filename):
            data = np.load(filename)
            # print(data.shape)
            min_length = min(min_length, len(data))
            all_data.append(data)  # Truncate to the minimum length
    # print(len(all_data), min_length)
    all_data = np.array([data[:min_length].reshape(min_length, -1) for data in all_data])
    # print(all_data.shape)
    all_data = np.mean(all_data, axis=0)
    # print(all_data.shape)
    return all_data, min_length


def plot_states(env, policies, seeds, file=None, show=True, figsize=(7, 4)):
    plt.figure(figsize=figsize)
    all_datas = []
    max_plot = 0

    _env = toy_env.MultiNormEnv()

    # Generate x values
    _y_values = np.linspace(_env.observation_space.low[0], _env.observation_space.high[0], 1000)
    _x_values = _env._reward_obs(_y_values)
    im = plt.imshow(np.flip(_x_values).reshape(-1, 1), aspect='auto', cmap='gray',
                    extent=[-0.05, 0, _y_values.min(), _y_values.max()], alpha=1)
    plt.colorbar(im, label='Immediate Reward')
    for policy in policies:
        all_data, min_length = load_all_state_data(env, policy, seeds)
        max_plot = max(max_plot, min_length)
        all_datas.append((all_data, min_length))
    _min_run = all_datas[0][0].shape[0]
    if len(all_datas) > 1:
        _min_run = min(*[d[0].shape[0] for d in all_datas])

    start, end = _env.observation_space.low[0], _env.observation_space.high[0]
    bins = np.linspace(start - 10, end + 10, 100)
    _, edges = np.histogram(np.linspace(start - 10, end + 10, 1000), bins=bins, density=True)
    y = edges[:-1].reshape(1, -1)
    for ((all_data, min_length), policy) in zip(all_datas, policies):
        bsz = all_data.shape[-1]
        step = np.linspace([0, ] * bsz, [min_length / max_plot, ] * bsz, min_length)
        _min, _max = all_data.min(), all_data.max()
        ys = np.repeat(y, min_length, axis=0)
        all_data = (all_data - _min) / (_max - _min)
        all_data = np.exp(all_data * 6)
        all_data = (all_data - all_data.min()) * 2 / (all_data.max() - all_data.min())
        plt.scatter(step, ys,
                    label=policy_map[f"{policy}"]['label'],
                    color=policy_map[f"{policy}"]['color'],
                    s=all_data)

    scale = 5000
    plt.xlabel(f"Time Steps ({(max_plot - 1) * scale / 1000000:.0f}e6)")
    plt.ylabel(f"State")

    if file is not None:
        plt.savefig(file, bbox_inches='tight')
    if show:
        plt.show()


def plot_state_values_over_time(env, policies, seeds, file=None, show=True, figsize=(7, 4)):
    plt.figure(figsize=figsize)
    all_datas = []
    max_plot = 0

    _env = toy_env.MultiNormEnv()

    # Generate x values
    _y_values = np.linspace(_env.observation_space.low[0], _env.observation_space.high[0], 1000)
    _x_values = _env._reward_obs(_y_values)
    im = plt.imshow(np.flip(_x_values).reshape(-1, 1), aspect='auto', cmap='gray',
                    extent=[-0.05, 0, _y_values.min(), _y_values.max()], alpha=1)
    plt.colorbar(im, label='Immediate Reward')
    _min, _max = 1e9, -1e9

    for policy in policies:
        all_data, min_length = load_state_values(env, policy, seeds)
        _min = min(all_data.min(), _min)
        _max = max(all_data.max(), _max)
        max_plot = max(max_plot, min_length)
        all_datas.append((all_data, min_length))
    _min_run = all_datas[0][0].shape[0]
    if len(all_datas) > 1:
        _min_run = min(*[d[0].shape[0] for d in all_datas])

    start, end = _env.observation_space.low[0], _env.observation_space.high[0]
    y = np.arange(-start - 1, end + 2).reshape(1, -1)
    for ((all_data, min_length), policy) in zip(all_datas, policies):
        all_data = all_data.reshape(all_data.shape[0], -1)
        all_data = np.exp(all_data * 10) * 10
        all_data = (all_data - all_data.min()) / (all_data.max() - all_data.min() + 0.003)
        bsz = all_data.shape[-1]
        step = np.linspace([0, ] * bsz, [min_length / max_plot, ] * bsz, min_length)
        ys = np.repeat(y, min_length, axis=0)
        # print(step.shape, ys.shape, all_data.shape)
        plt.scatter(step, ys,
                    label=policy_map[f"{policy}"]['label'],
                    color=policy_map[f"{policy}"]['color'],
                    s=all_data)

    scale = 5000
    plt.xlabel(f"Time Steps ({(max_plot - 1) * scale / 1000000:.0f}e6)")
    plt.ylabel(f"State value (scaled)")
    # plt.legend()
    # plt.grid(True)
    if file is not None:
        plt.savefig(file, bbox_inches='tight')
    if show:
        plt.show()


def plot_double_state_values(env, policies, seeds, file=None, show=True, figsize=(7, 4)):
    fig, axes = plt.subplots(1, len(policies), figsize=figsize, sharex=True, sharey=True)

    _env = toy_env.MultiNormEnv()
    _x_values = np.linspace(_env.observation_space.low[0], _env.observation_space.high[0], 1000)
    _y_values = _env._reward_obs(_x_values)
    _y_values = (_y_values - _y_values.min()) / (_y_values.max() - _y_values.min())
    subplot_index = 1
    for _policies in policies:
        all_datas = []
        max_plot = 0
        ax = plt.subplot(1, len(policies), subplot_index)
        ax.plot(_x_values, _y_values,
                label='Expected shape',
                color='green', )
        for policy in _policies:
            all_data, min_length = load_state_values(env, policy, seeds)
            max_plot = max(max_plot, min_length)
            all_datas.append((all_data, min_length))
        _min_run = all_datas[0][0].shape[0]
        if len(all_datas) > 1:
            _min_run = min(*[d[0].shape[0] for d in all_datas])

        start, end = _env.observation_space.low[0], _env.observation_space.high[0]

        to_sample = 60
        for ((all_data, min_length), policy) in zip(all_datas, _policies):
            # y = all_data.mean(axis=0).reshape(-1)
            y = all_data[-1, :].reshape(-1)
            x = np.linspace(start - 1, end + 2, len(y))
            ax.plot(x, y,
                    label=policy_map[f"{policy}"]['label'],
                    color=policy_map[f"{policy}"]['color'], )

        subplot_index += 1

        plt.xlabel("State")
        if subplot_index == 2:
            plt.ylabel(f"State value (normalized)")
        plt.legend(loc='lower right')
        plt.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if file is not None:
        fig.savefig(file, bbox_inches='tight')
    if show:
        fig.show()


def plot_double_results(env, policies, seeds, file=None, show=True, figsize=(7, 4)):
    fig, axes = plt.subplots(1, len(policies), figsize=figsize, sharex=True, sharey=True)
    subplot_index = 1
    for _policies in policies:
        ax = plt.subplot(1, len(policies), subplot_index)
        all_datas = []
        max_plot = 0
        is_ppo = False
        for policy in _policies:
            is_ppo = 'PPO' in policy or is_ppo
            all_data, min_length = load_data(env, policy, seeds)
            max_plot = max(max_plot, min_length)
            all_datas.append((all_data, min_length))
        _min_run = min(*[d[0].shape[0] for d in all_datas])
        for ((all_data, min_length), policy) in zip(all_datas, _policies):

            if len(all_data) > 0:  # Check if any valid runs were found
                # Take mean along the truncated length
                mean_data = np.mean(all_data[:_min_run, :min_length], axis=0)
                std_data = np.std(all_data[:_min_run, :min_length], axis=0)
                window_size = 10
                mean_data_smooth = np.convolve(mean_data.reshape(-1), np.ones(window_size) / window_size,
                                               mode='valid').reshape(-1)
                std_data_smooth = np.convolve(std_data.reshape(-1), np.ones(window_size) / window_size,
                                              mode='valid').reshape(-1)

                # Adjust step array for a smoother appearance
                step = np.linspace(0, min_length / max_plot, len(mean_data_smooth))

                # Plot the smoothed data
                ax.plot(step, mean_data_smooth, label=policy_map[f"{policy}"]['label'],
                        color=policy_map[f"{policy}"]['color'])
                ax.fill_between(step, (mean_data_smooth - std_data_smooth), (mean_data_smooth + std_data_smooth),
                                alpha=0.1, color=policy_map[f"{policy}"]['color'])

        subplot_index += 1
        plt.xlabel(f"Time Steps ({(max_plot - 1) * 5000 / 1000000:.0f}e6)")
        if subplot_index == 2:
            plt.ylabel(f"Average Return")
        plt.legend(loc='lower right')
        plt.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if file is not None:
        fig.savefig(file, bbox_inches='tight')
    if show:
        fig.show()


if __name__ == '__main__':
    seeds = range(10)
    envs = ['MultiNormEnv']

    plot_state_values(envs[0], ['TD3', 'TD3_not_hard'],
                      seeds, file='double-Q-multi-norm-TD3-compare.pdf')
    plot_double_states(envs[0], ['TD3_not_hard', 'memTD32_not_hard', 'memTD3_not_hard', ], seeds,
                       figsize=(7, 4),
                       file='exploit_multinormenv_not_hard.png')
    plot_double_states(envs[0], ['TD3', 'memTD32', 'memTD3', ], seeds,
                       figsize=(7, 4),
                       file='exploit_multinormenv.png')
    plot_double_states(envs[0], ['TD3_not_hard', 'TD3'], seeds, file='TD3_exploration.png')
    plot_double_states(envs[0], ['memTD3', 'memTD32'], seeds, file='memTD3_exploration.png')
    # plot_state_values_over_time(envs[0], ['memTD3_not_hard'], seeds)
    # plot_state_values_over_time(envs[0], ['memTD32_not_hard'], seeds)
    # plot_state_values_over_time(envs[0], ['TD3_not_hard'], seeds)
    # plot_state_values_over_time(envs[0], ['TD3'], seeds)
    # plot_state_values_over_time(envs[0], ['memTD3'], seeds)
    # plot_state_values_over_time(envs[0], ['memTD32'], seeds)
    plot_double_state_values(envs[0],
                             [['TD3', 'memTD3', 'memTD32'],
                              ['TD3_not_hard', 'memTD3_not_hard', 'memTD32_not_hard']],
                             seeds, file='double-Q-multi-norm.pdf')
    plot_double_results(envs[0],
                        [['TD3', 'memTD3', 'memTD32'],
                         ['TD3_not_hard', 'memTD3_not_hard', 'memTD32_not_hard']],
                        seeds, file='double-curve-multi-norm.pdf')
policy_map = policy_map.update(original_dict)
