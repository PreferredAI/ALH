import numpy as np
import matplotlib.pyplot as plt

from policy_map_offline import policy_map
from plot_offline import load_data_offline


def plot_results(env, policies, seeds, subplot_index):
    ax = plt.subplot(2, len(envs), subplot_index)
    ax.set_ylim(-5, 120)
    all_datas = []
    max_plot = 0

    for policy in policies:
        all_data, min_length = load_data_offline(env, policy, seeds)
        if all_data is not None:
            max_plot = max(max_plot, min_length)
            all_datas.append((all_data, min_length))
    if all_datas:
        _min_run = min(*[d[0].shape[0] for d in all_datas])
    for ((all_data, min_length), policy) in zip(all_datas, policies):
        # print(all_data.shape, policy,  min_length)
        if len(all_data) > 0:  # Check if any valid runs were found
            mean_data = np.mean(all_data[:_min_run, :min_length], axis=0)
            std_data = np.std(all_data[:_min_run, :min_length], axis=0)
            smooth_window = 20
            smooth_mean_data = np.convolve(mean_data, np.ones(smooth_window) / smooth_window, mode='valid')
            smooth_std = np.convolve(std_data, np.ones(smooth_window) / smooth_window, mode='valid')
            step = np.linspace(0, min_length / max_plot, len(smooth_mean_data))

            plt.plot(step, smooth_mean_data, label=policy_map[f"{policy}"]['label'],
                     color=policy_map[f"{policy}"]['color'])
            plt.fill_between(step,
                             (smooth_mean_data - smooth_std).flatten(),
                             (smooth_mean_data + smooth_std).flatten(),
                             alpha=0.1, color=policy_map[f"{policy}"]['color'])
    if subplot_index <= len(envs):
        plt.title(env.replace('-v2', ''))

    if subplot_index > (len(envs) - (len(envs) + 1) // 1):
        plt.xlabel(f"Time Steps ({(max_plot - 1) * 5000/1000000:.0f}e6)")
    if subplot_index == 1:
        plt.ylabel(f"Normalized Score (ALH+BC)")
    if subplot_index == 6:
        plt.ylabel(f"Normalized Score (ALH)")
    plt.grid(True)


if __name__ == '__main__':
    # List of seeds
    seeds = range(5)
    envs = (
        "halfcheetah-random-v2",
        "halfcheetah-medium-v2",
        "halfcheetah-expert-v2",
        "halfcheetah-medium-expert-v2",
        "halfcheetah-medium-replay-v2"
    )

    policies = [
        "memTD3",
        "memTD3_ab4",
        "memTD3_ab2",
        "memTD3_ab3",
    ]

    fig, axes = plt.subplots(2, len(envs) // 1, figsize=(13, 5), sharex=True)
    for i, env in enumerate(envs):
        plot_results(env, policies, seeds, i + 1)

    policies = [
        "memTD3_no_bc",
        "memTD3_ab4_no_bc",
        "memTD3_ab2_no_bc",
        "memTD3_ab3_no_bc",
    ]

    for i, env in enumerate(envs):
        plot_results(env, policies, seeds, len(envs) + i + 1)

    legends = [
        "memTD3",
        "memTD3_no_bc",
        "memTD3_ab4_no_bc",
        "memTD3_ab2_no_bc",
        "memTD3_ab3_no_bc",
    ]

    fig.legend(handles=[plt.Line2D([0], [0], color=policy_map[p]['color'], label=policy_map[p]['label'])
                        for p in legends],
               loc='lower center',
               bbox_to_anchor=(0.5, -0.02),
               fancybox=True, shadow=True, ncol=len(legends))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to accommodate the suptitle
    plt.savefig('./Offline_ablation_1M.pdf', bbox_inches='tight')
    plt.show()
