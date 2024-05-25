import os
import numpy as np
import matplotlib.pyplot as plt

# Define environments and policies
envs = [
    "HalfCheetah-v3",
]


policies = [
    "memTD32_ab4",
    "memTD32_ab2",
    "memTD32_ab3",
    "memTD32",
    "TD3"
]

result_files = []

from policy_map import policy_map
def load_ab_data(env, policy, seeds):
    all_data = []
    min_length = float('inf')

    for seed in seeds:
        files = [f"../results/{policy}_{env}_{seed}.npy"]
        _datas = [(a, np.load(a)) for a in files if os.path.exists(a)]
        if _datas:
            l = [len(a[1]) for a in _datas]
            most_complete_run = np.argmax(l)
            data = _datas[most_complete_run][1]
            result_files.append(_datas[most_complete_run][0])
            all_data.append(data)
            min_length = min(min_length, len(data))
    all_data = [data[:min_length] for data in all_data]
    return np.array(all_data), min_length


def plot_results(env, policies, seeds, subplot_index):
    plt.subplot(1, 2, subplot_index)  # 2x4 grid for 8 subplots
    all_datas = []
    max_plot = 0

    for policy in policies:
        all_data, min_length = load_ab_data(env, policy, seeds)
        if min_length < 10e9:
            # min_length = min(min_length, 201)
            max_plot = max(max_plot, min_length)
            all_datas.append((all_data, min_length))
    _min_run = min(*[d[0].shape[0] for d in all_datas])
    for ((all_data, min_length), policy) in zip(all_datas, policies):
        if len(all_data) > 0:  # Check if any valid runs were found
            mean_data = np.mean(all_data[:_min_run, :min_length], axis=0)
            std_data = np.std(all_data[:_min_run, :min_length], axis=0)
            smooth_window = 10
            smooth_mean_data = np.convolve(mean_data, np.ones(smooth_window) / smooth_window, mode='valid')
            smooth_std = np.convolve(std_data, np.ones(smooth_window) / smooth_window, mode='valid')
            step = np.linspace(0, min_length / max_plot, len(smooth_mean_data))

            plt.plot(step, smooth_mean_data, label=policy_map[f"{policy}"]['label'],
                     color=policy_map[f"{policy}"]['color'])
            plt.fill_between(step,
                             (smooth_mean_data - smooth_std).flatten(),
                             (smooth_mean_data + smooth_std).flatten(),
                             alpha=0.1, color=policy_map[f"{policy}"]['color'])
    if 'memTD32' in policies:
        plt.title("Ablation on ALH-a")
    else:
        plt.title("Ablation on ALH-g")

    # if subplot_index>4:
    plt.xlabel(f"Time Steps ({(max_plot - 1) * 5000/1000000:.0f}e6)")
    if subplot_index==1 or subplot_index==5:
        plt.ylabel(f"Average Return")
    # plt.ylabel(f"Mean Reward over {_min_run} seeds")
    plt.grid(True)


# List of seeds
seeds = range(10)

fig, axes = plt.subplots(1, 2, figsize=(6.5, 4), sharex=True, sharey=True)

for i, env in enumerate(envs):
    plot_results(env, policies, seeds, i + 1)



policies = [
    "memTD3_ab4",
    "memTD3_ab2",
    "memTD3_ab3",
    "memTD3",
    "TD3"
]

# List of seeds
seeds = range(10)


for i, env in enumerate(envs):
    plot_results(env, policies, seeds, 2)

policies = [
    "memTD3_ab4",
    "memTD3_ab2",
    "TD3",
    "memTD3_ab3",
    "memTD32",
    "memTD3",
]
# Create a unique legend for policies
fig.legend(handles=[plt.Line2D([0], [0],
                               color=policy_map[p]['color'],
                               label=policy_map[p]['label']) for p in policies],
           loc='lower center', bbox_to_anchor=(0.5, -0.07), fancybox=True, shadow=True,
           ncol=len(policies)//2)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to accommodate the suptitle
plt.savefig('./Online_ablation_1M.pdf', bbox_inches='tight')
plt.show()
