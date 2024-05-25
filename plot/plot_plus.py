import sys
import os
import numpy as np
import matplotlib.pyplot as plt



from policy_map import policy_map

policy_map = policy_map.update({
    'memDDPG': {
        'label': 'ALH-g + DDPG',
        'color': 'blue',
    },
    'memDDPG_adaptive': {
        'label': 'ALH-a + DDPG',
        'color': 'navy',
    },
})
envs = [
    "HalfCheetah-v3",
    "Hopper-v3",
    "Walker2d-v3",
    "Ant-v3",
    "Humanoid-v3",
    "Reacher-v2",
    "InvertedDoublePendulum-v2",
    "InvertedPendulum-v2"
]

policies = [
    "DDPG",
    "memDDPG",
    "memDDPG_adaptive"
    # "memTD32",
    # "memTD3",
    # "TD3",
    # "MBPO",
    # "DDPG",
    # "PPO",
]

result_files = []


def load_data(env, policy, seeds):
    all_data = []
    min_length = float('inf')

    for seed in seeds:
        files = [f"../results/{policy}_{env}_{seed}.npy"]
        if 'Humanoid' in env:
            files.append(f"../results/{policy}_HumanoidTruncatedObs-v3_{seed}.npy")
        if 'Ant' in env:
            files.append(f"../results/{policy}_AntTruncatedObs-v3_{seed}.npy")
        _datas = [(a, np.load(a)) for a in files if os.path.exists(a)]
        if _datas:
            l = [len(a[1]) for a in _datas]
            most_complete_run = np.argmax(l)
            data = _datas[most_complete_run][1]
            result_files.append(_datas[most_complete_run][0])
            all_data.append(data)
            min_length = min(min_length, len(data))
    min_length = min(min_length, 201)
    all_data = [data[:min_length] for data in all_data]
    return np.array(all_data), min_length


def plot_results(env, policies, seeds, subplot_index):
    plt.subplot(2, 4, subplot_index)  # 2x4 grid for 8 subplots
    all_datas = []
    max_plot = 0

    for policy in policies:
        all_data, min_length = load_data(env, policy, seeds)
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

    plt.title(env)

    if subplot_index > 4:
        plt.xlabel(f"Time Steps ({(max_plot - 1) * 5000 / 1000000:.0f}e6)")
    if subplot_index == 1 or subplot_index == 5:
        plt.ylabel(f"Average Return")
    # plt.ylabel(f"Mean Reward over {_min_run} seeds")
    plt.grid(True)


if __name__ == '__main__':
    # List of seeds
    seeds = range(10)

    # Create subplots without sharing y-axis
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True)
    # fig.suptitle('Training Results for Environments', fontsize=16)

    rev_policies = policies.copy()
    rev_policies.reverse()
    for i, env in enumerate(envs):
        plot_results(env, rev_policies, seeds, i + 1)

    # Create a unique legend for policies
    fig.legend(
        handles=[plt.Line2D([0], [0], color=policy_map[p]['color'], label=policy_map[p]['label']) for p in policies],
        loc='lower center', bbox_to_anchor=(0.5, 0), fancybox=True, shadow=True, ncol=len(policies))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to accommodate the suptitle
    plt.savefig('./Train_all_environments_1M_plus_memDDPG.pdf', bbox_inches='tight')
    # plt.show()

    import os
    import numpy as np


    def get_mean_std(env, policy, seeds):
        all_data, min_length = load_data(env, policy, seeds)

        # Check if the data has the expected structure
        if all_data.size == 0 or all_data.shape[1] == 0:
            return "N/A"

        if len(all_data) > 0:  # Check if any valid runs were found
            # Take mean along the truncated length
            mean_data = np.mean(all_data, axis=0)
            std_data = np.std(all_data, axis=0)
            max_val = np.argmax(mean_data)
            return f"{mean_data[max_val]:.2f} $\\pm$ {std_data[max_val]:.2f}"
        else:
            return "N/A"


    # List of seeds
    seeds = range(10)


    def to_column(name):
        return r"\multicolumn{3}{c}{\textbf{" + name + r"}}"



    original_stdout = sys.stdout
    output_file_path = './online_1m_memDDPG_plus.tex'

    try:
        sys.stdout = open(output_file_path, 'w')
    except:
        sys.stdout = original_stdout
    # Print LaTeX table header
    print("\\begin{table*}[!ht]")
    print("\\centering")
    print(r"\adjustbox{max width=\textwidth}{")
    print("\\begin{tabular}{l" + r"r@{}c@{}l" * len(policies) + "}")
    print("\\hline")
    print("\\textbf{Environment} & " + " & ".join(
        [to_column(policy_map[policy]['label']) for policy in policies]) + " \\\\")
    # print("Environment & " + " & ".join(policies) + " \\\\")
    print("\\hline")
    total_all = []
    # Print LaTeX table body
    for env in envs:
        row_data = [get_mean_std(env, policy, seeds) for policy in policies]
        means = [float(data.split(' ')[0]) if data != 'N/A' else float('-inf') for data in row_data]
        total_all.append([float(data.split(' ')[0]) if data != 'N/A' else float('-inf') for data in row_data])
        indices = np.argsort(means)
        max_index = indices[-1]
        _max_index = indices[-2]
        _to_print = []
        for r in row_data:
            _to_print.extend(r.split())

        for i in range(len(means)):
            if means[i] == means[max_index] or i == _max_index:
                for _i in range(i * 3, (i + 1) * 3):
                    _to_print[_i] = f"\\textbf{{{_to_print[_i]}}}"
                # row_data[i] = f"\\textbf{{{row_data[i]}}}"
        _max_index = indices[-2]

        # row_data[_max_index] = f"\\textbf{{{row_data[_max_index]}}}"
        print(f"{env} & " + " & ".join(_to_print) + " \\\\")

    print("\\hline")
    total_all = np.array(total_all)
    a = total_all - total_all.min(axis=1).reshape(-1, 1)
    b = (total_all.max(axis=1) - total_all.min(axis=1) + 0.03).reshape(-1, 1)
    b = np.repeat(b, a.shape[-1], axis=1)
    total_all = np.divide(a, b) * 100

    all = np.mean(total_all, axis=0)
    all_no_nan = np.copy(all)
    all_no_nan[np.isnan(all_no_nan)] = -np.inf
    indices = np.argsort(all_no_nan)

    to_print = []
    for i, k in enumerate(all):
        _k = r'\multicolumn{3}{*}{' + f'{k:.2f}' + r'}'
        if i in indices[-2:]:
            to_print.append(r'\multicolumn{3}{c}{' + f'\\textbf{{{k:.2f}}}' + '}')
        elif k == k:
            to_print.append(r'\multicolumn{3}{c}{' + f'{k:.2f}' + '}')
        else:
            to_print.append(r'\multicolumn{3}{c}{' + f'-' + '}')
    all = " & ".join(to_print)
    print((f"Avg (normalized) & " + all + " \\\\").replace("N/A", "-"))
    # Print LaTeX table footer
    print("\\hline")
    print("\\end{tabular}}")
    print(
        "\\caption{Average return of the best performed policy. Maximum values of each row are bolded."
        " $\pm$ corresponds to a single standard deviation over trials. "
        "The last row contains average of normalized scores of each algorithm.}")
    print("\\label{tab:gym-bench}")
    print("\\end{table*}")
    sys.stdout = original_stdout

    original_stdout = sys.stdout
    output_file_path = './online_1m_memDDPG_plus_avg.tex'
    def get_mean_std(env, policy, seeds):
        all_data, min_length = load_data(env, policy, seeds)

        # Check if the data has the expected structure
        if all_data.size == 0 or all_data.shape[1] == 0:
            return "N/A"

        if len(all_data) > 0:  # Check if any valid runs were found
            # Take mean along the truncated length
            mean_data = np.mean(all_data, axis=0).mean()
            std_data = np.std(all_data, axis=0).mean()
            return f"{mean_data:.2f} $\\pm$ {std_data:.2f}"
        else:
            return "N/A"


    try:
        sys.stdout = open(output_file_path, 'w')
    except:
        sys.stdout = original_stdout
    print("\\begin{table*}[!ht]")
    print("\\centering")
    print(r"\adjustbox{max width=\textwidth}{")
    print("\\begin{tabular}{l" + r"r@{}c@{}l" * len(policies) + "}")
    print("\\hline")
    print("\\textbf{Environment} & " + " & ".join(
        [to_column(policy_map[policy]['label']) for policy in policies]) + " \\\\")
    # print("Environment & " + " & ".join(policies) + " \\\\")
    print("\\hline")
    total_all = []
    # Print LaTeX table body
    for env in envs:
        row_data = [get_mean_std(env, policy, seeds) for policy in policies]
        means = [float(data.split(' ')[0]) if data != 'N/A' else float('-inf') for data in row_data]
        total_all.append([float(data.split(' ')[0]) if data != 'N/A' else float('-inf') for data in row_data])
        indices = np.argsort(means)
        max_index = indices[-1]
        _max_index = indices[-2]
        _to_print = []
        for r in row_data:
            _to_print.extend(r.split())

        for i in range(len(means)):
            if means[i] == means[max_index] or i == _max_index:
                for _i in range(i * 3, (i + 1) * 3):
                    _to_print[_i] = f"\\textbf{{{_to_print[_i]}}}"
                # row_data[i] = f"\\textbf{{{row_data[i]}}}"
        _max_index = indices[-2]

        # row_data[_max_index] = f"\\textbf{{{row_data[_max_index]}}}"
        print(f"{env} & " + " & ".join(_to_print) + " \\\\")

    print("\\hline")
    total_all = np.array(total_all)
    a = total_all - total_all.min(axis=1).reshape(-1, 1)
    b = (total_all.max(axis=1) - total_all.min(axis=1) + 0.03).reshape(-1, 1)
    b = np.repeat(b, a.shape[-1], axis=1)
    total_all = np.divide(a, b) * 100

    all = np.mean(total_all, axis=0)
    all_no_nan = np.copy(all)
    all_no_nan[np.isnan(all_no_nan)] = -np.inf
    indices = np.argsort(all_no_nan)

    to_print = []
    for i, k in enumerate(all):
        _k = r'\multicolumn{3}{*}{' + f'{k:.2f}' + r'}'
        if i in indices[-2:]:
            to_print.append(r'\multicolumn{3}{c}{' + f'\\textbf{{{k:.2f}}}' + '}')
        elif k == k:
            to_print.append(r'\multicolumn{3}{c}{' + f'{k:.2f}' + '}')
        else:
            to_print.append(r'\multicolumn{3}{c}{' + f'-' + '}')
    all = " & ".join(to_print)
    print((f"Avg (normalized) & " + all + " \\\\").replace("N/A", "-"))
    # Print LaTeX table footer
    print("\\hline")
    print("\\end{tabular}}")
    print(
        "\\caption{Average return of over all training steps. Maximum values of each row are bolded."
        " $\pm$ corresponds to a single standard deviation over trials. "
        "The last row contains average of normalized scores of each algorithm.}")
    print("\\label{tab:gym-bench}")
    print("\\end{table*}")
    sys.stdout = original_stdout