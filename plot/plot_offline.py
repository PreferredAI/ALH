import os
import numpy as np
import matplotlib.pyplot as plt

envs = [
    "halfcheetah-random-v2",
    "halfcheetah-medium-v2",
    "halfcheetah-expert-v2",
    "halfcheetah-medium-expert-v2",
    "halfcheetah-medium-replay-v2",
    "hopper-random-v2",
    "hopper-medium-v2",
    "hopper-expert-v2",
    "hopper-medium-expert-v2",
    "hopper-medium-replay-v2",
    "walker2d-random-v2",
    "walker2d-medium-v2",
    "walker2d-expert-v2",
    "walker2d-medium-expert-v2",
    "walker2d-medium-replay-v2",
]

policies = [
    "TD3_BC",
    "BC",
    "memTD3",
    "memTD3_no_bc",
]

from policy_map_offline import policy_map

result_files = []


def load_data_offline(env, policy, seeds):
    all_data = []
    min_length = float('inf')

    for seed in seeds:
        files = [f"../results/{policy}_{env}_{seed}.npy", ]
        _datas = [(a, np.load(a)) for a in files if os.path.exists(a)]
        if _datas:
            l = [len(a[1]) for a in _datas]
            most_complete_run = np.argmax(l)
            data = _datas[most_complete_run][1]
            result_files.append(_datas[most_complete_run][0])
            all_data.append(data)
            min_length = min(min_length, len(data))
    all_data = [data[:min_length] for data in all_data]
    if min_length == float('inf'):
        return None, min_length
    return np.array(all_data), min_length


def plot_results(env, policies, seeds, subplot_index):
    ax = plt.subplot(3, (len(envs) + 1) // 3, subplot_index)
    ax.set_ylim(-5, 120)
    all_datas = []
    max_plot = 0

    for policy in policies:
        all_data, min_length = load_data_offline(env, policy, seeds)
        max_plot = max(max_plot, min_length)
        all_datas.append((all_data, min_length))
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

    plt.title(env.replace('-v2', ''))

    if subplot_index > (len(envs) - (len(envs) + 1) // 3):
        plt.xlabel(f"Time Steps ({(max_plot - 1) * 5000/1000000:.0f}e6))")
    if (subplot_index - 1) % ((len(envs) + 1) // 3) == 0:
        plt.ylabel(f"Normalized Score")
    plt.grid(True)


if __name__ == '__main__':
    import sys

    original_stdout = sys.stdout
    output_file_path = './offline_1m.tex'

    try:
        sys.stdout = open(output_file_path, 'w')
    except:
        sys.stdout = original_stdout

    seeds = range(5)

    # Create subplots without sharing y-axis
    fig, axes = plt.subplots(3, (len(envs) + 1) // 3, figsize=(14, 7), sharex=True)

    for i, env in enumerate(envs):
        plot_results(env, policies, seeds, i + 1)

    # Create a unique legend for policies
    fig.legend(
        handles=[plt.Line2D([0], [0], color=policy_map[p]['color'], label=policy_map[p]['label']) for p in policies],
        loc='lower center', bbox_to_anchor=(0.5, 0), fancybox=True, shadow=True, ncol=len(policies))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to accommodate the suptitle
    plt.savefig('./Train_all_environments_1M_offline.pdf', bbox_inches='tight')
    # plt.show()

    policies = [
        "memTD3",
        "memTD3_no_bc",
        "TD3_BC",
        "BC",
        "DT",
        # "CQL",
        "APE-V"
    ]

    with open('./envs.txt') as f:
        envs = f.read().split()

    # Define environments and policies
    from_paper_data = {
        # 'CQL': {
        #     'halfcheetah-random-v2': 35.4,
        #     'hopper-random-v2': 10.8,
        #     'walker2d-random-v2': 7.0,
        #     'halfcheetah-medium-v2': 44.4,
        #     'hopper-medium-v2': 58,
        #     'walker2d-medium-v2': 79.2,
        #     'halfcheetah-expert-v2': 104.8,
        #     'hopper-expert-v2': 109.9,
        #     'walker2d-expert-v2': 153.9,
        #     'halfcheetah-medium-expert-v2': 62.4,
        #     'hopper-medium-expert-v2': 111,
        #     'walker2d-medium-expert-v2': 98.7,
        #     'halfcheetah-medium-replay-v2': 46.2,
        #     'hopper-medium-replay-v2': 48.6,
        #     'walker2d-medium-replay-v2': 26.7,
        # },
        'DT': {
            'halfcheetah-medium-v2': 42.6,
            'hopper-medium-v2': 67.6,
            'walker2d-medium-v2': 74.0,
            'halfcheetah-medium-expert-v2': 86.8,
            'hopper-medium-expert-v2': 107.6,
            'walker2d-medium-expert-v2': 108.1,
            'halfcheetah-medium-replay-v2': 36.6,
            'hopper-medium-replay-v2': 82.7,
            'walker2d-medium-replay-v2': 66.6,
        },
        'APE-V': {
            'halfcheetah-random-v2': 29.9,
            'halfcheetah-medium-v2': 69.1,
            'halfcheetah-medium-expert-v2': 101.4,
            'halfcheetah-medium-replay-v2': 64.6,
            'hopper-random-v2': 31.3,
            'hopper-medium-expert-v2': 105.72,
            'hopper-medium-replay-v2': 98.5,
            'walker2d-random-v2': 15.5,
            'walker2d-medium-v2': 90.3,
            'walker2d-medium-expert-v2': 110.0,
            'walker2d-medium-replay-v2': 82.9
        },
    }


    def to_column(name):
        return r"\multicolumn{3}{c}{\textbf{" + name + r"}}"


    def get_mean_std(env, policy, seeds):
        all_data, min_length = load_data_offline(env, policy, seeds)
        if all_data is None:
            if policy in from_paper_data:
                if env in from_paper_data[policy]:
                    return str(from_paper_data[policy][env])
                return "N/A"

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
    seeds = range(5)

    # Print LaTeX table header
    print("\\begin{table*}[!ht]")
    print("\\centering")
    print(r"\adjustbox{max width=0.8\textwidth}{")
    print("\\begin{tabular}{l" + r"r@{}c@{}l" * len(policies) + "}")
    print("\\hline")
    print("\\textbf{Environment} & " + " & ".join(
        [to_column(policy_map[policy]['label']) for policy in policies]) + " \\\\")
    # print("Environment & " + " & ".join(policies) + " \\\\")
    print("\\hline")

    MARIN = 2

    total_all = []
    # Print LaTeX table body
    for _i, env in enumerate(envs):
        row_data = [get_mean_std(env, policy, seeds) for policy in policies]
        means = [float(data.split(' ')[0]) if data != 'N/A' else float('-inf') for data in row_data]
        total_all.append([k if k != float('-inf') else np.nan for k in means])
        indices = np.argsort(means)
        max_index = indices[-1]
        _max_index = indices[-2]
        _to_print = []
        for i in range(len(means)):
            _a = row_data[i].split()
            if means[max_index] - MARIN <= means[i] <= means[max_index] + MARIN:
                if len(_a) > 1:
                    _to_print.extend([f'\\textbf{{{__a}}}' for __a in _a])
                else:
                    _to_print.append(r'\multicolumn{3}{c}{' + f'\\textbf{{{row_data[i]}}}' + '}')
            else:
                if len(_a) > 1:
                    _to_print.extend(_a)
                else:
                    _to_print.append(r'\multicolumn{3}{c}{' + row_data[i] + '}')

                # row_data[i] = r'\multicolumn{3}{c}{'+f'\\textbf{{{row_data[i]}}}'+'}'
            # else:
            #     to_print.append(f'{k:.2f}')
            # if means[i] == means[max_index] or means[i] == means[_max_index]:
            #     row_data[i] = f"\\textbf{{{row_data[i]}}}"
        _max_index = indices[-2]

        print((f"{env} & " + " & ".join(_to_print) + " \\\\").replace("N/A", "-"))
        if (_i + 1) % 3 == 0:
            print("\\hline")

    all = np.mean(total_all, axis=0)
    all_no_nan = np.copy(all)
    all_no_nan[np.isnan(all_no_nan)] = -np.inf
    indices = np.argsort(all_no_nan)

    to_print = []
    for i, k in enumerate(all):
        if i in indices[-2:]:
            to_print.append(r'\multicolumn{3}{c}{' + f'\\textbf{{{k:.2f}}}' + '}')
        elif k == k:
            to_print.append(r'\multicolumn{3}{c}{' + f'{k:.2f}' + '}')
        else:
            to_print.append(r'\multicolumn{3}{c}{' + f'-' + '}')
    all = " & ".join(to_print)
    print((f"Avg All & " + all + " \\\\").replace("N/A", "-"))

    all = np.nanmean(total_all, axis=0)
    all_no_nan = np.copy(all)
    all_no_nan[np.isnan(all_no_nan)] = -np.inf
    indices = np.argsort(all_no_nan)
    to_print = []
    for i, k in enumerate(all):
        if i in indices[-2:]:
            if all[indices[-1]] - MARIN < k <= all[indices[-1]] + MARIN:
                to_print.append(r'\multicolumn{3}{c}{' + f'\\textbf{{{k:.2f}}}' + '}')
            else:
                to_print.append(r'\multicolumn{3}{c}{' + f'{k:.2f}' + '}')
        elif k == k:
            to_print.append(r'\multicolumn{3}{c}{' + f'{k:.2f}' + '}')
        else:
            to_print.append(r'\multicolumn{3}{c}{' + f'-' + '}')
    valid = " & ".join(to_print)
    print((f"Avg Reported & " + valid + " \\\\").replace("N/A", "-"))

    # Print LaTeX table footer
    print("\\hline")
    print("\\end{tabular}}")
    print(
        "\\caption{Average return of the best performed policy. Maximal values of each row are bolded. "
        "$\pm$ corresponds to a single standard deviation over evaluation seeds.}")
    print("\\label{tab:d4rl-bench}")
    print("\\end{table*}")
    sys.stdout = original_stdout
