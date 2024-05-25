import sys
import os
import numpy as np

original_stdout = sys.stdout
output_file_path = './ablation.tex'

try:
    sys.stdout = open(output_file_path, 'w')
except:
    sys.stdout = original_stdout

policies = [
    "memTD3",
    "memTD3_ab4",
    "memTD3_ab2",
    "memTD3_ab3",
]


def load_ab_data_online():
    env, seeds = "HalfCheetah-v3", range(10)
    all_data = []
    for policy in policies:
        _all_data = []
        for seed in seeds:
            data = np.load(f"../results/{policy}_{env}_{seed}.npy")
            _all_data.append(data)
        all_data.append(_all_data)
    all_data = np.array(all_data)
    return np.mean(all_data, axis=1)


def load_ab_data_online_a():
    env, seeds = "HalfCheetah-v3", range(10)
    all_data = []
    for policy in policies:
        policy = policy.replace('memTD3', 'memTD32')
        _all_data = []
        for seed in seeds:
            data = np.load(f"../results/{policy}_{env}_{seed}.npy")
            _all_data.append(data)
        all_data.append(_all_data)
    all_data = np.array(all_data)
    return np.mean(all_data, axis=1)


def load_ab_data_offline_bc():
    all_data = []
    envs = (
        "halfcheetah-random-v2",
        "halfcheetah-medium-v2",
        "halfcheetah-expert-v2",
        "halfcheetah-medium-expert-v2",
        "halfcheetah-medium-replay-v2"
    )
    from plot_offline import load_data_offline
    for env in envs:
        _all_data = []
        for policy in policies:
            data, _ = load_data_offline(env, policy, range(5))
            _all_data.append(np.transpose(data))
        all_data.append(_all_data)
    all_data = np.array(all_data)
    result = np.transpose(np.array(all_data), (1, 0, 2, 3))
    return np.mean(result, axis=2).reshape((result.shape[0], -1))


def load_ab_data_offline_no_bc():
    all_data = []
    envs = (
        "halfcheetah-random-v2",
        "halfcheetah-medium-v2",
        "halfcheetah-expert-v2",
        "halfcheetah-medium-expert-v2",
        "halfcheetah-medium-replay-v2"
    )
    from plot_offline import load_data_offline
    for env in envs:
        _all_data = []
        for policy in policies:
            data, _ = load_data_offline(env, policy + '_no_bc', range(5))
            _all_data.append(np.transpose(data))
        all_data.append(_all_data)

    all_data = np.array(all_data)

    # print('-'*20)
    # print(all_data.shape)
    # print('-' * 20)
    result = np.transpose(np.array(all_data), (1, 0, 2, 3))
    return np.mean(result, axis=2).reshape((result.shape[0], -1))


data = {
    'online': load_ab_data_online(),
    'online2': load_ab_data_online_a(),
    'offline': load_ab_data_offline_bc(),
    'offline2': load_ab_data_offline_no_bc()
}
from policy_map import policy_map

tab = {
    'max_return': {k: np.max(v, axis=-1) for k, v in data.items()},
    'mean_return': {k: np.mean(v, axis=-1) for k, v in data.items()}
}

base = {
    'max_return': {k: v[0] for k, v in tab['max_return'].items()},
    'mean_return': {k: v[0] for k, v in tab['mean_return'].items()},
}
print("\\begin{table*}[!ht]")
print("\\centering")
print(r"\adjustbox{max width=0.8\textwidth}{")
print("\\begin{tabular}{ll" + "c" * len(data) + "}")
print("\\hline")
print("\\textbf{Aggregate type} & \\textbf{Ablation Setting} & " + " & ".join(
    [f"\\textbf{{{setting}}}" for setting in ['ALH-g', 'ALH-a', 'ALH+BC (offline)', 'ALH (offline)']]) + " \\\\")
print("\\hline")

_a = ['online', 'online2', 'offline', 'offline2']
aggs = ['max_return', ] * len(policies) + ['mean_return', ] * len(policies)
for i, policy in enumerate(policies + policies):
    agg_type = aggs[i]
    abl_type = policy_map[policy]['label']
    _rs = [tab[agg_type][j][i % len(policies)] for j in _a]
    to_plots = [f"\\textbf{{{_rs[j]:.1f}}}" if _rs[j] == np.max(tab[agg_type][_r]) else f"{_rs[j]:.1f}"
                for j, _r in enumerate(_a)]

    if i % len(policies) != 0:
        for j, _r in enumerate(_a):
            _p = (_rs[j]) / base[agg_type][_r] * 100
            if 'textbf' in to_plots[j]:
                to_plots[j] = f"\\textbf{{{_p:.2f}\%}}"
            else:
                to_plots[j] = f"{_p:.2f}\%"
    else:
        to_plots = [f"\\textbf{{100\%}}" if _rs[j] == np.max(tab[agg_type][_r]) else f"100\%"
                    for j, _r in enumerate(_a)]

    agg_type = 'Max' if agg_type == 'max_return' else 'Average'
    if i % len(policies) == 0:
        agg_type = r"\multirow{" + str(len(policies)) + r"}{*}{" + agg_type + r"}"
    else:
        agg_type = ' '
    if 'ALH' in abl_type:
        abl_type = 'our'
    print(" & ".join([agg_type, abl_type, *to_plots]) + " \\\\")
    if (i + 1) % len(policies) == 0:
        print("\\hline")
print("\\end{tabular}}")

print(
    "\\caption{Ablation study}")
print("\\label{tab:ablation}")
print("\\end{table*}")
