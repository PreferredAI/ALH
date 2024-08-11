# BibTex
```
@InProceedings{pmlr-v235-quang24a,
  title = 	 {Augmenting Decision with Hypothesis in Reinforcement Learning},
  author =       {Quang, Nguyen Minh and Lauw, Hady W.},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {41804--41820},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/quang24a/quang24a.pdf},
  url = 	 {https://proceedings.mlr.press/v235/quang24a.html},
  abstract = 	 {Value-based reinforcement learning is the current State-Of-The-Art due to high sampling efficiency. However, our study shows it suffers from low exploitation in early training period and bias sensitiveness. To address these issues, we propose to augment the decision-making process with hypothesis, a weak form of environment description. Our approach relies on prompting the learning agent with accurate hypotheses, and designing a ready-to-adapt policy through incremental learning. We propose the ALH algorithm, showing detailed analyses on a typical learning scheme and a diverse set of Mujoco benchmarks. Our algorithm produces a significant improvement over value-based learning algorithms and other strong baselines. Our code is available at Github URL.}
}
```

# Setup
(Linux only)
Create `test_env` environment by either:
```bash
cd code && sh setup.sh
```
or:

```bash
cd code
conda create --name test_env --file environment.yml
conda activate test_env
pip install -r requirements.txt
```
then install MuJoCo as [instruction](https://github.com/openai/mujoco-py?tab=readme-ov-file#install-mujoco)

# Online setting

Our ALH is implemented in file [```./code/online/memTD3.py```](./code/online/memTD3.py).
The adaptive rollout and two variants of discovery scheme are described in both files [```./code/online/main.py```](./code/online/main.py) (Mujoco test) and [```./code/online/main_toy.py```](./code/online/main_toy.py) (MultiNormEnv analysis).
For a clear presentation, please refer to our paper.

## Run MultiNormEnv
```bash
conda activate test_env
cd code/online && sh run_experiments_toy.sh [n]
```
where `[n]` is the number of parallel processes.
If the machine does not have gpu `[n]`, the total number of parallel processes is equal to `[n]`.
If the machine has multiple gpus, the total number of parallel processes is equal to `[n] x [number of gpus]`

## Run Mujoco-Gym
```bash
conda activate test_env
cd code/online && sh run_experiments.sh [n]
```
# Offline setting

Our ALH is implemented in file [```./code/offline/memTD3.py```](./code/offline/memTD3.py).
The adaptive rollout is described in file [```./code/offline/evaluate.py```](./code/offline/evaluate.py).

```bash
conda activate test_env
cd code/offline && sh run_experiments.sh [n]
```

# References
To be fairly compared with TD3/TD3+BC, our implementation bases on author implementation of:
```
@inproceedings{fujimoto2018addressing,
    title={Addressing Function Approximation Error in Actor-Critic Methods},
    author={Fujimoto, Scott and Hoof, Herke and Meger, David},
    booktitle={International Conference on Machine Learning},
    pages={1582--1591},
    year={2018}
}
@inproceedings{fujimoto2021minimalist,
    title={A Minimalist Approach to Offline Reinforcement Learning},
    author={Scott Fujimoto and Shixiang Shane Gu},
    booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
    year={2021},
}
```
And compares with implementation in:
```
@inproceedings{janner2019mbpo,
    author = {Michael Janner and Justin Fu and Marvin Zhang and Sergey Levine},
    title = {When to Trust Your Model: Model-Based Policy Optimization},
    booktitle = {Advances in Neural Information Processing Systems},
    year = {2019}
}
@misc{pytorch_minimal_ppo,
    author = {Barhate, Nikhil},
    title = {Minimal PyTorch Implementation of Proximal Policy Optimization},
    year = {2021},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/nikhilbarhate99/PPO-PyTorch}},
}
```


# Figures

Reproducing all experiments can take about several thousands of GPU hour.
If you only want to compare with our results, we publish all result files in [```./results```](./results). 
For visualization (tables, figures) reported in our paper, refer to [```./plot```](./plot).
## Reproduce our figures
We provide code to reproduce our all reported figures in our paper.
Please refer to [this notebook file](plot/plot.ipynb)

## Extra experiments
We provide descriptions for our extra experiments in our appendix.
Please refer to [this notebook file](plot/extra_experiments.ipynb)
