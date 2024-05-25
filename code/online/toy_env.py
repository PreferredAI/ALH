from typing import Optional, Union, Tuple

import gym
from gym.core import ObsType, ActType
import numpy as np

"""
Implementation of MultiNormEnv
"""
def gaussian_function(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


class BiasActionSpace(gym.spaces.Box):
    _is_hard = False # is bias scheme or not

    def set_hard(self, is_hard):
        self._is_hard = is_hard

    def sample(self) -> np.ndarray:
        if self._is_hard:
            # if is bias scheme, sampled action is zero
            return np.zeros(self.shape)
        return super().sample()


class MultiNormEnv(gym.Env):
    def __init__(self, dis_gap=60, max_time_step=100):
        n_dis: int = 6
        self._reset_time = 0
        self.n_dis = n_dis
        self.locs = np.arange(n_dis, dtype=np.float32) * 2 * dis_gap
        self.scale = np.array([dis_gap, ] * n_dis).astype(np.float32) / 3
        self.reward_scales = np.array([-3 ** 4, -3 ** 3, 3, -3 ** 2, 3 ** 3, -3 ** 4], dtype=np.float32)
        obs_range = [0, 600]
        self.observation_space = gym.spaces.Box(low=obs_range[0], high=obs_range[1], shape=(1,), dtype=np.float32)
        action_step = (obs_range[1] - obs_range[0]) / max_time_step
        self.action_space = BiasActionSpace(low=-action_step, high=action_step, shape=(1,), dtype=np.float32)

        self.is_hard = False
        self._current_pos = None
        self._max_episode_steps = max_time_step
        self._times = self._max_episode_steps
        self._hard_range = [200, 260]
        self.hard_observation_space = gym.spaces.Box(low=self._hard_range[0], high=self._hard_range[1], shape=(1,),
                                                     dtype=np.float32)

    def seed(self, seed=None):
        super().seed(seed)
        self.hard_observation_space.seed(seed)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

    def _reward_obs(self, obs: Union[float, np.ndarray]):
        return_float = False
        if isinstance(obs, float):
            return_float = True
            obs = np.array([obs, ] * self.n_dis).astype(np.float32)
        else:
            _obs = np.array(obs).reshape(-1).astype(np.float32)
            obs = np.zeros((_obs.shape[0], self.n_dis), dtype=np.float32) + _obs[:, None]
        rw = gaussian_function(obs, self.locs, self.scale) * self.reward_scales
        rw = rw.sum(axis=-1)
        if return_float:
            if isinstance(rw, np.ndarray):
                return float(rw.reshape(-1)[0])
            return float(rw)
        return rw

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        self._times -= 1
        self._current_pos += action
        reward = self._reward_obs(self._current_pos)
        assert (isinstance(reward, np.ndarray) and reward.shape == (1,)) or isinstance(reward, float), reward
        is_out_of_range = np.logical_or(self._current_pos < self.observation_space.low,
                                        self._current_pos > self.observation_space.high)
        return (
            np.array(self._current_pos), reward, self._times <= 0 or any(is_out_of_range), {'is_hard': self.is_hard})

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> Union[
        ObsType, tuple[ObsType, dict]]:

        self._times = self._max_episode_steps
        if options and 'is_hard' in options:
            self.is_hard = options['is_hard']
            self.action_space.set_hard(self.is_hard)
        if self.is_hard:
            # bias scheme
            self._current_pos = self.hard_observation_space.sample()
        else:
            self._current_pos = self.observation_space.sample()
        if return_info:
            return np.array(self._current_pos), {'is_hard': self.is_hard}
        else:
            return np.array(self._current_pos)

    def render(self, mode="human"):
        pass


gym.register(id='MultiNormEnv', entry_point='toy_env:MultiNormEnv', )

if __name__ == '__main__':
    env = MultiNormEnv()
    import numpy as np
    import matplotlib.pyplot as plt

    x_values = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 1000)
    y_values = env._reward_obs(x_values)
    plt.figure(figsize=(9, 6))

    # Plot the reward distribution
    plt.plot(x_values, y_values, linewidth=2, label=f'reward')
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    x_hard = np.linspace(env._hard_range[0], env._hard_range[1], 1000)
    plt.axvspan(env._hard_range[0], env._hard_range[1], alpha=0.3, color='red', label=f'train range')
    action_range = np.linspace(env.action_space.low[0], env.action_space.high[0], 10)
    zero = np.zeros((10,)) + 0.1
    plt.title('MultiNormEnv', fontsize=16)
    plt.xlabel('State', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.legend(fontsize=13)
    plt.grid(True)
    plt.show()
