import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
from gymnasium import spaces
from matplotlib.patches import Rectangle


class OneHotObservationWrapper(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)

        # Dynamically create one-hot dimensions for each part of the observation space
        self.one_hot_dims = {}

        # Iterate over each key in the original observation space
        for key, space in env.observation_space.spaces.items():
            if isinstance(space, spaces.Box) and space.dtype == np.int32:
                # For Box spaces, we assume the space is a discrete range that can be one-hot encoded
                self.one_hot_dims[key] = int(space.high[0]) + 1

        # Redefine the observation space to reflect one-hot encoded spaces
        new_spaces = {
            key: spaces.Box(low=0, high=1, shape=(dim,), dtype=np.float32)
            for key, dim in self.one_hot_dims.items()
        }

        self.observation_space = spaces.Dict(new_spaces)

    def observation(self, obs):
        # One-hot encode each component of the observation based on the derived dimensions
        one_hot_obs = {}
        for key, value in obs.items():
            if key in self.one_hot_dims:
                one_hot_vector = np.zeros(
                    self.one_hot_dims[key], dtype=np.float32)
                one_hot_vector[value[0]] = 1.0
                one_hot_obs[key] = one_hot_vector

        return one_hot_obs


Any = object()


class CustomBlackjackWrapper(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)

        # dim = env.observation_space.shape[0] + 1
        self.observation_space = spaces.Box(
            low=np.concatenate((self.observation_space.low, [0, 0])),
            high=np.concatenate((self.observation_space.high, [1, 1])), dtype=env.observation_space.dtype)
        self._info = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ):
        self._info = None
        return gym.ObservationWrapper.reset(self, seed=seed, options=options)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(
            action)
        self._info = info
        modified_observation = self.observation(observation)
        return modified_observation, reward, terminated, truncated, info

    def observation(self, observation):
        if self._info and self._info.get('double', False):
            observation = np.append(observation, 1)
        else:
            observation = np.append(observation, 0)

        if self._info and self._info.get('player_has_ace', False):
            observation = np.append(observation, 1)
        else:
            observation = np.append(observation, 0)
        return observation


def PlotColorGrid(data, ax):
    grid = {}

    for x, y, z in data:
        if (x, y) not in grid:
            # [count of z=0, count of z=1, count of z=2]
            grid[(x, y)] = [0, 0, 0]
        grid[(x, y)][z] += 1

    # Clear previous plot
    ax.cla()

    # Add grid lines with spacing of 1
    ax.set_xticks(np.arange(0.5, 11, 1))
    ax.set_yticks(np.arange(0.5, 10, 1))
    ax.grid(True)

    # Rebuild x, y, and color lists based on updated grid
    for (x, y), counts in grid.items():
        total = sum(counts)
        if total > 0:
            # Normalize the counts to get relative frequencies
            r_freq = counts[0] / total  # Frequency of z = 0 (red)
            g_freq = counts[2] / total  # Frequency of z = 2 (green)
            b_freq = counts[1] / total  # Frequency of z = 1 (blue)

            # Create a color based on frequencies
            color = (r_freq, g_freq, b_freq)

            # Add a colored square (rectangle) at position (x, y)
            rect = Rectangle((x - 0.5, y - 0.5), 1, 1,
                             facecolor=color, edgecolor='black')
            ax.add_patch(rect)

    # Set axis limits and labels
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)
    ax.set_xlabel('Dealer\'s Card')
    ax.set_ylabel('Player\'s Hand')
    ax.set_title('Blackjack Strategy with Frequency-based Colors')

    # Set custom tick labels
    ax.set_xticklabels([2, 3, 4, 5, 6, 7, 8, 9, 10, 1])
    ax.set_yticklabels([17, 16, 15, 14, 13, 12, 11, 10, 9, 8])

    # Update the plot without blocking
    plt.draw()
    plt.pause(0.01)
