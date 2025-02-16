# Standard library imports
import os
import time
import argparse
from typing import Optional, Callable

# Third-party imports
import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt
import wandb
from scipy.stats import entropy
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EventCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import spaces
from wandb.integration.sb3 import WandbCallback

# Local imports
from blackjack import BlackjackEnv
from utils import OneHotObservationWrapper, CustomBlackjackWrapper, PlotColorGrid
from gymnasium.wrappers import FlattenObservation
from state import DeckState

class DeckProbabilitiesWrapper(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Wrapper that adds deck probability information to the observation space."""
    
    def __init__(self, env: gym.Env, deck: DeckState, include_entropy: bool = False):
        """
        Initialize the wrapper.
        
        Args:
            env: The environment to wrap
            deck: The deck state to track
            include_entropy: Whether to include entropy in the observation
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)

        self.deck_state = deck
        dim = env.observation_space.shape[0] + deck.state_dim()
        dim += include_entropy
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(dim,), dtype=np.float32)
        self.include_entropy = include_entropy

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Add deck probabilities to the observation."""
        pbs = self.deck_state.get_probabilities()
        obs = np.append(obs, pbs)
        if self.include_entropy:
            ent = entropy(pbs, base=len(pbs))
            obs = np.append(obs, ent)
        return obs

class SaveModelCallback(EventCallback):
    """Callback to save the best model during training."""
    
    def __init__(self, eval_env: gym.Env, eval_freq: int,
                 best_model_save_path: Optional[str] = None, n_episodes: int = 10):
        super().__init__(verbose=True)
        self.env = eval_env
        self.eval_freq = eval_freq
        self.best_model_save_path = best_model_save_path
        self.n_episodes = n_episodes
        self.best_mean_reward = -100
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            mean_reward = eval_model(self.model, self.env, self.n_episodes)
            if mean_reward > self.best_mean_reward:
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(
                        self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
        return True

def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    """Create a linear learning rate schedule."""
    def schedule(progress: float) -> float:
        return final_value + progress * (initial_value - final_value)
    return schedule

def make_env(num_deck: int) -> gym.Env:
    """Create and configure the Blackjack environment."""
    env: gym.Env = BlackjackEnv(num_decks=num_deck)
    deck_state = DeckState(env.deck)
    env = OneHotObservationWrapper(env)
    env = FlattenObservation(env)
    env = DeckProbabilitiesWrapper(env, deck_state, include_entropy=True)
    env = CustomBlackjackWrapper(env)
    env = Monitor(env)
    return env

def make_vec_env(num_deck: int, num_envs: int = 1):
    """Create vectorized environment."""
    def _make_env():
        return make_env(num_deck)
    env = DummyVecEnv([_make_env for _ in range(num_envs)])
    deck = env.get_attr('deck')[0]
    return env, deck

def eval_model(model, env: gym.Env, n_episodes: int) -> float:
    """Evaluate the model for n_episodes."""
    obs, _ = env.reset()
    avg_reward = 0.0
    episode_reward = 0.0
    remaining_episodes = n_episodes
    
    while remaining_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        if reward in [-1, 1]:
            episode_reward += reward
            
        if terminated or truncated:
            avg_reward += episode_reward
            episode_reward = 0
            obs, _ = env.reset()
            remaining_episodes -= 1
            
    return avg_reward / n_episodes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=False, action='store_true',
                        help='Train and Eval')
    parser.add_argument('--deck_size', type=int, default=1,
                        help="Number of decks to use in the Blackjack game")
    args = parser.parse_args()
    
    num_envs = 100
    env, _ = make_vec_env(args.deck_size, num_envs)

    config = {
        "env_name": f"blackjack_{args.deck_size}",
        "policy_type": "MlpPolicy",
        "total_timesteps": 1000000,
        "numb_env": num_envs,
    }

    if args.train:
        run = wandb.init(
            project="black_jack",
            config=config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )

        policy_kwargs = dict(
            net_arch=[dict(pi=[32, 16, 8], vf=[32, 16, 8])]
        )
        
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            batch_size=1024,
            learning_rate=linear_schedule(0.0004, 0.0001), 
            tensorboard_log=f"runs/{run.id}", 
            device="cpu", 
            policy_kwargs=policy_kwargs
        )
        
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=WandbCallback(
                model_save_path='./logs/twodecks/',
                gradient_save_freq=100,
                model_save_freq=1000,
                verbose=2,
            )
        )
        run.finish()
    else:
        model = PPO.load('./logs/twodecks/model.zip', env=env)

    # Evaluation loop
    env = make_env(args.deck_size)
    obs, _ = env.reset()
    episode_reward = 0
    episode_count = 0
    total_reward = 0

    data = [[], []]
    fig, ax = plt.subplots()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward in [-1, 1]:
            episode_reward += reward
            
        if action > 1:
            data[info['player_has_ace']].append((
                info['player_hand_value'],
                info['dealer_visible_card'], 
                action-2
            ))
            
        if terminated or truncated:
            print(f"Episode return = {episode_reward}")
            episode_count += 1
            total_reward += episode_reward
            episode_reward = 0
            
            if len(env.deck) < 20:
                print(f"Average return = {total_reward / episode_count}")
                episode_count = 0
                total_reward = 0
                
            env.render()
            obs, _ = env.reset()
            PlotColorGrid(data[0], ax)
            time.sleep(1)

if __name__ == "__main__":
    main()
