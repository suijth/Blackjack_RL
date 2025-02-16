import gymnasium as gym

from stable_baselines3 import DQN


def eval(model, env: gym.Env, n_episodes):
    obs, _ = env.reset()
    avg_reward = 0.0
    g = 0.0
    n = n_episodes
    while n_episodes:
        action, _states = model.predict(obs, deterministic=True)
        # print(f"actoin:{action}")
        obs, reward, terminated, truncated, info = env.step(action)
        if reward == 1 or reward == -1:
            g += reward
        # env.render()
        if terminated or truncated:
            avg_reward += g
            g = 0
            obs, info = env.reset()
            n_episodes -= 1
            # print(f"terminated:{n_episodes}")
    return avg_reward/n


if __name__ == "__main__":
    path = './logs/twodeck/best_model.zip'

    model = DQN()
    model.load(path=path)
