import gymnasium as gym
import h5py
import numpy as np
from stable_baselines3 import PPO
from typing import Dict, List, Tuple


def collect_episodes(
        env: gym.Env,
        model: PPO,
        num_episodes: int,
        max_steps: int = 1000
) -> Tuple[Dict[str, List[np.ndarray]], List[int]]:
    """
    Collects trajectory data from the environment and organizes it by episodes.

    Args:
        env: Gymnasium environment
        model: Trained PPO model
        num_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode

    Returns:
        episode_data: Dictionary containing episode-wise trajectory data
        episode_returns: List of total returns for each episode
    """
    episode_data = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "returns_to_go": [],
        "timesteps": []
    }
    episode_returns = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_obs = []
        episode_actions = []
        episode_rewards = []
        episode_timesteps = []
        step = 0

        # Run episode until termination or max steps reached
        while not done and step < max_steps:
            action, _states = model.predict(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            # Store step data
            episode_obs.append(obs)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_timesteps.append(step)

            done = terminated or truncated
            obs = next_obs
            step += 1

        # Calculate total episode return
        episode_return = sum(episode_rewards)
        episode_returns.append(episode_return)

        # Calculate returns-to-go for each timestep
        # returns-to-go represents the sum of future rewards from each timestep
        returns_to_go = []
        curr_return = 0
        for r in reversed(episode_rewards):
            curr_return = r + curr_return
            returns_to_go.append(curr_return)
        returns_to_go.reverse()

        # Store episode data
        episode_data["observations"].append(np.array(episode_obs))
        episode_data["actions"].append(np.array(episode_actions))
        episode_data["rewards"].append(np.array(episode_rewards))
        episode_data["returns_to_go"].append(np.array(returns_to_go))
        episode_data["timesteps"].append(np.array(episode_timesteps))

        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            print(f"Collected {episode + 1} episodes. Latest return: {episode_return}")

    return episode_data, episode_returns


def save_to_hdf5(
        episode_data: Dict[str, List[np.ndarray]],
        episode_returns: List[int],
        filename: str
) -> None:
    """
    Saves collected trajectory data to an HDF5 file.

    Args:
        episode_data: Dictionary containing episode-wise trajectory data
        episode_returns: List of total returns for each episode
        filename: Name of the output HDF5 file
    """
    with h5py.File(filename, "w") as f:
        # Save metadata (episode returns)
        f.create_dataset("episode_returns", data=np.array(episode_returns))

        # Create a group for episode data
        episodes_group = f.create_group("episodes")

        # Save each episode's data
        for episode_idx in range(len(episode_data["observations"])):
            episode_group = episodes_group.create_group(f"episode_{episode_idx}")

            # Save each data type (observations, actions, etc.)
            for key in episode_data.keys():
                episode_group.create_dataset(
                    key,
                    data=episode_data[key][episode_idx],
                    compression="gzip"  # Use compression to reduce file size
                )


# Main execution code
if __name__ == "__main__":
    # Initialize environment and model
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the PPO model
    model.learn(total_timesteps=50000)

    # Collect trajectory data
    num_episodes = 50
    episode_data, episode_returns = collect_episodes(env, model, num_episodes)

    # Save collected data
    save_to_hdf5(episode_data, episode_returns, "cartpole_data_50000.hdf5")

    env.close()
    print("Data collection completed and saved to cartpole_data.hdf5")
    print(f"Average return: {np.mean(episode_returns):.2f}")