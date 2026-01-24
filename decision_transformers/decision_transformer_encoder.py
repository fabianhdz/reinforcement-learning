import torch
import torch.nn as nn
import numpy as np
import h5py
import gymnasium as gym
from torch.utils.data import Dataset, DataLoader


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class DecisionTransformer(nn.Module):
    def __init__(
            self,
            state_dim: int = 4,
            action_dim: int = 2,
            hidden_size: int = 128,
            max_length: int = 20,
            max_ep_len: int = 1000,
            action_tanh: bool = True,
            device=None
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.max_length = max_length
        self.device = device if device is not None else get_device()

        # Embedding layers for state, action, and return
        self.state_encoder = nn.Linear(state_dim, hidden_size)
        self.action_encoder = nn.Linear(action_dim, hidden_size)
        self.return_encoder = nn.Linear(1, hidden_size)

        # Transformer encoder with multi-head attention
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,  # Number of attention heads
                dim_feedforward=4 * hidden_size,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=3  # Number of transformer layers
        )

        self.action_predictor = nn.Linear(hidden_size, action_dim)

        # Timestep embedding for temporal information
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)

        # Move model to specified device
        self.to(self.device)

    def forward(self, states, actions, returns_to_go, timesteps):
        # Move all inputs to the model's device
        states = states.to(self.device)
        actions = actions.to(self.device)
        returns_to_go = returns_to_go.to(self.device)
        timesteps = timesteps.to(self.device)

        batch_size, seq_length = states.shape[0], states.shape[1]

        # Encode states, actions, and returns
        state_embeddings = self.state_encoder(states)
        action_embeddings = self.action_encoder(actions)
        returns_embeddings = self.return_encoder(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # Stack embeddings into sequence
        # Shape: [batch_size, 3 * seq_length, hidden_size]
        sequence = torch.stack(
            [returns_embeddings, state_embeddings, action_embeddings], dim=2
        ).reshape(batch_size, 3 * seq_length, self.hidden_size)

        # Add temporal embeddings
        sequence = sequence + time_embeddings.repeat_interleave(3, dim=1)

        # Pass through transformer
        output = self.transformer(sequence)

        # Predict actions (taking only state positions)
        action_preds = self.action_predictor(output[:, 1::3, :])

        return action_preds


class DecisionTransformerDataset(Dataset):
    def __init__(self, file_path, max_length=20):
        self.max_length = max_length
        self.sequences = []

        with h5py.File(file_path, 'r') as f:
            episodes = f['episodes']

            # Process each episode into training sequences
            for episode_key in episodes.keys():
                episode = episodes[episode_key]
                states = episode['observations'][:]
                actions = episode['actions'][:]
                returns = episode['returns_to_go'][:]
                timesteps = episode['timesteps'][:]

                # Create overlapping sequences of length max_length
                for i in range(0, len(states) - max_length + 1):
                    self.sequences.append({
                        'states': states[i:i + max_length],
                        'actions': actions[i:i + max_length],
                        'returns': returns[i:i + max_length],
                        'timesteps': timesteps[i:i + max_length]
                    })

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return (
            torch.FloatTensor(sequence['states']),
            torch.FloatTensor(self._to_one_hot(sequence['actions'])),
            torch.FloatTensor(sequence['returns']).reshape(-1, 1),
            torch.LongTensor(sequence['timesteps'])
        )

    def _to_one_hot(self, actions):
        one_hot = np.zeros((len(actions), 2))
        one_hot[np.arange(len(actions)), actions] = 1
        return one_hot


def train_epoch(model, optimizer, dataloader, device):

    model.train()
    total_loss = 0

    for states, actions, returns, timesteps in dataloader:
        # Move data to device
        states = states.to(device)
        actions = actions.to(device)
        returns = returns.to(device)
        timesteps = timesteps.to(device)

        # Forward pass
        action_preds = model(states, actions, returns, timesteps)

        # Compute loss (predict next action given current state)
        loss = nn.MSELoss()(action_preds[:, :-1], actions[:, 1:])

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_model(
        model_path: str,
        num_episodes: int = 10,
        target_return: float = 500.0,
        render: bool = True
):

    device = get_device()

    # Create environment with rendering if specified
    env = gym.make("CartPole-v1", render_mode="human" if render else None)

    # Load model
    model = DecisionTransformer(device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    max_ep_len = 500  # Maximum episode length for CartPole-v1

    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        current_reward = 0
        episode_reward = 0

        # Initialize sequence buffers
        states = torch.zeros((1, model.max_length, 4), dtype=torch.float32, device=device)
        actions = torch.zeros((1, model.max_length, 2), dtype=torch.float32, device=device)
        returns_to_go = torch.zeros((1, model.max_length, 1), dtype=torch.float32, device=device)
        timesteps = torch.arange(0, model.max_length, device=device).long().unsqueeze(0)

        # Set initial return-to-go
        returns_to_go[:, 0, 0] = target_return

        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print("Step: 0, Current Reward: 0", end='\r')

        for t in range(max_ep_len):
            if render:
                env.render()

            # Update state history
            states = states.roll(-1, dims=1)
            states[:, -1] = torch.tensor(state, device=device)

            # Get action from model
            with torch.no_grad():
                action_preds = model(states, actions, returns_to_go, timesteps)
                action = torch.argmax(action_preds[0, -1]).item()

            # Execute action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            current_reward += reward
            episode_reward += reward

            # Display current progress
            print(f"Step: {t + 1}, Current Reward: {current_reward}", end='\r')

            # Update action history
            actions = actions.roll(-1, dims=1)
            action_one_hot = torch.zeros(2, device=device)
            action_one_hot[action] = 1
            actions[:, -1] = action_one_hot

            # Update return-to-go
            returns_to_go = returns_to_go.roll(-1, dims=1)
            returns_to_go[:, -1, 0] = max(0, returns_to_go[:, -2, 0] - reward)

            state = next_state

            if terminated or truncated:
                break

        episode_rewards.append(episode_reward)
        print(f"\nEpisode {episode + 1} finished with total reward: {episode_reward}")

    env.close()

    # Print summary statistics
    print("\n=== Evaluation Results ===")
    print(f"Average Reward over {num_episodes} episodes: {np.mean(episode_rewards):.2f}")
    print(f"Standard Deviation: {np.std(episode_rewards):.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")


def train(num_epochs=50):

    device = get_device()
    print(f"Using device: {device}")

    # Initialize model
    model = DecisionTransformer(device=device)
    print(f"Model initialized on {device}")

    # Setup dataset and dataloader
    dataset = DecisionTransformerDataset("cartpole_data_50000.hdf5")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Training loop
    try:
        for epoch in range(num_epochs):
            loss = train_epoch(model, optimizer, dataloader, device)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

            # Save checkpoints periodically
            if (epoch + 1) % 10 == 0:
                checkpoint_path = f"decision_transformer_checkpoint_epoch_{epoch + 1}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, checkpoint_path)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model state...")
    finally:
        # Save final model
        torch.save(model.state_dict(), "decision_transformer_cartpole_final.pth")
        print("Final model saved as decision_transformer_cartpole_final.pth")

# python decision_transformer.py --mode train --num_epochs 50
# python decision_transformer.py --mode eval --model_path decision_transformer_cartpole_final.pth --num_episodes 10
if __name__ == "__main__":
    import argparse

    # Setup command line argument parser
    parser = argparse.ArgumentParser(description='Train or evaluate Decision Transformer')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], required=True,
                        help='train or eval mode')
    parser.add_argument('--model_path', type=str, default="decision_transformer_cartpole_final.pth",
                        help='path to model file for evaluation')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs for training')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='number of episodes for evaluation')
    parser.add_argument('--no_render', action='store_true',
                        help='disable rendering during evaluation')

    args = parser.parse_args()

    # Run either training or evaluation based on arguments
    if args.mode == 'train':
        train(num_epochs=args.num_epochs)
    else:
        evaluate_model(
            args.model_path,
            num_episodes=args.num_episodes,
            render=not args.no_render
        )