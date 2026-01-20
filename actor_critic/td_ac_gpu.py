import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard writer

# Hyperparameters
learning_rate = 0.0002
gamma = 0.98
n_rollout = 10

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor-Critic Neural Network
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []  # Stores transitions (state, action, reward, next_state, done)

        # Fully connected layers for policy and value function
        self.fc1 = nn.Linear(4, 256)  # Input layer (4 states in CartPole), hidden layer with 256 units
        self.fc_pi = nn.Linear(256, 2)  # Output for policy (2 actions in CartPole: left or right)
        self.fc_v = nn.Linear(256, 1)  # Output for value function (state value)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)  # Adam optimizer for both actor and critic

    # Policy network to return action probabilities
    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))  # ReLU activation for hidden layer
        x = self.fc_pi(x)  # Output layer for policy
        prob = F.softmax(x, dim=softmax_dim)  # Softmax to get action probabilities
        return prob

    # Value network to return state value
    def v(self, x):
        x = F.relu(self.fc1(x))  # ReLU activation for hidden layer
        v = self.fc_v(x)  # Output layer for value function
        return v

    # Store transition in memory (for batch processing)
    def put_data(self, transition):
        self.data.append(transition)

    # Create a batch from stored transitions
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(np.array(s))  # Convert state to numpy array
            a_lst.append([a])
            r_lst.append([r / 100.0])  # Normalize reward
            s_prime_lst.append(np.array(s_prime))
            done_mask = 0.0 if done else 1.0  # Set done flag as 0 if episode is over, else 1
            done_lst.append([done_mask])

        # Convert numpy arrays to tensors and move them to the specified device (CPU or GPU)
        s_batch = torch.tensor(np.array(s_lst), dtype=torch.float32).to(device)
        a_batch = torch.tensor(a_lst).to(device)
        r_batch = torch.tensor(r_lst, dtype=torch.float32).to(device)
        s_prime_batch = torch.tensor(np.array(s_prime_lst), dtype=torch.float32).to(device)
        done_batch = torch.tensor(done_lst, dtype=torch.float32).to(device)

        self.data = []  # Clear the data after batch creation
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    # Train the actor-critic network using a batch of data
    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()

        # Calculate TD target: reward + gamma * value of next state (if not done)
        td_target = r + gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)  # TD error

        # Actor loss (policy gradient): maximize log probability of selected action weighted by TD error
        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1, a)  # Select the probability of the taken action
        loss = -torch.log(pi_a) * delta.detach()  # Policy loss
        loss += F.smooth_l1_loss(self.v(s), td_target.detach())  # Critic loss (value function regression)

        # Backpropagation to update network weights
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

def main():
    env = gym.make('CartPole-v1')  # Initialize CartPole environment
    model = ActorCritic().to(device)  # Move model to the specified device (GPU or CPU)
    writer = SummaryWriter(log_dir='./runs/test')  # Initialize TensorBoard writer
    print_interval = 20  # Print stats every 20 episodes
    score = 0.0  # Initialize score tracking

    for n_epi in range(1000):  # Loop episodes
        done = False
        s, _ = env.reset()  # Reset environment at the start of each episode
        s = torch.tensor(s, dtype=torch.float32).to(device)  # Convert state to tensor and move to device

        episode_score = 0.0  # Track score for this episode

        while not done:
            for t in range(n_rollout):  # Collect data for 'n_rollout' steps
                prob = model.pi(s)  # Get action probabilities from the policy
                m = Categorical(prob)  # Sample action based on the probabilities
                a = m.sample().item()

                # Step the environment using the selected action
                s_prime, r, done, truncated, _ = env.step(a)
                done = done or truncated  # If truncated, treat as done

                # Store transition in memory
                model.put_data((s.cpu().numpy(), a, r, s_prime, done))
                s = torch.tensor(s_prime, dtype=torch.float32).to(device)  # Update state

                episode_score += r  # Accumulate reward for the episode
                score += r  # Accumulate reward for print interval

                if done:  # Break the loop if the episode ends
                    break

            model.train_net()  # Train the model after each rollout

        # Record episode score to TensorBoard
        writer.add_scalar('Reward/Episode', episode_score, n_epi)

        if n_epi % print_interval == 0 and n_epi != 0:  # Print stats every 'print_interval' episodes
            print(f"# of episodes: {n_epi}, avg score: {score / print_interval:.1f}")
            score = 0.0  # Reset score after printing

    env.close()  # Close the environment after training
    writer.close()  # Close the TensorBoard writer

if __name__ == '__main__':
    main()
