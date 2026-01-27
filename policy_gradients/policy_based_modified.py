# -*- coding: utf-8 -*-

# %%
# pip install gymnasium

# %%
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameters
learning_rate = 0.0002  # Learning rate
gamma = 0.98  # Discount factor

# Policy network class definition
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []  # List to store rewards and probabilities

        # Define the neural network layers
        self.fc1 = nn.Linear(4, 128)  # Input: state (4-dimensional vector)
        self.fc2 = nn.Linear(128, 2)  # Output: actions (2-dimensional vector)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)  # Use Adam optimizer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation to the first layer
        x = F.softmax(self.fc2(x), dim=0)  # Apply softmax to generate action probabilities
        return x

    # Function to store data
    def put_data(self, item):
        self.data.append(item)

    # Function to train the policy network
    def train_net(self):
        R = 0
        loss_sum = 0  # 누적 손실을 저장할 변수
        for r, prob in self.data[::-1]:  # 역순으로 데이터를 가져와서
            R = r + gamma * R  # 할인된 보상을 계산
            loss = -torch.log(prob) * R  # 손실 계산
            loss_sum += loss  # 누적 손실 계산

        self.optimizer.zero_grad()  # 경사 초기화
        loss_sum.backward()  # 역전파 (한 번만 호출)
        self.optimizer.step()  # 매개변수 업데이트 (한 번만 호출)
        self.data = []  # 데이터
# Main function definition
def main():
    # Create gymnasium environment
    env = gym.make('CartPole-v1')
    pi = Policy()  # Initialize the policy network
    score = 0.0  # Initialize score
    print_interval = 20  # Set print interval

    for n_epi in range(10000):  # Repeat for 10000 episodes
        s, _ = env.reset()  # Reset the environment and return the state
        s = torch.tensor(s, dtype=torch.float32)  # Convert the state to a tensor
        done = False  # Initialize the done flag

        while not done:  # Continue until the episode ends
            prob = pi(s)  # Compute action probabilities from the current state
            m = Categorical(prob)  # Create a categorical distribution based on probabilities
            a = m.sample()  # Sample an action
            s_prime, r, done, _, _ = env.step(a.item())  # Take a step in the environment
            pi.put_data((r, prob[a]))  # Store the reward and probability
            s = torch.tensor(s_prime, dtype=torch.float32)  # Update to the next state
            score += r  # Accumulate the score

        pi.train_net()  # Train the policy network after the episode

        if n_epi % print_interval == 0 and n_epi != 0:  # Print results at regular intervals
            print(f"# of episodes: {n_epi}, avg score: {score/print_interval:.2f}")
            score = 0.0  # Reset score
    env.close()  # Close the environment

if __name__ == '__main__':
    main()
