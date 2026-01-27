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
import numpy as np  # numpy 추가

# Hyperparameters
learning_rate = 0.0002
gamma = 0.98
n_rollout = 10

# Actor-Critic Neural Network
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(4, 256)  # Input: state (CartPole has 4 states)
        self.fc_pi = nn.Linear(256, 2)  # Output: action probabilities
        self.fc_v = nn.Linear(256, 1)  # Output: state value
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)


    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(np.array(s))  # 상태를 numpy 배열로 변환하여 저장
            a_lst.append([a])
            r_lst.append([r / 100.0])
            s_prime_lst.append(np.array(s_prime))  # 다음 상태도 numpy 배열로 변환
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

        # numpy 배열을 텐서로 변환
        s_batch = torch.tensor(np.array(s_lst), dtype=torch.float32)
        a_batch = torch.tensor(a_lst)
        r_batch = torch.tensor(r_lst, dtype=torch.float32)
        s_prime_batch = torch.tensor(np.array(s_prime_lst), dtype=torch.float32)
        done_batch = torch.tensor(done_lst, dtype=torch.float32)

        self.data = []  # 데이터 초기화
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch


    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()

        # Calculate TD target and TD error
        td_target = r + gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)

        # Actor loss (policy loss) using policy gradient
        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1, a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

        # Backpropagation
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

def main():
    env = gym.make('CartPole-v1')  # Create CartPole environment using gymnasium
    model = ActorCritic()  # Initialize the Actor-Critic model
    print_interval = 20  # Set print interval for results
    score = 0.0  # Initialize score

    for n_epi in range(10000):  # Loop for 10000 episodes
        done = False
        s, _ = env.reset()  # Reset the environment, discard extra info
        s = torch.tensor(s, dtype=torch.float32)  # Convert state to tensor

        while not done:
            for t in range(n_rollout):  # Execute n_rollout steps before training
                prob = model.pi(s)
                m = Categorical(prob)
                a = m.sample().item()  # Sample action

                # Step the environment and get new state and reward
                s_prime, r, done, truncated, _ = env.step(a)
                done = done or truncated  # Ensure episode ends correctly

                model.put_data((s, a, r, s_prime, done))  # Store transition
                s = torch.tensor(s_prime, dtype=torch.float32)  # Update state
                score += r  # Accumulate score

                if done:
                    break  # End episode if done

            # Train the network after every n_rollout steps
            model.train_net()

        # Print the results every print_interval episodes
        if n_epi % print_interval == 0 and n_epi != 0:
            print(f"# of episodes: {n_epi}, avg score: {score / print_interval:.1f}")
            score = 0.0  # Reset score

    env.close()  # Close the environment

if __name__ == '__main__':
    main()
