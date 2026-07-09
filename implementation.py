import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class CNNPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNNPolicy, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        mean = self.fc2(x)
        std = torch.exp(self.log_std)
        return mean, std

class ActorBasedAgent:
    def __init__(self, input_dim, output_dim, lr=1e-3, gamma=0.99, risk_sensitivity=1.0):
        self.policy = CNNPolicy(input_dim, output_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.risk_sensitivity = risk_sensitivity
        self.memory = []

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        mean, std = self.policy(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        return action.detach().numpy()[0], log_prob

    def store_transition(self, reward, log_prob):
        self.memory.append((reward, log_prob))

    def update_policy(self):
        R = 0
        policy_loss = []
        returns = []
        for reward, _ in reversed(self.memory):
            R = reward + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for (reward, log_prob), R in zip(self.memory, returns):
            adjusted_reward = R - self.risk_sensitivity * torch.abs(R)
            policy_loss.append(-log_prob * adjusted_reward)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        self.memory = []

if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)

    # Dummy data for testing
    num_steps = 100
    input_dim = 10
    output_dim = 1
    dummy_prices = np.random.randn(num_steps, input_dim)
    dummy_rewards = np.random.randn(num_steps)

    agent = ActorBasedAgent(input_dim=input_dim, output_dim=output_dim)

    for t in range(num_steps):
        state = dummy_prices[t]
        action, log_prob = agent.select_action(state)
        reward = dummy_rewards[t]
        agent.store_transition(reward, log_prob)

        if (t + 1) % 10 == 0:  # Update policy every 10 steps
            agent.update_policy()

    print("Policy updated successfully.")