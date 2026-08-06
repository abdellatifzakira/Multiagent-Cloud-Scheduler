import torch
import torch.nn as nn
import random
import numpy as np

try:
    from Agent import Agent
except ModuleNotFoundError:
    from RL.agents.Agent import Agent


class QNetwork(nn.Module):

    def __init__(self, state_size, action_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(state_size, 16),
            nn.ReLU(),

            nn.Linear(16, 64),
            nn.ReLU(),

            nn.Linear(64, action_size)
        )


    def forward(self, x):
        return self.model(x)



class DQNAgent(Agent):

    def __init__(self):

        super().__init__(
            "DQN",
            "Deep Q-Network Agent"
        )

        self.state = None

        self.network = None
        self.optimizer = None

        self.epsilon = 1.0


    def build(self):


        if self.action_space is None:
            raise RuntimeError(
                "Action space not provided by environment"
            )


        self.network = QNetwork(
            state_size=9,
            action_size=self.action_space.n
        )

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=1e-3
        )


    def observe(self, state):

        self.state = np.array(state, dtype=np.float32)



    def take_action(self):

        if random.random() < self.epsilon:

            return self.action_space.sample()


        state = torch.FloatTensor(
            self.state
        ).unsqueeze(0)


        with torch.no_grad():

            q_values = self.network(state)


        return torch.argmax(q_values).item()
    
    def update(self, action, reward, next_state, done):
        # Current state
        state_tensor = torch.FloatTensor(
            self.state
        ).unsqueeze(0)

        # Next state
        next_state_tensor = torch.FloatTensor(
            next_state
        ).unsqueeze(0)


        # Q(s,a)
        q_values = self.network(state_tensor)

        current_q = q_values[0, action]


        # Target: r + gamma max(Q(s',a'))
        with torch.no_grad():

            if done:
                target_q = torch.tensor(reward,
                                         dtype=torch.float32)

            else:
                next_q_values = self.network(next_state_tensor)

                target_q = reward + 0.99 * torch.max(next_q_values)


        # Loss
        loss = nn.MSELoss()(
            current_q,
            target_q
        )


        # Gradient update
        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()