import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class FCNetwork(nn.Module):

    def __init__(self, state_size, action_size, output_gate=None):
        super(FCNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size * 2, 256)
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2 = nn.Linear(256, 256)
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3 = nn.Linear(256, action_size)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.output_gate = output_gate

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if self.output_gate is not None:
            x = self.output_gate(x)

        return x

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.network = FCNetwork(state_size, action_size, F.tanh)

    def forward(self, state):
        return self.network(state)

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.network = FCNetwork(state_size + action_size, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network(x)
