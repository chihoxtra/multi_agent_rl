import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class ActorNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, output_dim, seed=0):

        super(ActorNet, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_dim,hidden_dim[0])
        self.bn1 = nn.BatchNorm1d(hidden_dim[0])

        self.fc2 = nn.Linear(hidden_dim[0],hidden_dim[1])
        self.bn2 = nn.BatchNorm1d(hidden_dim[1])

        self.fc3 = nn.Linear(hidden_dim[-1],output_dim)

        self.activation = f.relu # relu

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        # batchnorm before activation
        h1 = self.activation(self.bn1(self.fc1(x)))

        h2 = self.activation(self.bn2(self.fc2(h1)))

        a = torch.tanh(self.fc3(h2)) #removed torch.clamp
        return a


class CriticNet(nn.Module):
    def __init__(self, full_state_dim, full_action_dim, hidden_dim, output_dim, seed=0):

        super(CriticNet, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(full_state_dim,hidden_dim[0])
        self.bn1 = nn.BatchNorm1d(hidden_dim[0])

        self.fc2 = nn.Linear(hidden_dim[0]+full_action_dim, hidden_dim[1])
        self.bn2 = nn.BatchNorm1d(hidden_dim[1])

        self.fc3 = nn.Linear(hidden_dim[-1],output_dim)

        self.activation = f.relu
        #self.PReLU = nn.PReLU() # leaky relu

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, full_s, full_a):
        # critic network simply outputs a number

        h1 = self.activation(self.bn1(self.fc1(full_s)))

        m1 = torch.cat((h1, full_a), dim=-1)

        m2 = self.activation(self.bn2(self.fc2(m1)))

        v = self.fc3(m2)
        return v
