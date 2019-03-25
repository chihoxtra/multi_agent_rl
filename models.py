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

        self.fc1 = nn.Linear(state_dim,hidden_dim[0], bias=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim[0])

        self.fc2 = nn.Linear(hidden_dim[0],hidden_dim[1], bias=True)
        self.bn2 = nn.BatchNorm1d(hidden_dim[1])

        self.fc4 = nn.Linear(hidden_dim[1],output_dim)

        #self.activation = nn.PReLU() #leaky_relu relu
        self.activation = f.relu # relu

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x, use_bn=True):
        h1 = self.activation(self.fc1(x))
        if use_bn: h1 = self.bn1(h1)

        h2 = self.activation(self.fc2(h1))
        if use_bn: h2 = self.bn2(h2)

        a = torch.clamp(torch.tanh(self.fc4(h2)), min=-1.0, max=+1.0)
        return a


class CriticNet(nn.Module):
    def __init__(self, full_state_dim, full_action_dim, hidden_dim, output_dim, actor=False, seed=0):

        super(CriticNet, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(full_state_dim,hidden_dim[0], bias=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim[0])

        self.fc2 = nn.Linear(hidden_dim[0]+full_action_dim, hidden_dim[1], bias=True)
        self.bn2 = nn.BatchNorm1d(hidden_dim[1])

        #self.fc3 = nn.Linear(hidden_dim[1], hidden_dim[2], bias=True)
        #self.bn3 = nn.BatchNorm1d(hidden_dim[2])

        self.fc4 = nn.Linear(hidden_dim[-1],output_dim)

        self.activation = f.relu
        self.PReLU = nn.PReLU() # relu

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        #self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, full_s, full_a, use_bn=True):
        # critic network simply outputs a number
        h1 = self.activation(self.fc1(full_s))
        if use_bn: h1 = self.bn1(h1)
        m1 = torch.cat((h1, full_a), dim=-1)

        h2 = self.activation(self.fc2(m1))
        if use_bn: h2 = self.bn2(h2)

        #h3 = self.activation(self.fc3(h2))
        #if use_bn: h3 = self.bn3(h3)

        v = self.PReLU(self.fc4(h2)) #pRELU
        return v
