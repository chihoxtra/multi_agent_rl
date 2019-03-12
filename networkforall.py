import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, actor=False):

        super(Network, self).__init__()

        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""


        self.fc1 = nn.Linear(input_dim,hidden_dim[0], bias=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim[0])

        self.fc2 = nn.Linear(hidden_dim[0],hidden_dim[1], bias=True)
        self.bn2 = nn.BatchNorm1d(hidden_dim[1])

        #self.fc3 = nn.Linear(hidden_dim[1],hidden_dim[2], bias=True)
        #self.bn3 = nn.BatchNorm1d(hidden_dim[2])

        self.fc4 = nn.Linear(hidden_dim[1],output_dim)
        #self.fc4 = nn.Linear(hidden_dim[2],output_dim)

        self.activation = f.relu #leaky_relu
        self.actor = actor

        self.std = nn.Parameter(torch.ones(1, output_dim)*0.15)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        #self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x, use_bn=True):
        #print(x.shape, self.actor)
        if self.actor: # this network is an actor network
            # return a vector of the force
            h1 = self.activation(self.fc1(x))
            if use_bn: h1 = self.bn1(h1)

            h2 = self.activation(self.fc2(h1))
            if use_bn: h2 = self.bn2(h2)

            #h3 = self.activation(self.fc3(h2))
            #if use_bn: h3 = self.bn3(h3)
            a = torch.tanh(self.fc4(h2))
            return a

        else:
            # critic network simply outputs a number
            h1 = self.activation(self.fc1(x))
            if use_bn: h1 = self.bn1(h1)

            h2 = self.activation(self.fc2(h1))
            if use_bn: h2 = self.bn2(h2)

            #h3 = self.activation(self.fc3(h2))
            #if use_bn: h3 = self.bn3(h3)
            v = self.fc4(h2)
            return v
