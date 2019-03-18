# individual network settings for each actor + critic pair
# see networkforall for details

from models import ActorNet, CriticNet
from utilities import hard_update, toTorch
from torch.optim import Adam
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPGAgent:
    def __init__(self, state_size, action_size, num_agents,
                 hidden_actor, hidden_critic, lr_actor, lr_critic, seed=0):

        super(DDPGAgent, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.action_size = action_size

        self.actor_local = ActorNet(state_size, hidden_actor, action_size, seed=seed).to(device)
        self.critic_local = CriticNet(num_agents*state_size, num_agents*action_size, hidden_critic, 1, seed=seed).to(device)
        self.actor_target = ActorNet(state_size, hidden_actor, action_size, seed=seed).to(device)
        self.critic_target = CriticNet(num_agents*state_size, num_agents*action_size, hidden_critic, 1, seed=seed).to(device)

        # initialize targets same as original networks
        hard_update(self.actor_target, self.actor_local)
        hard_update(self.critic_target, self.critic_local)

        self.actor_optimizer = Adam(self.actor_local.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=0.0) #weight_decay=1.e-5

    def _act(self, obs, use_bn=True):
        obs = obs.to(device)
        if len(obs.shape)==1: #1D tensor cannot do batchnorm
            use_bn = False

        action = self.actor_local(obs, use_bn=use_bn)

        return action

    def _target_act(self, obs, use_bn=True):
        obs = obs.to(device)
        if len(obs.shape)==1: #1D tensor cannot do batchnorm
            use_bn = False

        action = self.actor_target(obs, use_bn=use_bn)

        return action
