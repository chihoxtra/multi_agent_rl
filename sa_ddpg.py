# individual network settings for each actor + critic pair
# see networkforall for details

from networkforall import Network
from utilities import hard_update, toTorch
from torch.optim import Adam
import torch
import numpy as np

# add OU noise for exploration
from OUNoise import OUNoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPGAgent:
    def __init__(self, state_size, action_size, num_agents,
                 hidden_actor, hidden_critic, lr_actor, lr_critic):

        super(DDPGAgent, self).__init__()

        self.action_size = action_size

        in_critic = num_agents * (state_size + action_size) #2*(24+2) = 52

        self.actor_local = Network(state_size, hidden_actor, action_size, actor=True).to(device)
        self.critic_local = Network(in_critic, hidden_critic, 1).to(device)
        self.actor_target = Network(state_size, hidden_actor, action_size, actor=True).to(device)
        self.critic_target = Network(in_critic, hidden_critic, 1).to(device)

        self.noise = OUNoise(action_size, scale=1.0) #use external scale

        # initialize targets same as original networks
        hard_update(self.actor_target, self.actor_local)
        hard_update(self.critic_target, self.critic_local)

        self.actor_optimizer = Adam(self.actor_local.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic_local.parameters(), lr=lr_critic) #weight_decay=1.e-5

    def act(self, obs, noise_scale=0.00, useOUnoise=False, use_bn=True):

        if len(obs.shape)==1: #1D tensor cannot do batchnorm
            use_bn = False

        if not useOUnoise:
            noise = noise_scale * toTorch(np.random.normal(0,1.0,self.action_size))
        else:
            noise = self.noise.noise(scale=noise_scale)

        action = self.actor_local(obs, use_bn=use_bn) + noise
        #print(noise_scale,noise_scale*G_noise.mean(), action.mean())
        return action

    def target_act(self, obs, noise_scale=0.00, useOUnoise=False, use_bn=True):
        obs = obs.to(device)

        if len(obs.shape)==1: #1D tensor cannot do batchnorm
            use_bn = False

        if not useOUnoise:
            # Gaussian noise
            noise = noise_scale * toTorch(np.random.normal(0,1.0,self.action_size))
        else:
            noise = self.noise.noise(scale=noise_scale)

        action = self.actor_target(obs, use_bn=use_bn) + noise

        #print(noise_scale,noise_scale*G_noise.mean(), action.mean())
        return action
