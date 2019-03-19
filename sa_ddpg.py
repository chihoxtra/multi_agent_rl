# individual network settings for each actor + critic pair
# see networkforall for details

from models import ActorNet, CriticNet
from utilities import hard_update, toTorch
from torch.optim import Adam
from collections import namedtuple, deque
import torch
import numpy as np

#USE_TREE = True                          # use tree structure as memory
#INIT_TD_ERROR = 1.0                      # initial value of td error

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPGAgent:
    def __init__(self, state_size, action_size, num_agents,
                 hidden_actor, hidden_critic, lr_actor, lr_critic,
                 buffer_size, batch_size, seed=0):

        super(DDPGAgent, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.actor_local = ActorNet(state_size, hidden_actor, action_size, seed=seed).to(device)
        self.critic_local = CriticNet(num_agents*state_size, num_agents*action_size, hidden_critic, 1, seed=seed).to(device)
        self.actor_target = ActorNet(state_size, hidden_actor, action_size, seed=seed).to(device)
        self.critic_target = CriticNet(num_agents*state_size, num_agents*action_size, hidden_critic, 1, seed=seed).to(device)

        # initialize targets same as original networks
        hard_update(self.actor_target, self.actor_local)
        hard_update(self.critic_target, self.critic_local)

        self.actor_optimizer = Adam(self.actor_local.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=0.0) #weight_decay=1.e-5

        self.memory = ReplayBuffer(buffer_size, num_agents, state_size, action_size,
                                   batch_size)


    def _act(self, obs, use_bn=False):
        obs = obs.to(device)
        if len(obs.shape)==1: #1D tensor cannot do batchnorm
            use_bn = False

        action = self.actor_local(obs, use_bn=use_bn)

        return action

    def _target_act(self, obs, use_bn=False):
        obs = obs.to(device)
        if len(obs.shape)==1: #1D tensor cannot do batchnorm
            use_bn = False

        action = self.actor_target(obs, use_bn=use_bn)

        return action

class ReplayBuffer:
    def __init__(self, buffer_size, num_agents, state_size, action_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size

    def add(self, data):
        """add into the buffer"""
        self.memory.append(data)

    def sample(self, batch_size):
        """sample from the buffer"""
        sample_ind = np.random.choice(len(self.memory), batch_size)
        #print(samples[0].states.shape) #2,24

        # get the selected experiences: avoid using mid list indexing
        s_samp, a_samp, r_samp, d_samp, ns_samp = ([] for l in range(5))

        i = 0
        while i < batch_size: #while loop is faster
            self.memory.rotate(-sample_ind[i])
            e = self.memory[0]
            s_samp.append(e.states)
            a_samp.append(e.actions)
            r_samp.append(e.rewards)
            d_samp.append(e.dones)
            ns_samp.append(e.next_states)
            self.memory.rotate(sample_ind[i])
            i += 1

        return (s_samp, a_samp, r_samp, d_samp, ns_samp)


    def __len__(self):
        return len(self.memory)
