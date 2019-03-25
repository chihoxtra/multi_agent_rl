"""
DDPGAgent a class for one single agent
"""
import torch
import numpy as np
from torch.optim import Adam
from utilities import hard_update, toTorch
from collections import namedtuple, deque

from models import ActorNet, CriticNet
from utilities import toTorch
from sumTree import SumTree

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPGAgent:
    def __init__(self, state_size, action_size, num_agents,
                 hidden_actor, hidden_critic, lr_actor, lr_critic,
                 buffer_size, agent_id, use_PER=False, seed=0):

        super(DDPGAgent, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.agent_id = agent_id

        # num_agents*action_size
        self.actor_local = ActorNet(1*state_size, hidden_actor, action_size, seed=seed).to(device)
        self.critic_local = CriticNet(num_agents*state_size, num_agents*action_size, hidden_critic, 1, seed=seed).to(device)
        self.actor_target = ActorNet(1*state_size, hidden_actor, action_size, seed=seed).to(device)
        self.critic_target = CriticNet(num_agents*state_size, num_agents*action_size, hidden_critic, 1, seed=seed).to(device)

        # initialize targets same as original networks
        hard_update(self.actor_target, self.actor_local)
        hard_update(self.critic_target, self.critic_local)

        self.actor_optimizer = Adam(self.actor_local.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=1.e-5) #weight_decay=1.e-5

        self.memory = ReplayBuffer(buffer_size, num_agents, state_size, action_size, use_PER)


    def _act(self, obs, use_bn=False):
        obs = obs.to(device)
        if len(obs.shape)==1: use_bn = False #1D tensor cannot do batchnorm

        return self.actor_local(obs, use_bn=use_bn)

    def _target_act(self, obs, use_bn=False):
        obs = obs.to(device)
        if len(obs.shape)==1: use_bn = False #1D tensor cannot do batchnorm

        return self.actor_target(obs, use_bn=use_bn)

class ReplayBuffer:
    def __init__(self, buffer_size, num_agents, state_size, action_size, use_PER=False):

        self.buffer_size = buffer_size
        self.use_PER = use_PER
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size

        if use_PER:
            self.tree = SumTree(buffer_size) #create tree instance
        else:
            self.memory = deque(maxlen=buffer_size)

        self.buffer_size = buffer_size
        self.p_replay_alpha = 0.7
        self.leaves_count = 0

    def add_tree(self, e, td_error=1.0):
        """PER function. Add a new experience to memory. td_error: abs value"""
        self.tree.add(td_error, e)
        self.leaves_count = min(self.leaves_count+1,self.buffer_size)

    def add(self, data):
        """add into the buffer"""
        self.memory.append(data)

    def sample_tree(self, batch_size, p_replay_beta, td_eps=1e-3):
        """PER function. Segment piece wise sampling"""
        s_samp, a_samp, r_samp, d_samp, ns_samp = ([] for l in range(5))

        sample_ind = np.empty((batch_size,), dtype=np.int32)
        weight = np.empty((batch_size, 1))

        # priority segments across td score range
        td_score_segment = self.tree.total_td_score / batch_size

        # Calculating the max_weight
        #p_min = np.min(self.tree.tree[-self.buffer_size:]) / self.tree.total_td_score
        #if p_min == 0:
        #    p_min = td_eps # avoid div by zero
        #max_weight = (p_min * self.buffer_size) ** (-p_replay_beta)

        for i in range(batch_size):
            # A value is uniformly sample from each range
            a, b = td_score_segment * i, td_score_segment * (i + 1)
            value = np.random.uniform(a, b)

            # Experience that correspond to each value is retrieved
            leaf_index, td_score, data = self.tree.get_leaf(value)

            #P(j)
            sampling_p = td_score / self.tree.total_td_score

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            weight[i,0] = (1/sampling_p * 1/self.leaves_count)**p_replay_beta

            sample_ind[i] = leaf_index

            s_samp.append(data.states)
            a_samp.append(data.actions)
            r_samp.append(data.rewards)
            d_samp.append(data.dones)
            ns_samp.append(data.next_states)

        weight_n = toTorch(weight/np.max(weight)).mean() #change form

        return (s_samp, a_samp, r_samp, d_samp, ns_samp, weight_n, sample_ind)


    def sample(self, batch_size):
        """sample from the buffer"""
        sample_ind = np.random.choice(len(self.memory), batch_size)

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

        # last 2 values for functions compatibility with PER
        return (s_samp, a_samp, r_samp, d_samp, ns_samp, 1.0, [])

    def update_tree(self, td_updated, index, p_replay_alpha, td_eps):
        """ PER function.
        update the td error values while restoring orders
        td_updated: abs value; np.array of shape 1,batch_size,1
        index: in case of tree, it is the leaf index
        """
        # apply alpha power
        td_updated = (td_updated.squeeze() ** p_replay_alpha) + td_eps

        for i in range(len(index)):
            self.tree.update(index[i], td_updated[i])

    def __len__(self):
        if not self.use_PER:
            return len(self.memory)
        else:
            return self.leaves_count
