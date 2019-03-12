# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network
import torch
import numpy as np
import torch.nn.functional as F
import random
#from tensorboardX import SummaryWriter
from collections import namedtuple, deque
from sa_ddpg import DDPGAgent
from utilities import toTorch, soft_update

BUFFER_SIZE = int(1e5)            # size of memory replay buffer
BATCH_SIZE = 64                   # min batch size
MIN_BUFFER_SIZE = int(1e4)        # min buffer size before replay
LR_ACTOR = 1e-4                   # learning rate
LR_CRITIC = 1e-4                  # learning rate
UNITS_ACTOR = (64,32)             # number of hidden units for actor inner layers
UNITS_CRITIC = (64,32)            # number of hidden units for critic inner layers
GAMMA = 0.99                      # discount factor
TAU = 0.01                        # soft network update
LEARN_EVERY = 2                  # how often to learn
UPDATE_EVERY = 4                 # how many steps before updating the network
NOISE_WGT_INIT = 10.0             # noise scaling weight
NOISE_WGT_DECAY = 0.9999          # noise decay rate per STEP
NOISE_WGT_MIN = 0.1               # min noise scale
NOISE_DC_START = int(1e4)         # when to start noise decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG:
    def __init__(self, state_size, action_size, num_agents, seed=0):
        super(MADDPG, self).__init__()

        self.num_agents = num_agents

        # critic input = obs_full + actions = 14+2+2+2=20
        self.maddpg_agent = [DDPGAgent(state_size, action_size, num_agents,
                                       UNITS_ACTOR, UNITS_CRITIC, LR_ACTOR, LR_CRITIC)
                             for _ in range(num_agents)]

        # replay buffer
        self.memory = ReplayBuffer(BUFFER_SIZE)

        # data structure for storing individual experience
        self.data = namedtuple("data", field_names=["states", "actions", "rewards",
                                                    "dones", "next_states"])

        self.gamma = GAMMA
        self.noise_scale = NOISE_WGT_INIT
        self.tau = TAU

        self.t_step = 0 # count number of steps went thru

        self.is_training = False

        # for tracking
        self.cl_history = deque(maxlen=100)
        self.ag_history = deque(maxlen=100)


    def _toTorch(self, s, dtype=torch.float32):
        return torch.tensor(s, dtype=dtype, device=device)

    def act(self, obs_all_agents):
        """get actions from all agents in the MADDPG object"""
        obs_all_agents = toTorch(obs_all_agents) #num_agents x space_size (24)
        actions = [agent.act(obs, self.noise_scale)
                   for agent, obs
                   in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def target_acts(self, obs_all_agents, noise_scale):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs, noise_scale)
                          for ddpg_agent, obs
                          in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions

    def step(self, data):
        states, actions, rewards, dones, next_states = data

        # add to memory
        e = self.data(toTorch(states), #num_agent x state_size
                      actions, #tensor: #num_agent x actions size
                      toTorch(rewards).unsqueeze(-1), #num_agent x 1
                      toTorch(1.*np.array(dones)).unsqueeze(-1), #num_agent x 1
                      toTorch(next_states)) #num_agent x state_size

        self.memory.add(e)

        # sample from memory
        if len(self.memory) >= MIN_BUFFER_SIZE:

            if self.is_training == False:
                print("")
                print("Prefetch completed. Training starts! \r")
                print("Number of Agents: ", self.num_agents)
                print("Device: ", device)
                self.is_training = True

            if self.t_step % LEARN_EVERY == 0:

                for agent_id in range(self.num_agents): #do it agent by agent

                    agent_inputs = self.memory.sample(BATCH_SIZE) #by fields

                    self.learn(agent_inputs, agent_id)

                if self.t_step >= NOISE_DC_START:
                    self.noise_scale = max(self.noise_scale * NOISE_WGT_DECAY, NOISE_WGT_MIN)

            if self.t_step % UPDATE_EVERY == 0:

                self.update_targets()

        self.t_step += 1


    def learn(self, agent_inputs, agent_id):
        """update the critics and actors of all the agents """

        s_full, a_full , ns_full, s, a, r, d, ns = agent_inputs

        agent = self.maddpg_agent[agent_id]

        ####################### CRITIC LOSS #########################
        """
        update params of critic network by minimizing td error
        td target = expected Q value using the critic target network
                    reward of this timestep + discount * Q_target(st+1,at+1)
                                                         from target network
        td current = current Q
        critic loss = mse(td_target - td_current)
        """
        # 1) compute td target of full next obversation
        ns_full_actions = self.target_acts(ns, 0.0) #input list of len num_agents [batch_size x state_size]
        ns_full_actions = torch.cat(ns_full_actions, dim=-1) #batch_size x (action sizexnum_agents)

        ns_critic_input = torch.cat((ns_full,ns_full_actions), dim=-1).to(device)
        # batch size x (num_agent x (state_size + action size))

        with torch.no_grad():
            q_next_target = agent.critic_target(ns_critic_input)

        td_target = r[agent_id] + GAMMA * q_next_target * (1.-d[agent_id])

        # 2) compute td current (full obs) using raw data feed into critic current
        critic_current_input = torch.cat((s_full, a_full), dim=-1).to(device)

        td_current = agent.critic_local(critic_current_input)
        assert(td_current.requires_grad==True)

        # 3) compute the critic loss
        critic_loss = (td_target.detach() - td_current).pow(2).mean()

        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 0.8)
        agent.critic_optimizer.step()

        ####################### ACTOR LOSS #########################
        """
        evaluate performance of current actor network
        update the actor network by MAXimizing the Q value produced by
        critic network which inputs are controlled by actor networks (actions)
        """
        # 1) get the latest predicted actions with current states
        # noted that we are using LOCAL network as this is the real actor
        # detach the other agents to save computation saves some computation
        latest_a_full = [self.maddpg_agent[i].actor_local(ob) if i == agent_id \
                         else self.maddpg_agent[i].actor_local(ob).detach()
                         for i, ob in enumerate(s)]
        # combine latest prediction from 2 agents to form full actions
        latest_a_full = torch.cat(latest_a_full, dim=-1)
        # actions has to be differtiable so that parameters can change
        # to produce an action that produce a higher critic score
        assert(latest_a_full.requires_grad==True)

        # 2) actions (by actor local network) feed to local critic for score
        # combine all the actions and states for input to critic
        full_critic_input = torch.cat((s_full, latest_a_full), dim=-1)

        # 3) compute the score and Maximize it (-ve loss)
        # get the policy gradient
        agent.actor_optimizer.zero_grad()
        actor_loss = -agent.critic_local(full_critic_input).mean()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.actor_local.parameters(),0.8)
        agent.actor_optimizer.step()

        # track historical data
        self.ag_history.append(-actor_loss.data.detach())
        self.cl_history.append(critic_loss.data.detach())


    def update_targets(self):
        """soft update targets"""
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.actor_target, ddpg_agent.actor_local, self.tau)
            soft_update(ddpg_agent.critic_target, ddpg_agent.critic_local, self.tau)


class ReplayBuffer:
    def __init__(self,size):
        self.size = size
        self.memory = deque(maxlen=self.size)

    def add(self, data):
        """add into the buffer"""

        self.memory.append(data)

    def sample(self, batch_size, num_agents=2):
        """sample from the buffer"""
        # len(samples)=batch size; len(samples[0])=5; len(samples[0][0])=2
        samples = random.sample(self.memory, batch_size)

        # len(samples_byF)=5; len(samples_byF[0])=64; len(samples_byF[0][0])=2
        samples_byF = list(map(list, zip(*samples))) #list

        # cat all these 5 list input into a tensor
        # (1, 2)   -->    (1,2,3,4)
        # (3, 4)
        all_states = torch.cat(samples_byF[0]) #torch: (2 x batch_size), 24
        all_actions = torch.cat(samples_byF[1]) #torch: (2 x batch_size), 2
        all_rewards = torch.cat(samples_byF[2]) #torch: (2 x batch_size), 1
        all_dones = torch.cat(samples_byF[3]) #float torch: (2 x batch_size), 1
        all_next_states = torch.cat(samples_byF[4]) #torch: (2 x batch_size), 24

        # separate agents input by odd/even entries

        s, a, r, d, ns = ([] for l in range(5))

        # (1,2,3,4) -> (1,3)
        #              (2,4)
        for i in range(num_agents): #list of len num_agents with items:
            s.append(all_states[i::2]) #batch_size x 24
            a.append(all_actions[i::2]) #batch_size x 2
            r.append(all_rewards[i::2]) #batch_size x 1
            d.append(all_dones[i::2]) #batch_size x 1
            ns.append(all_next_states[i::2]) #batch_size x 24

        s_full = torch.stack([torch.cat(si) for si in zip(*(_s for _s in s))]) #batchsize x (num_agentsxstate_size)
        a_full = torch.stack([torch.cat(ai) for ai in zip(*(_a for _a in a))]) #batchsize x (num_agentsxaction_size)
        ns_full = torch.stack([torch.cat(ni) for ni in zip(*(_n for _n in ns))]) #batchsize x (num_agentsxstate_size)

        # test the values are instact after transposing
        assert(s_full[2,24:].mean() == s[1][2].mean())

        return (s_full, a_full, ns_full, s, a, r, d, ns)


    def __len__(self):
        return len(self.memory)
