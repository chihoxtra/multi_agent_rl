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
# add OU noise for exploration
from OUNoise import OUNoise
from utilities import toTorch, soft_update

BUFFER_SIZE = int(1e6)                   # size of memory replay buffer
BATCH_SIZE = 128                         # min batch size
MIN_BUFFER_SIZE = BATCH_SIZE               # min buffer size before replay
LR_ACTOR = 1e-5                          # learning rate
LR_CRITIC = 1e-5                         # learning rate
UNITS_ACTOR = (256,128)                  # number of hidden units for actor inner layers
UNITS_CRITIC = (256,128)                 # number of hidden units for critic inner layers
GAMMA = 0.99                             # discount factor
TAU = 1e-4                               # soft network update
LEARN_EVERY = 1                          # how often to learn per step
LEARN_LOOP = 2                           # how many learning cycle per learn
UPDATE_EVERY = 4                         # how many steps before updating the network
USE_OUNOISE = False                       # use OUnoise or else Gaussian noise
NOISE_WGT_INIT = 5.0                     # noise scaling weight
NOISE_WGT_DECAY = 0.9995                 # noise decay rate per STEP
NOISE_WGT_MIN = 0.05                     # min noise scale
NOISE_DC_START = int(2e3)                # when to start noise
NOISE_DECAY_EVERY = 100                  # noise decay step
NOISE_RESET_EVERY = MIN_BUFFER_SIZE      # noise reset step

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG:
    def __init__(self, state_size, action_size, num_agents, seed=0):
        super(MADDPG, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.state_size = state_size
        self.action_size = action_size
        self.min_buffer = MIN_BUFFER_SIZE

        self.num_agents = num_agents

        # critic input = obs_full + actions = 14+2+2+2=20
        self.maddpg_agent = [DDPGAgent(state_size, action_size, num_agents,
                                       UNITS_ACTOR, UNITS_CRITIC, LR_ACTOR, LR_CRITIC, seed)
                             for _ in range(num_agents)]

        # replay buffer
        self.memory = ReplayBuffer(BUFFER_SIZE, num_agents, state_size, action_size,
                                   BATCH_SIZE)

        # data structure for storing individual experience
        self.data = namedtuple("data", field_names=["states", "actions", "rewards",
                                                    "dones", "next_states"])

        # noise
        self.noise = OUNoise(action_size)
        self.noise_scale = NOISE_WGT_INIT

        self.gamma = GAMMA
        self.tau = TAU

        self.t_step = 0 # count number of steps went thru

        self.is_training = False

        # for tracking
        self.noise_history = deque(maxlen=100)
        self.cl_history = deque(maxlen=100)
        self.ag_history = deque(maxlen=100)


    def _toTorch(self, s, dtype=torch.float32):
        return torch.tensor(s, dtype=dtype, device=device)

    def add_noise(self):
        if not USE_OUNOISE:
            noise = self.noise_scale * toTorch(np.random.normal(0,1.0,self.action_size))
        else:
            noise = self.noise_scale * self.noise.sample()
        return noise

    def noise_reset(self):
        self.noise.reset()

    def acts(self, obs_all_agents):
        """get actions from all agents in the MADDPG object
           inputs: (array) #num_agents x space_size (24)
           outputs: (list) len = num_agents @each tensor of action_size
        """
        obs_all_agents = toTorch(obs_all_agents) #num_agents x space_size (24)

        actions = []
        with torch.no_grad():

            for i in range(self.num_agents):
                agent = self.maddpg_agent[i]
                noise = self.add_noise()
                action = agent._act(obs_all_agents[i,:]) + noise.to(device)
                actions.append(action)

                self.noise_history.append(noise.abs().mean())

        return actions

    def target_acts(self, obs_all_agents):
        """get target network actions from all the agents in the MADDPG object
           inputs: (array) #num_agents x space_size (24)
           outputs: (list) len = num_agents @each tensor of action_size
        """
        target_actions = []
        with torch.no_grad():
            for i in range(self.num_agents):
                agent = self.maddpg_agent[i]
                target_action = agent._target_act(obs_all_agents[i], self.noise_scale)
                target_actions.append(target_action)

        return target_actions


    def step(self, data):
        states, actions, rewards, dones, next_states = data

        # add experience to memory
        e = self.data(toTorch(states), #num_agent x state_size
                      actions, #tensor: #num_agent x actions size
                      toTorch(rewards).unsqueeze(-1), #num_agent x 1
                      toTorch(1.*np.array(dones)).unsqueeze(-1), #num_agent x 1
                      toTorch(next_states)) #num_agent x state_size

        self.memory.add(e)

        # size of memory large enough...
        if len(self.memory) >= MIN_BUFFER_SIZE:
            if self.is_training == False:
                print("Prefetch completed. Training starts! \r")
                print("Number of Agents: ", self.num_agents)
                print("Device: ", device)
                self.is_training = True

            if self.t_step % LEARN_EVERY == 0:
                for _ in range(LEARN_LOOP):

                    for agent_id in range(self.num_agents): #do it agent by agent

                        agent_inputs = self.memory.sample(BATCH_SIZE) #by fields

                        self.learn(agent_inputs, agent_id)

            if self.t_step >= NOISE_DC_START and self.t_step % NOISE_DECAY_EVERY:
                self.noise_scale = max(self.noise_scale * NOISE_WGT_DECAY, NOISE_WGT_MIN)

            if self.t_step % NOISE_RESET_EVERY == 0:
                self.noise_reset()

            if self.t_step % UPDATE_EVERY == 0: #sync network params values
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
        ns_actions = self.target_acts(ns) #input list of len num_agents [batch_size x state_size]
        ns_full_actions = torch.cat(ns_actions, dim=1).to(device) #batch_size x (action sizexnum_agents)

        #ns_critic_input = torch.cat((ns_full,ns_full_actions), dim=-1).to(device)
        # batch size x (num_agent x (state_size + action size))
        q_next_target = agent.critic_target(ns_full, ns_full_actions).to(device)

        td_target = r[agent_id] + GAMMA * q_next_target * (1.-d[agent_id])

        td_current = agent.critic_local(s_full, a_full).to(device) #req_grad=false, false
        assert(td_current.requires_grad==True)

        # 3) compute the critic loss
        critic_loss = F.mse_loss(td_current, td_target.detach())

        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1.0)
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
        latest_a_full = []
        for i in range(self.num_agents):
            action = self.maddpg_agent[i].actor_local(s[i])
            if i != agent_id: # not the learning target
                action = action.detach()
            latest_a_full.append(action)

        # combine latest prediction from 2 agents to form full actions
        latest_a_full = torch.cat(latest_a_full, dim=1).to(device)

        # actions has to be differtiable so that parameters can change
        # to produce an action that produce a higher critic score
        assert(latest_a_full.requires_grad==True)

        # 2) actions (by actor local network) feed to local critic for score
        # input ful states and full actions to local critic
        # maximize Q score by gradient asscent

        agent.actor_optimizer.zero_grad()
        actor_loss = -agent.critic_local(s_full, latest_a_full).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor_local.parameters(),0.8)
        agent.actor_optimizer.step()

        # track historical data
        self.ag_history.append(-actor_loss.data.detach()*1.e4)
        self.cl_history.append(critic_loss.data.detach()*1.e4)


    def update_targets(self):
        """soft update targets"""
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.actor_target, ddpg_agent.actor_local, self.tau)
            soft_update(ddpg_agent.critic_target, ddpg_agent.critic_local, self.tau)


class ReplayBuffer:
    def __init__(self, size, num_agents, state_size, action_size, batch_size):
        self.size = size
        self.memory = deque(maxlen=self.size)
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size

    def add(self, data):
        """add into the buffer"""

        self.memory.append(data)

    def sample(self, batch_size, num_agents=2):
        """sample from the buffer"""
        # list(len=batchsize) of named tuples (of 5 fields)
        #  [(s1,a1,r1,d1,ns1),
        #   (s2,a2,r2,d2,ns2), # batch_size = 3
        #   (s3,a3,r3,d3,ns3)]
        #samples = random.sample(self.memory, batch_size)
        sample_ind = np.random.choice(len(self.memory), batch_size)
        #print(samples[0].states.shape) #2,24

        # get the selected experiences: avoid using mid list indexing
        s_s, a_s, r_s, d_s, ns_s = ([] for l in range(5))

        i = 0
        while i < batch_size: #while loop is faster
            self.memory.rotate(-sample_ind[i])
            e = self.memory[0]
            s_s.append(e.states)
            a_s.append(e.actions)
            r_s.append(e.rewards)
            d_s.append(e.dones)
            ns_s.append(e.next_states)
            self.memory.rotate(sample_ind[i])
            i += 1

        #print(len(s))

        # list of 5 fields columns. Each columns have len=batchsize
        # len(samples_byF)=5; len(samples_byF[0])=64; len(samples_byF[0][0])=2
        #  [(s1,s2,s3),
        #   (a1,a2,a3),
        #   (r1,r2,r3),    # batch_size = 3
        #   (d1,d2,d3),
        #   (ns1,ns2,ns3)]
        #samples_by_fields = list(map(list, zip(*samples)))

        s = list(map(torch.stack, zip(*s_s)))
        a = list(map(torch.stack, zip(*a_s)))
        r = list(map(torch.stack, zip(*r_s)))
        d = list(map(torch.stack, zip(*d_s)))
        ns = list(map(torch.stack, zip(*ns_s)))

        #assert(s[1][32].sum() == self.memory[sample_ind[32]].states[1].sum())

        s_full = torch.zeros(self.batch_size, self.num_agents*self.state_size)
        ns_full = torch.zeros(self.batch_size, self.num_agents*self.state_size)
        a_full = torch.zeros(self.batch_size, self.num_agents*self.action_size)
        for i in range(self.batch_size):
            for ai in range(self.num_agents):
                s_full[i,ai*self.state_size:(ai+1)*self.state_size] = s[ai][i]
                ns_full[i,ai*self.state_size:(ai+1)*self.state_size] = ns[ai][i]
                a_full[i,ai*self.action_size:(ai+1)*self.action_size] = a[ai][i]

        #s_full = torch.stack([torch.cat(si) for si in zip(*(s[0],s[1]))]).to(device) #batchsize x (num_agentsxstate_size)
        #a_full = torch.stack([torch.cat(ai) for ai in zip(*(_a for _a in a))]).to(device) #batchsize x (num_agentsxaction_size)
        #ns_full = torch.stack([torch.cat(ni) for ni in zip(*(_n for _n in ns))]).to(device) #batchsize x (num_agentsxstate_size)

        # test the values are instact after transposing
        #assert(s_full[5,24:].sum() == s[1][5].sum())

        return (s_full, a_full, ns_full, s, a, r, d, ns)


    def __len__(self):
        return len(self.memory)
