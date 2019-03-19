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

BUFFER_SIZE = int(1e4)                   # size of memory replay buffer
BATCH_SIZE = 128                         # min batch size
MIN_BUFFER_SIZE = int(1e3)               # min buffer size before replay
LR_ACTOR = 1e-3                          # learning rate
LR_CRITIC = 1e-3                         # learning rate
UNITS_ACTOR = (256,128)                  # number of hidden units for actor inner layers
UNITS_CRITIC = (256,128)                 # number of hidden units for critic inner layers
GAMMA = 0.99                             # discount factor
TAU = 1e-4                               # soft network update
LEARN_EVERY = 1                          # how often to learn per step
LEARN_LOOP = 6                           # how many learning cycle per learn
UPDATE_EVERY = 4                         # how many steps before updating the network
USE_OUNOISE = False                      # use OUnoise or else Gaussian noise
NOISE_WGT_INIT = 5.0                     # noise scaling weight
NOISE_WGT_DECAY = 0.9999                 # noise decay rate per STEP
NOISE_WGT_MIN = 0.1                      # min noise scale
NOISE_DC_START = MIN_BUFFER_SIZE         # when to start noise
NOISE_DECAY_EVERY = 150                  # noise decay step
NOISE_RESET_EVERY = int(1e3)             # noise reset step

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
        self.agents_list = [DDPGAgent(state_size, action_size, num_agents,
                                      UNITS_ACTOR, UNITS_CRITIC, LR_ACTOR, LR_CRITIC,
                                      BUFFER_SIZE, BATCH_SIZE, seed)
                            for _ in range(num_agents)]

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
                agent = self.agents_list[i]
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
                agent = self.agents_list[i]
                target_action = agent._target_act(obs_all_agents[i], self.noise_scale)
                target_actions.append(target_action)

        return target_actions


    def step(self, data):
        states, actions, rewards, dones, next_states = data

        # add experience of each agent to it's corresponding memory
        for ai in range(self.num_agents):
            e = self.data(toTorch(states[ai]), #num_agent x state_size
                          actions[ai], #tensor: #num_agent x actions size
                          toTorch(rewards[ai]).unsqueeze(-1), #num_agent x 1
                          toTorch(1.*np.array(dones[ai])).unsqueeze(-1), #num_agent x 1
                          toTorch(next_states[ai])) #num_agent x state_size

            self.agents_list[ai].memory.add(e)

        # size of memory large enough...
        if len(self.agents_list[0].memory) >= MIN_BUFFER_SIZE:
            if self.is_training == False:
                print("Prefetch completed. Training starts! \r")
                print("Number of Agents: ", self.num_agents)
                print("Device: ", device)
                self.is_training = True

            if self.t_step % LEARN_EVERY == 0:
                for _ in range(LEARN_LOOP):

                    # learn by each agent
                    for ai in range(self.num_agents): #do it agent by agent
                        agents_inputs = self.get_samples()
                        self.learn(agents_inputs, ai)

            if self.t_step >= NOISE_DC_START and self.t_step % NOISE_DECAY_EVERY:
                self.noise_scale = max(self.noise_scale * NOISE_WGT_DECAY, NOISE_WGT_MIN)

            if self.t_step % NOISE_RESET_EVERY == 0:
                self.noise_reset()

            if self.t_step % UPDATE_EVERY == 0: #sync network params values
                self.update_targets()

        self.t_step += 1

    def get_samples(self):
        """generates inputs from all agents for actor/critic network"""

        # initialize variables
        s_full = torch.zeros(BATCH_SIZE, self.num_agents*self.state_size)
        ns_full = torch.zeros(BATCH_SIZE, self.num_agents*self.state_size)
        a_full = torch.zeros(BATCH_SIZE, self.num_agents*self.action_size)

        s, a, r, d, ns = ([] for l in range(5))

        # get individual agent samples from their own memory
        for ai in range(self.num_agents): #do it agent by agent
            (s_a, a_a, r_a, d_a, ns_a) = self.agents_list[ai].memory.sample(BATCH_SIZE)

            s.append(torch.stack(s_a))
            a.append(torch.stack(a_a))
            r.append(torch.stack(r_a))
            d.append(torch.stack(d_a))
            ns.append(torch.stack(ns_a))

            # prepare full obs and actions after collection
            for i in range(BATCH_SIZE):
                s_full[i,ai*self.state_size:(ai+1)*self.state_size] = s[ai][i]
                ns_full[i,ai*self.state_size:(ai+1)*self.state_size] = ns[ai][i]
                a_full[i,ai*self.action_size:(ai+1)*self.action_size] = a[ai][i]

        assert(s_full[12,self.state_size:].sum() == s[1][12].sum())

        return (s_full, a_full , ns_full, s, a, r, d, ns)

    def learn(self, agents_inputs, agent_id):
        """update the critics and actors of all the agents """

        s_full, a_full , ns_full, s, a, r, d, ns = agents_inputs

        agent = self.agents_list[agent_id]

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
        with torch.no_grad():
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
            action = self.agents_list[i].actor_local(s[i])
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
        for ddpg_agent in self.agents_list:
            soft_update(ddpg_agent.actor_target, ddpg_agent.actor_local, self.tau)
            soft_update(ddpg_agent.critic_target, ddpg_agent.critic_local, self.tau)
