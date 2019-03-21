"""
MADDPG class coordinates flows, inputs and outputs across agents.
Most of the actual work are done in individual agent level.
"""
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
MIN_BUFFER_SIZE = int(1e3)               # min buffer size before replay
LR_ACTOR = 1e-3                          # learning rate
LR_CRITIC = 1e-3                         # learning rate
UNITS_ACTOR = (256,128)                  # number of hidden units for actor inner layers
UNITS_CRITIC = (256,128)                 # number of hidden units for critic inner layers
GAMMA = 0.99                             # discount factor
TAU = 1e-4                               # soft network update
LEARN_EVERY = 1                          # how often to learn per step
LEARN_LOOP = 4                           # how many learning cycle per learn
UPDATE_EVERY = 4                         # how many steps before updating the network
USE_OUNOISE = True                       # use OUnoise or else Gaussian noise
NOISE_WGT_INIT = 5.0                     # noise scaling weight
NOISE_WGT_DECAY = 0.9998                 # noise decay rate per STEP
NOISE_WGT_MIN = 0.1                      # min noise scale
NOISE_DC_START = MIN_BUFFER_SIZE         # when to start noise
NOISE_DECAY_EVERY = int(1e3)             # noise decay step
NOISE_RESET_EVERY = int(1e4)             # noise reset step
USE_BATCHNORM = False                    # use batch norm?

### PER related params, testing only
USE_PER = True                           # flag indicates use of PER
P_REPLAY_ALPHA = 0.7                     # power discount factor for samp. prob.
P_REPLAY_BETA = 0.6                      # weight adjustmnet factor
P_BETA_DELTA = 1e-4                      # beta 'increment' factor
TD_DEFAULT = 1.0                         # default TD error value
TD_EPS = 1e-3                            # minimal td value to avoid zero prob.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG:
    def __init__(self, state_size, action_size, num_agents, seed=0):
        super(MADDPG, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents

        # critic input = obs_full + actions = 14+2+2+2=20
        self.agents_list = [DDPGAgent(state_size, action_size, num_agents,
                                      UNITS_ACTOR, UNITS_CRITIC, LR_ACTOR, LR_CRITIC,
                                      BUFFER_SIZE, USE_PER, seed)
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

        # for PER
        self.p_replay_beta = P_REPLAY_BETA


    def add_noise(self):
        if not USE_OUNOISE:
            noise = self.noise_scale * toTorch(np.random.normal(0,1.0,self.action_size))
        else:
            noise = self.noise_scale * self.noise.sample()
        return noise

    def noise_reset(self):
        self.noise.reset()

    def acts(self, obs_all_agents):
        """(external callable) get actions from all agents in the agents_list
           inputs: (np array) #num_agents x space_size (24)
           outputs: (list) len = num_agents @each tensor of action_size
        """
        obs_all_agents = toTorch(obs_all_agents)
        actions = []
        with torch.no_grad():
            for i in range(self.num_agents):
                agent = self.agents_list[i]
                noise = self.add_noise()
                action = agent._act(obs_all_agents[i], USE_BATCHNORM) + noise.to(device)
                actions.append(action)

                self.noise_history.append(noise.abs().mean())

        return actions

    def _target_acts(self, obs_all_agents):
        """get target network actions from all the agents in the MADDPG object
           inputs: (list of tensor) len of list: num_agents @ space_size (24)
           outputs: (list) len = num_agents @each tensor: batch size x action_size
        """
        target_actions = []
        with torch.no_grad():
            for i in range(self.num_agents):
                agent = self.agents_list[i]
                target_action = agent._target_act(obs_all_agents[i], USE_BATCHNORM)
                target_actions.append(target_action)

        return target_actions #list of num_agents; @batchsize x action size


    def step(self, data):
        states, actions, rewards, dones, next_states = data

        # add experience of each agent to it's corresponding memory
        for ai in range(self.num_agents):
            e = self.data(toTorch(states[ai]), #num_agent x state_size
                          actions[ai], #tensor: #num_agent x actions size
                          toTorch(rewards[ai]).unsqueeze(-1), #num_agent x 1
                          toTorch(1.*np.array(dones[ai])).unsqueeze(-1), #num_agent x 1
                          toTorch(next_states[ai])) #num_agent x state_size

            if USE_PER:
                self.agents_list[ai].memory.add_tree(e, TD_DEFAULT)
            else:
                self.agents_list[ai].memory.add(e)

        # size of memory large enough...
        if len(self.agents_list[0].memory) >= MIN_BUFFER_SIZE:
        #if self.t_step >= MIN_BUFFER_SIZE:
            if self.is_training == False:
                print("Prefetch completed. Training starts! \r")
                print("Number of Agents: ", self.num_agents)
                print("Device: ", device)
                self.is_training = True

            if self.t_step % LEARN_EVERY == 0:
                for _ in range(LEARN_LOOP):
                    # learn by each agent
                    for ai in range(self.num_agents): #do it agent by agent
                        agents_inputs = self.get_all_samples()
                        self.learn(agents_inputs, ai)

            if self.t_step >= NOISE_DC_START and self.t_step % NOISE_DECAY_EVERY:
                self.noise_scale = max(self.noise_scale * NOISE_WGT_DECAY, NOISE_WGT_MIN)

            if self.t_step % NOISE_RESET_EVERY == 0:
                self.noise_reset()

            if self.t_step % UPDATE_EVERY == 0: #sync network params values
                self.update_targets()

            if USE_PER and self.p_replay_beta < 1.0: #weight adjustment
                self.p_replay_beta = min(1.0, self.p_replay_beta + P_BETA_DELTA)

        self.t_step += 1

    def get_all_samples(self):
        """generates inputs from all agents for actor/critic network to learn"""

        # initialize variables
        s_full1 = torch.zeros(BATCH_SIZE, self.num_agents*self.state_size)
        ns_full1 = torch.zeros(BATCH_SIZE, self.num_agents*self.state_size)
        a_full1 = torch.zeros(BATCH_SIZE, self.num_agents*self.action_size)

        s, a, r, d, ns, w, ind = ([] for l in range(7))

        # get individual agent samples from their own memory
        for ai in range(self.num_agents): #do it agent by agent
            if USE_PER:
                data = self.agents_list[ai].memory.sample_tree(BATCH_SIZE,
                                                               self.p_replay_beta,
                                                               TD_EPS)
            else:
                data = self.agents_list[ai].memory.sample(BATCH_SIZE)

            (s_a, a_a, r_a, d_a, ns_a, w_a, ind_a) = data

            s.append(torch.stack(s_a))
            a.append(torch.stack(a_a))
            r.append(torch.stack(r_a))
            d.append(torch.stack(d_a))
            ns.append(torch.stack(ns_a))
            w.append(w_a)
            ind.append(ind_a)

        # prepare full obs and actions after collection
        for ai in range(self.num_agents): #do it agent by agent
            for i in range(BATCH_SIZE):
                #assert(len(s)==ai+1)
                s_full1[i,ai*self.state_size:(ai+1)*self.state_size] = s[ai][i]
                ns_full1[i,ai*self.state_size:(ai+1)*self.state_size] = ns[ai][i]
                a_full1[i,ai*self.action_size:(ai+1)*self.action_size] = a[ai][i]

        # cleaner implementation, checked values are consistent
        s_full = torch.stack([torch.cat(_s) for _s in zip(*(_ for _ in s))]).to(device) #batchsize x (num_agentsxstate_size)
        a_full = torch.stack([torch.cat(_a) for _a in zip(*(_ for _ in a))]).to(device) #batchsize x (num_agentsxstate_size)
        ns_full = torch.stack([torch.cat(_n) for _n in zip(*(_ for _ in ns))]).to(device) #batchsize x (num_agentsxstate_size)
        assert(s_full.sum()==s_full1.sum())
        assert(a_full.sum()==a_full1.sum())
        assert(ns_full.sum()==ns_full1.sum())

        rand_ind = np.random.randint(BATCH_SIZE)
        assert(s_full[rand_ind,self.state_size:].sum() == s[1][rand_ind].sum())

        return (s_full, a_full, ns_full, s, a, r, d, ns, w, ind)


    def learn(self, data, agent_id):
        """update the critics and actors of all the agents """

        s_full, a_full, ns_full, s, a, r, d, ns, w, ind = data

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
        # 1) compute td target using ACTOR TARGET network in: ns->ns actions
        ns_a = self._target_acts(ns) #input list of len num_agents [batch_size x state_size]
        ns_a_full = torch.cat(ns_a, dim=-1).to(device) #batch_size x (action sizexnum_agents)
        assert(ns_a_full.requires_grad==False)

        ### TESTING ### get action and ns action for this agent
        ai_a_full = torch.cat((ns_a[agent_id],a[agent_id]),dim=-1).to(device)

        with torch.no_grad():
            q_next_target = agent.critic_target(ns_full, ai_a_full).to(device)

        td_target = r[agent_id] + GAMMA * q_next_target * (1.-d[agent_id])

        # 2) compute td current using critic LOCAL network: in s full, a full, ns ai
        td_current = agent.critic_local(s_full, a_full).to(device) #req_grad=false, false
        assert(td_current.requires_grad==True)

        # 3) compute the critic loss by minimizing td error
        td_error = torch.abs(td_target.detach() - td_current)
        critic_loss = w[agent_id] * td_error.pow(2).mean()

        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 0.5)
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
        latest_actions = []
        for i in range(self.num_agents):
            latest_a = self.agents_list[i].actor_local(s[i])
            if i != agent_id: # not the learning target
                latest_a = latest_a.detach()
            latest_actions.append(latest_a)

        # combine latest prediction from 2 agents to form full actions
        latest_action_full = torch.cat(latest_actions, dim=-1).to(device)

        # actions has to be differtiable so that parameters can change
        # to produce an action that produce a higher critic score
        assert(latest_action_full.requires_grad==True)

        ### TESTING # get action and latest predicted action for this agent
        latest_ai_a_full = torch.cat((latest_actions[agent_id],a[agent_id]),dim=-1).to(device)
        assert(latest_ai_a_full.requires_grad==True)

        # 2) actions (by actor local network) feed to local critic for score
        # input ful states and full actions to local critic
        # maximize Q score by gradient asscent
        agent.actor_optimizer.zero_grad()
        actor_loss = w[agent_id] * -agent.critic_local(s_full, latest_ai_a_full).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor_local.parameters(),0.5)
        agent.actor_optimizer.step()

        # update tree
        if USE_PER:
            agent.memory.update_tree(td_error.detach().numpy(), ind[agent_id],
                                     P_REPLAY_ALPHA, TD_EPS)

        # track historical data
        self.ag_history.append(-actor_loss.data.detach()*1.e4)
        self.cl_history.append(critic_loss.data.detach()*1.e4)


    def update_targets(self):
        """soft update targets"""
        for agent in self.agents_list:
            soft_update(agent.actor_target, agent.actor_local, self.tau)
            soft_update(agent.critic_target, agent.critic_local, self.tau)
