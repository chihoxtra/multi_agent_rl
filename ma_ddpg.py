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

BUFFER_SIZE = int(3e5)                   # size of memory replay buffer
BATCH_SIZE = 256                         # min batch size
MIN_BUFFER_SIZE = int(1e4)               # min buffer size before replay
LR_ACTOR = 5e-4                          # learning rate
LR_CRITIC = 1e-3                         # learning rate
UNITS_ACTOR = (512,256)                  # number of hidden units for actor inner layers
UNITS_CRITIC = (512,256)                 # number of hidden units for critic inner layers
GAMMA = 0.99                             # discount factor
TAU = 1e-3                               # soft network update
LEARN_EVERY = 20                          # how often to learn per step
LEARN_LOOP = 15                           # how many learning cycle per learn
#UPDATE_EVERY = 10                        # how many steps before updating the network
USE_OUNOISE = True                       # use OUnoise or else Gaussian noise
OUNOISE_SIGMA = 0.05                      # signma value for OUnoise
NOISE_WGT_INIT = 1.0                     # noise scaling weight
NOISE_WGT_PRETRAIN = NOISE_WGT_INIT      # noise scaling weight before training
NOISE_WGT_DECAY = 0.9995                 # noise decay rate per STEP
NOISE_WGT_MIN = 0.1                      # min noise scale
NOISE_DC_START = MIN_BUFFER_SIZE         # when to start noise
#NOISE_DECAY_EVERY = 2                    # noise decay step
TARGET_NOISE = False                     # target network add noise?
#NOISE_RESET_EVERY = int(1e3)            # noise reset step
#USE_BATCHNORM = True                    # use batch norm?
REWARD_SCALE = 1.0                       # use reward scaling
REWARD_NORM = False                      # use reward normalizer
#CRITIC_ACT_FORM = 1                      # [1,2,3] actions form for critic network (testing)

### PER related params, testing only
USE_PER = False                         # flag indicates use of PER
P_REPLAY_ALPHA = 0.5                     # power discount factor for samp. prob.
P_REPLAY_BETA = 0.5                      # weight adjustmnet factor
P_BETA_DELTA = 1e-4                      # beta 'increment' factor
TD_DEFAULT = 1.0                         # default TD error value
TD_EPS = 1e-4                            # minimal td value to avoid zero prob.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG:
    def __init__(self, state_size, action_size, num_agents, seed=0):
        super(MADDPG, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.use_per = USE_PER

        # critic input = obs_full + actions = 24*2+2*2=52
        self.agents_list = []
        for ai in range(self.num_agents):
            agent = DDPGAgent(state_size, action_size, num_agents,
                              UNITS_ACTOR, UNITS_CRITIC, LR_ACTOR, LR_CRITIC,
                              BUFFER_SIZE, ai, USE_PER, seed)
            self.agents_list.append(agent)

        # data structure for storing individual experience
        self.data = namedtuple("data", field_names=["states", "actions", "rewards",
                                                    "dones", "next_states"])
        # noise
        self.noise = OUNoise(action_size, sigma=OUNOISE_SIGMA)
        self.noise_scale = NOISE_WGT_PRETRAIN

        self.learn_every = LEARN_EVERY

        self.t_step = 0 # count number of steps went thru

        self.is_training = False

        # for tracking and normalization
        self.reward_history = deque(maxlen=int(1e5))
        self.noise_history = deque(maxlen=100)
        self.cl_history = deque(maxlen=100)
        self.ag_history = deque(maxlen=100)
        self.td_history = deque(maxlen=100)

        # for PER weight adjustment
        self.p_replay_beta = P_REPLAY_BETA

    def print_params(self):
        print("Number of Agents: ", self.num_agents)
        print("Device: ", device)
        print("BATCH_SIZE: ", BATCH_SIZE)
        print("LR_ACTOR: ", LR_ACTOR)
        print("LR_CRITIC: ", LR_CRITIC)
        print("REWARD_NORM: ", REWARD_NORM)
        print("REWARD_SCALE: ", REWARD_SCALE)
        print("LEARN_EVERY: ", LEARN_EVERY)
        print("LEARN_LOOP: ", LEARN_LOOP)
        print("UNITS_ACTOR: ", UNITS_ACTOR)
        print("UNITS_CRITIC: ", UNITS_CRITIC)
        print("USE_OUNOISE: ", USE_OUNOISE)
        print("NOISE_WGT_INIT: ", NOISE_WGT_INIT)
        print("OUNOISE_SIGMA: ", OUNOISE_SIGMA)
        print("NOISE_WGT_DECAY: ", NOISE_WGT_DECAY)
        print("NOISE_DC_START: ", NOISE_DC_START)
        print("NOISE_WGT_MIN: ", NOISE_WGT_MIN)
        print("TARGET_NOISE: ", TARGET_NOISE)
        print("USE_PER: ", USE_PER)
        print("P_REPLAY_ALPHA: ", P_REPLAY_ALPHA)
        print("P_REPLAY_BETA: ", P_REPLAY_BETA)
        #print("USE_BATCHNORM: ", USE_BATCHNORM)
        #print("CRITIC_ACT_FORM: ", CRITIC_ACT_FORM)
        print("")

    def add_noise(self):
        if USE_OUNOISE:
            return self.noise_scale * self.noise.sample()
        else: #Gaussian noise
            return self.noise_scale * toTorch(np.random.normal(0,1.0,self.action_size))

    def noise_reset(self):
        # reset noise mean; controlled by "main" logic in jpt notebook
        self.noise.reset()

    def acts(self, obs_all_agents):
        """(FOR EXTERNAL) get actions from all agents in the agents_list
           inputs: (np array) #num_agents x space_size (24)
           outputs: (list) len = num_agents @each tensor of action_size
        """
        obs_all_agents = toTorch(obs_all_agents) #np->tensor: num_agents x space_size (24)

        actions = []
        noise = self.add_noise().to(device)
        for ai in range(self.num_agents):
            agent = self.agents_list[ai]
            action = agent._act(obs_all_agents[ai]) + noise.to(device)
            actions.append(action) #@ action_size torch.Size([2])
            self.noise_history.append(noise.abs().mean().numpy()) #keep track noise trend

        return actions

    def _target_acts(self, obs_all_agents):
        """(called by learn() ONLY)
           get target network actions from all the agents in the MADDPG object
           inputs: (list of tensor) len of list: batchsize @ space_size (24)
           outputs: (list) len = num_agents @each tensor: batch size x action_size
        """
        target_actions = []
        if TARGET_NOISE: noise = self.add_noise().to(device)
        for ai in range(self.num_agents):
            agent = self.agents_list[ai]
            target_action = agent._target_act(obs_all_agents[ai])
            if TARGET_NOISE: target_action += noise
            target_actions.append(target_action)

        return target_actions #list of num_agents; @batchsize x action size

    def normalize_reward(self, r, eps=1e-5):
        # input r as list of tensor len=batchsize @ tensor 1
        mu = toTorch(self.reward_history).mean()
        std = toTorch(self.reward_history).std()
        r = [(ri-mu)/(std+eps) for ri in r]
        return r #list of tensor

    def step(self, data):
        states, actions, rewards, dones, next_states = data

        if REWARD_NORM: self.reward_history.extend(list(rewards))

        # add experience of EACH agent to it's corresponding memory
        for ai in range(self.num_agents):
            e = self.data(toTorch(states[ai]), #@ state_size torch.Size([24])
                          actions[ai], #@ tensor: action_size torch.Size([2])
                          toTorch(rewards[ai]).unsqueeze(-1), #@ 1 torch.Size([1])
                          toTorch(dones[ai]).unsqueeze(-1), #@ 1 torch.Size([1])
                          toTorch(next_states[ai])) #@ state_size torch.Size([24])

            if USE_PER:
                self.agents_list[ai].memory.add_tree(e, TD_DEFAULT)
            else:
                self.agents_list[ai].memory.add(e)

        # size of memory large enough...
        if len(self.agents_list[0].memory) >= MIN_BUFFER_SIZE:
            if self.is_training == False:
                print("Prefetch experience completed. Training starts! \r")
                self.print_params()
                self.is_training = True
                self.noise_scale = NOISE_WGT_INIT

            if self.t_step % LEARN_EVERY == 0:
                for _ in range(LEARN_LOOP):
                    # learn by each agent
                    for ai in range(self.num_agents): #do it agent by agent
                        agents_inputs = self.get_all_samples()
                        self.learn(agents_inputs, ai)

            #if self.t_step % UPDATE_EVERY == 0:
            #   self.soft_update() #soft update target networks params

            if USE_PER and self.p_replay_beta < 1.0: #weight increase to 1. eventually
                self.p_replay_beta = min(1.0, self.p_replay_beta + P_BETA_DELTA)

        self.t_step += 1


    def get_all_samples(self):
        """generates inputs from all agents for actor/critic network to learn"""

        s, a, r, d, ns, w, ind = ([] for l in range(7))

        # get individual agent samples from their own memory
        for ai in range(self.num_agents): #do it agent by agent
            if USE_PER:
                data = self.agents_list[ai].memory.sample_tree(BATCH_SIZE,
                                                               self.p_replay_beta,
                                                               TD_EPS)
            else:
                data = self.agents_list[ai].memory.sample(BATCH_SIZE)

            (s_a, a_a, r_a, d_a, ns_a, w_a, ind_a) = data #@list of len=batch_size
            #@ Tensor: [24] [2] [1] [1] [24] scalar, list of scalar

            # reward normalizer
            if REWARD_NORM:
                r_a = self.normalize_reward(r_a) #output tensor, batchsize x 1

            s.append(torch.stack(s_a).to(device))
            a.append(torch.stack(a_a).to(device))
            r.append(REWARD_SCALE*torch.stack(r_a).to(device))
            d.append(torch.stack(d_a).to(device))
            ns.append(torch.stack(ns_a).to(device))
            w.append(w_a)
            ind.append(ind_a)

        # cleaner implementation, checked values are consistent
        s_full = torch.stack([torch.cat(_s) for _s in zip(*(_ for _ in s))]).to(device) #batchsize x (num_agentsxstate_size)
        a_full = torch.stack([torch.cat(_a) for _a in zip(*(_ for _ in a))]).to(device) #batchsize x (num_agentsxstate_size)
        ns_full = torch.stack([torch.cat(_n) for _n in zip(*(_ for _ in ns))]).to(device) #batchsize x (num_agentsxstate_size)

        # check data consistency
        rand_ind = np.random.randint(BATCH_SIZE)
        assert(s_full[rand_ind,self.state_size:].sum() == s[1][rand_ind].sum())

        return (s_full, a_full, ns_full, s, a, r, d, ns, w, ind)


    def learn(self, data, agent_id):
        """update the critics and actors of all the agents """

        s_full, a_full, ns_full, s, a, r, d, ns, w, ind = data
        #len of num_agents, @tensor of batchsize X respective sizes

        agent = self.agents_list[agent_id]

        ####################### CRITIC LOSS #########################

        # 1) compute td target using ACTOR TARGET network in: ns->ns actions
        # method 1) use next state actions from 2 agents
        ns_a = self._target_acts(ns) #input list of len num_agents [batch_size x state_size]

        ns_a_full = torch.cat(ns_a, dim=-1).to(device) #batch_size x (action sizexnum_agents)
        assert(ns_a_full.requires_grad==False)

        #with torch.no_grad(): #TESTING ns_a_full ns_a_full ns_a[agent_id]
        q_next_target = agent.critic_target(ns_full, ns_a_full).to(device)

        td_target = r[agent_id] + GAMMA * q_next_target * (1.-d[agent_id]) #batchsize x 1

        # 2) compute td current using critic LOCAL network: in s full; a_full, a[agent_id]
        td_current = agent.critic_local(s_full, a_full).to(device)
        assert(td_current.requires_grad==True)

        # 3) compute the critic loss by minimizing td error
        if USE_PER:
            td_error = torch.abs(td_target.detach() - td_current)
            critic_loss = w[agent_id] * 0.5 * td_error.pow(2)
            critic_loss = critic_loss.mean()
        else:
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
        # method 1) use next state actions from 2 agents
        latest_actions = []
        for i in range(self.num_agents):
            latest_a = self.agents_list[i].actor_local(s[i])
            latest_actions.append(latest_a.to(device))

        # combine latest actions prediction from 2 agents -> full actions

        latest_action_full = torch.cat(latest_actions, dim=-1).to(device)
        assert(latest_action_full.requires_grad==True)

        # 2) actions (by actor local network) feed to local critic for score
        # input ful states and full actions to local critic
        # maximize Q score by gradient asscent
        agent.actor_optimizer.zero_grad() #TESTING(down) #latest_action_full, _mixed_actions, latest_actions[agent_id]
        actor_loss = -agent.critic_local(s_full, latest_action_full)
        if USE_PER: actor_loss *= w[agent_id]
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor_local.parameters(),1.0)
        agent.actor_optimizer.step()

        # update tree
        if USE_PER:
            agent.memory.update_tree(td_error.detach().numpy(), ind[agent_id],
                                     P_REPLAY_ALPHA, TD_EPS)
        # track historical data
        self.ag_history.append(-actor_loss.mean().data.detach())
        self.cl_history.append(critic_loss.mean().data.detach())
        if USE_PER: self.td_history.append(td_error.mean().detach().numpy())

        self.soft_update() #soft update target networks params

        if self.t_step >= NOISE_DC_START: #and self.t_step % NOISE_DECAY_EVERY == 0
            self.noise_scale = max(self.noise_scale * NOISE_WGT_DECAY, NOISE_WGT_MIN)


    def soft_update(self):
        """soft update targets"""
        for agent in self.agents_list:
            soft_update(agent.actor_target, agent.actor_local, TAU)
            soft_update(agent.critic_target, agent.critic_local, TAU)
