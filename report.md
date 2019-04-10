
## Key Learnings on Implementing Multi-agent Deep Deterministic Policy Gradient (MADDPG) for Unity Environment Cooperative Multi-agent Environment 'Tennis'

### Implementation Details

#### About the environment
This environment involves 2 *independent* agents. Each of these agents controls the actions of a tennis rack. These agents are not aware of the existence of the other agent. To gain score/reward, agent must control the rack to hit the tennis ball. However, in order to achieve the passing benchmark score threshold, these agents have to learn to bounce the tennis ball back and forth to maximize reward/scores. Hence successfully trained agents should behave in a cooperative way. The goal of this training is to train these agents to cooperate to get a higher scores by bouncing the ball.

#### Multi-agent DDPG Actor-Critic Architecture
To achieve the goal score, a multi-agent DDPG (deep deterministic Policy Gradient) Actor-Critic architecture was chosen.

![Mult-agent DDPG Actor Critic Architecture](https://github.com/chihoxtra/multi_agent_rl/blob/master/maddpg.png)

Similar to single-agent Actor Critic architecture, each agent has it's own actor and critic network. The actor network takes in the current state of agent and output a recommended action for that agent. However the critic part is slightly different from ordinary single-agent DDPG. Here the critic network of each agent has *full visibility* on the environment. It not only takes in the state and action of that particular agent, but also states and actions of *all other* agents as well. Critic network has much higher visibility on what is happening while actor network can only access states information of the respective agent. The output of the critic network is, nevertheless, still the Q value estimated given an input state and an input action. And the output of the actor network is a recommended action for that particular agent.

The critic network is active only during training time. This network will be absent in running time.

#### Experience Replay and target network
Experience replay is deployed to maximize the utility of experience/trajectory gained during the interactions with environment. In this deployment, a separated experience memory buffer was deployed for each agent. The idea is to make sure there is enough randomness across sampling by avoiding coupling of experience entries across agents.

Also in order to improve the stability of learning, a target network was deployed for both actor and critic network. Parameters of these target network are *softly* updated from time to time so that the active/learning/local network will have a relatively more stable 'target' to go after.

#### Exploration using noise
Noise was added to the action recommended by the actor network as a way of exploration. It is found that the agents are very sensitive to noise. The magnitude of noise, its variation and how it is being added and decayed, if any, is very critical.

It turns out that Ornstein–Uhlenbeck seems to be the best algorithm to add noise. Noise with a relatively smaller magnitude of standard deviation was added during the learning cycle when the agents are interacting with the environment. The magnitude of the noise is *slowly* decayed. And the is a minimal magnitude that noise will never reach zero in this case.

#### Painstaking Tuning process
The biggest challenge here is synchronization. A well trained single agent is not good enough for passing the benchmark score unless the other agent is also well trained and behave correctly at the roughly the same time. During the initial stage of training, agents are not aware of the possibility of bouncing. They learned only to hit the ball falling off from the sky. And the reward is very limited. Agent only get a chance to learn bouncing when the other agent is able to hit the ball and send it to the other side of the court. Thus in a way agent has to be able to reach that 'able to bounce' stage together to make bouncing occur. To do that, both agents ideally should share a roughly similar learning cycle. That is when agent A is ready to bounce, B is also ready. And this is very difficult to achieve. Many attempts were made before a passing score were finally reached.

#### Hyper Parameters chosen:
Here are a summary of the hyper parameters used:
<table width=600>
<tr><td>Memory buffer size  </td><td> 1e6    </td></tr>     
<tr><td>Batch size </td><td>  256   </td></tr>
<tr><td>Gamma  </td><td> 0.99    </td></tr>
<tr><td>Actor LR  </td><td> 5e-4   </td></tr>
<tr><td>Critic LR  </td><td> 1e-3   </td></tr>
<tr><td>Actor Model  </td><td> (512,256)<br> Adam optimizer   </td></tr>
<tr><td>Critic Model  </td><td> (512,256)<br> Adam optimizer    </td></tr>     
<tr><td>Tau (soft update)  </td><td> 1e-3          </td></tr>           
<tr><td>Actor Network Learning Rate  </td><td>  5e-4   </td></tr>
<tr><td>Critic Network Learning Rate  </td><td>  1e-3   </td></tr>
<tr><td>update target network frequency  </td><td> 20    </td></tr>
<tr><td>Learning times per step  </td><td> 15    </td></tr>
<tr><td>OU noise initial scale  </td><td> 1.0    </td></tr>
<tr><td>noise decay factor  </td><td> 0.9995    </td></tr>
<tr><td>noise sigma  </td><td> 0.05    </td></tr>
</table>


#### The Result:
After soooooo many different trial and errors, the MADDPG agent is able to solve the environment (average score over 100 episodes across max of 2 agents > 0.5) in episode 1537.<P>

![Average Reward over 100 episodes across the max scores of 2 agents](https://github.com/chihoxtra/multi_agent_rl/blob/master/tennis_scores.png)

[Watch my trained agent(s) in action](https://youtu.be/IA2EcOPUNck)

#### Future Ideas:
- Implementation of prioritized replay for faster learning
