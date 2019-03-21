
## Key Learnings on Implementing Multi-agent DDPG for Unity Environment Cooperative Multi-agent Environment 'Tennis'

### Implementation Details

#### About the environment
The environment involves 2 *independent* agents. Each of these agents control a tennis rack. These agents are not aware of the existence of the other agent. To gain score/reward, agent must control the rack to hit the tennis ball. However, in order to achieve the passing benchmark score threshold, these agents have to learn to bounce the tennis ball back and forth to maximize hitting the ball and hence reward/scores. Hence successfully trained agents should behave in a cooperative way. The goal of this training is to train these agents to cooperate to get a higher scores by bouncing the ball.

#### Multi-agent DDPG Actor-Critic Architecture
To achieve the goal score, a multi-agent DDPG (deep deterministic Policy Gradient) Actor-Critic architecture was chosen.

![Mult-agent DDPG Actor Critic Architecture](https://github.com/chihoxtra/multi_agent_rl/blob/master/maddpg.png)

Similar to single-agent Actor Critic architecture, each agent has it's own actor and critic network. The actor network takes in the current state of agent and output a recommended action for that agent. However the critic part is slightly different from ordinary single-agent DDPG. Here the critic network of each agent not just tae in the state and action of that particular agent, but also states and actions of *all other* agents as well. Critic network has much higher visibility on what is happening while actor network can only access information related to its respective agent. The output of the critic network is, nevertheless, still the Q value estimated given an input state and an input action.

The critic network is active only during training time. This network will be absent in running time.

#### Experience Replay and target network
Experience replay is deployed to maximize the utility of experience/trajectory gained during the interactions with environment. In this deployment, a separated experience memory buffer was deployed for each agent. The idea is to make sure there is enough randomness across sampling by avoiding coupling of experience entries across agents.

Also in order to improve the stability of learning, a target network was deployed for both actor and critic network. Parameters of these target network are updated from time to time so that the active/learning/local network will have a relatively more stable 'target' to go for.

#### Exploration using noise
Noise was added to the action outputted by the actor network as a way of exploration. After conducting some researches, Ornstein-Uhlenbeck Noise was finally chosen. To make the 'learning cycle' across agents a bit more synchronized, a centralized noise generation process was used and the same magnitude of noise are added to each agents in each step. Scale of noise are initiated with relatively large magnitude and is gradually decayed to smaller magnitude until a minimal is reached. Noise is still added even after the minimal is reached. It is found that the environment is very sensitive to noise. A slightly change in magnitude of noise or its decay rate result in big difference in learning performance.

#### Painstaking Tuning process
The biggest challenge here is synchronization. A well trained single agent is not good enough for passing the benchmark score unless the other agent is also well trained and behave correctly at the roughly the same time. During the initial stage of training, agents are not aware of the possibility of bouncing. They learned only to hit the ball falling off from the sky. And the reward is very limited. Agent only get a chance to learn bouncing when the other agent is able to hit the ball and send it to the other side of the court. Thus in a way agent has to be able to reach that 'able to bounce' stage together to make bouncing occur. To do that, both agents ideally should share a roughly similar learning cycle. That is when agent A is ready to bounce, B is also ready. And this is very difficult to achieve. Many attempts were made before a passing score were finally reached.

#### Hyper Parameters chosen:
Here are a summary of the hyper parameters used:
<table width=600>
<tr><td>Memory buffer size  </td><td> 1e6    </td></tr>     
<tr><td>Batch size </td><td>  128   </td></tr>
<tr><td>Gamma  </td><td> 0.99    </td></tr>
<tr><td>Actor LR  </td><td> 1e-3   </td></tr>
<tr><td>Critic LR  </td><td> 1e-3   </td></tr>
<tr><td>Actor Model  </td><td> (256,128)<br> Adam optimizer   </td></tr>
<tr><td>Critic Model  </td><td> (256,128)<br> Adam optimizer    </td></tr>     
<tr><td>Tau (soft update)  </td><td> 1e-3          </td></tr>           
<tr><td>Learning Rate  </td><td>  1e-4   </td></tr>
<tr><td>update target network frequency  </td><td> 10    </td></tr>
<tr><td>Learning times per step  </td><td> 4    </td></tr>
</table>


#### The Result:
After soooooo many different trial and errors, I am glad that I am finally able to reach an average score of over 30 (per episode) across all 20 agents over last 100 episodes at around episode 100th (the reason is that the agent is able to maintain a score of over 38 for around 60+ episodes and so after reaching 100 episodes, the average score is still greater than 30). <P>
Average Reward across 20 agents across episodes<br>
![Average Reward across 20 agents across episodes](https://github.com/chihoxtra/continuous_actions_rl/blob/master/graph.png)

![Trained Agent Capture](https://github.com/chihoxtra/continuous_actions_rl/blob/master/reacher_final_20agents_38score.gif)

[Video of the trained agent](https://youtu.be/hlC8Ttg320c)

#### Future Ideas:
- Implementation of prioritized replay for faster learning
- Use PPO (actor critic style) instead of DDPG as it is known to provide even better result.
