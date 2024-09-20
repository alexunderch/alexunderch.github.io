# Multi-agent pathfinding (MAPF) algorithms

## MAPF problem statement 
The MAPF problem consists of a set of agents, obstacles, and target points in a given map. Consider $$n$$ agents, operating on an undirected graph $$G=(\mathcal{V}, \mathcal{E})$$, in a discretised timeline $$T = \{0, 1, 2, \ldots\}$$. The graph confines the learning environment and defines its map. Initially, at $$t = 0$$, the agents located at their vertices $$\text{Starts} = \{\text{start}_1, \ldots, \text{start}_n\}$$, while their goal vertices are defined as $$\text{Goals} = \{\text{goal}_1, \ldots, \text{goal}_n\}$$. There could also be present some obstacles, dynamic $$\mathfrak{O}_{\text{dyn}}$$ and static $$\mathfrak{O}_{\text{static}}$$. At each following timestep, an agent can either wait in its current vertex or move in adjastent one, each action takes exactly one time step. The individual plan $$\{\text{pl}_i\}_{i=1}^n$$ could be denoted as a sequence of actions performed at consecutive time steps that brings the $$i^{\text{th}}$$ agent to traverse from the $$\text{start}_i$$ to the $$\text{goal}_i$$. The agents and dynamic obstacles are alowed to move from $$v_i \in \mathcal{V}$$ to $$v_j \in \mathcal{V}$$ through $$e_{ij} \in \mathcal{E}$$.


Every two individual plans are said to contain a vertex conflict if the agents following them occupy the same graph vertex at the timestep. Similarly, an edge conflict occurs when the agents traverse the same edge in the opposite directions at the same timestep. The other issue is a "deadlock" --- when an agent, achieved its goal, blocks plans for the other agents.  The objective of the MAPF a joint plan $$\pi = \{\text{pl}_1, \ldots, \text{pl}_n\}$$, such that any pair of them is conflict-free, and there is no deadlock.  

### Partially observable MAPF 

Partially observable MAPF (PO-MAPF) is an extension of the MAPF problem where the graph $$G$$ in not fully known for the agents, but instead, they have an access to the observation function $$O(v, t)$$ that allows to obtain information about all the necessary information from the $$k$$-hop neighbourhood of the vertex $$v$$ that the agent occupies at the time step $$t$$. Futhermore, the MAPF problem fully transitions to sequential decision making, aiming to find a policy $$\pi$$ hat maps the history of observations onto actions. Additionally, to ensure that a finite-time plans exist, the time is limited to some $$T_{\text{max}}$$.

## Connections to Deep Reinforcement Learning (DRL)

The problem of multi-agent pathfinding could be formalised as solving a stochastic (Markov) game, while PO-MAPF encases a partially observed stochastic game (POSG). The latter could be defined as a tuple $$\begin{align}
    G = \langle \mathcal{I}, \mathcal{S}, \{\mathcal{A}\}_{i=1}^n, P,  \rho_0, \{R^i\}_{i=1}^n, \{\mathcal{O}^i\}_{i=1}^n, \{O^i\}_{i=1}^n, \gamma \rangle,
\end{align}$$ where $$\mathcal{I}$$ is a finite set of $$n$$ agents, $$\mathcal{S}$$ is a finite set of the environment states, $$\mathbf{A} = \prod_{i \in \mathcal{I}} \mathcal{A}^i$$ --- a finite set of joint actions, $$\mathbf{O} = \prod_{i \in \mathcal{I}} \mathcal{O}^i$$  --- a joint observation space,  $$O^i: \mathcal{S} \times \mathbf{A} \times \mathcal{O}^i \rightarrow \Delta(\mathcal{O}^i)$$ --- an observation function of the $$i^{\text{th}}$$ agent, $$\rho_0 \in \Delta(\mathcal{S})$$ --- initial state distribution, $$\gamma \in (0, 1)$$ is a discount factor.

The game proceeds as follows. Starting from an initial state $$s_0 \sim \rho$$, at time $$t$$, an agent $$i \in \mathcal{I}$$ receives a private observation $$o_t^i$$ governed by the observation function, $$O^i(o_t^i \vert s_{t+1}, a_t)$$, and chooses an action $$a^i_t \in \mathcal{A}^i$$ that gets simultaneously executed with all other agents.  Given the state $$s_t$$ and agents' joint action $$\mathbf{a}_t = \{a^i_t\}$$, the environment transitions to the next state according to the state transition function $$P(s_{t+1}\vert s_t, \mathbf{a}_t): \ \mathcal{S} \times \mathcal{A} \times S \rightarrow \Delta(S)$$ and gets reward according to the reward function  $$r_t^i = R^i(s_{t+1}, \mathbf{a}_t, s_{t}): \mathcal{S} \times \mathbf{A} \times \mathcal{S} \rightarrow \mathbb{R}$$. Each agent aims to find a behavioural policy $$\pi^i(a^i_t\vert h^i_t) \in \Pi^i: \mathcal{T} \rightarrow \Delta(\mathcal{A}^i)$$ conditioned on action-observation history $$h_t = (s_0, \mathbf{o}_0, \mathbf{a}_0, s_1,  \mathbf{o}_1, \mathbf{a}_1, \ldots, s_t, \mathbf{o}_t) \in \mathcal{T} = (\mathcal{S} \times \mathbf{A} \times \mathbf{O})^*$$ that can guide the agent to take sequential actions such that the discounted cumulative reward of the joint policy $$\pi = (\pi^i, \pi^{-i})$$ is maximised:

$$
\begin{align}
    \pi^i = \arg\max_{\pi^i}\mathbb{E}_{\pi, P} \left[ \sum_{t=0}^{T_{\text{max}}} \gamma^t r_{t+1} \vert  s_0 \sim \rho \right]
\end{align}
$$

As the definition is quite general, let's revisit some special cases:
1. if the observation space for each agent is the same as the state space of the game and the observation function is an identity function, then it appears as *fully observable* stochastic game, or just stochastic game
2. if agents learn and act in decentralised manner, i.e. without communication, the reward function $$R(s_{t+1}, a^i_t, s_{t})$$ is shared across the agents, so agent can observe only their own actions. During execution, the agents are assumed to act based on their individual observations only and no additional communication is assumed.  

The problem statement using reinforcement learning (RL) is simialt to the one in MAPF. However, in RL, the transition model and reward function of the POSG (which represents the learning environment) are usually unknown. Therefore, the only way of finding or estimating an optimal policy that will allow the agent to (near-optimally) behave in this environment is to interact with the environment and gather some info regarding its "dynamics", while in MAPF the agents plan ahead of execution.

## Information access

There is a common assumption that there is a shared between all the agents controller, allowing them to share the information about their positions during the planning. However, for many practical applications, it's very expensive or difficult make centralisation possible due to resource constrains. Thus, there is necessity for designing the agents in a way to make their decisions based only on their own observations or observation history. This approach is called decetralised MAPF.  

## Metrics

The existing works related to solving MAPF problems evaluates the performance by two major criteria -- success rate and the primary performance indicators: sum-of-costs, makespan, throughput. While these metrics allow to evaluate the algorithms at some particular instance, it might be difficult to get a high-level conclusion about the performance of the algorithms.

1. Base metrics:

$$
\begin{gather}
    \text{SoC} = \sum_{1 \leq i \leq n} |\pi^i|\\
    \text{makespan} = \max_{1 \leq i \leq n} |\pi^i|\\
    \text{throughput} = \frac{\text{\#solved sub-goals}}{\text{\#steps}}
\end{gather}
$$

2. Performance metrics:

$$
\begin{gather}
    \text{Performance}_\text{MAPF} = \begin{cases} \text{SoC}_\text{best}/\text{SoC} \\ 0 \text{ if not solved}\end{cases}\\
    \text{Performance}_\text{LMAPF} = \text{throughput} / \text{throughput}_\text{best}
\end{gather}
$$

## MAPF approaches

| Algorithm                                | Decentralised in training/planning phase | Partial Observability Supports MAPF | Supports LifeLong MAPF | No Communication | Parameter Sharing | Model-Based 
|------------------------------------------|---------------|-----------------------------------|---------------------|-------------------------|-----------------|-------------|
| [MAMBA](#mamba)          | &#x2717;           | &#x2713;                   | &#x2713;                          | &#x2713;           | &#x2713;                    | &#x2717;                     | &#x2713;              | &#x2713;               | &#x2713;         | 
| [QPLEX](#qplex)               | &#x2717;           | &#x2713;                   | &#x2713;                          | &#x2713;           | &#x2713;                    | &#x2717;                     | &#x2713;              | &#x2713;               | &#x2717;         | 
| [IQL](#iql)                  | &#x2717;           | &#x2713;                   | &#x2713;                          | &#x2713;           | &#x2713;                    | &#x2717;                     | &#x2713;              | &#x2713;               | &#x2717;         | 
| [VDN](#vdn)              | &#x2717;           | &#x2713;                   | &#x2713;                          | &#x2713;           | &#x2713;                    | &#x2717;                     | &#x2713;              | &#x2713;               | &#x2717;         | 
| [QMIX](#qmix)          | &#x2717;           | &#x2713;                   | &#x2713;                          | &#x2713;           | &#x2713;                    | &#x2717;                     | &#x2713;              | &#x2713;               | &#x2717;         | 
| [Follower](#follower)        | &#x2713;           | &#x2713;                   | &#x2713;                          | &#x2717;           | &#x2713;                    | &#x2717;                     | &#x2713;              | &#x2713;               | &#x2717;         | 
| [MATS-LP](#mats-lp) | &#x2713;           | &#x2713;                   | &#x2713;                          | &#x2717;           | &#x2713;                    | &#x2717;                     | &#x2713;              | &#x2713;               | &#x2713;         | 
| [Switcher](#switcher)       | &#x2713;           | &#x2713;                   | &#x2713;                          | &#x2717;           | &#x2713;                    | &#x2713;                     | &#x2713;              | &#x2713;               | &#x2717;         | 
| [SCRIMP](#scrimp)             | &#x2717;           | &#x2713;                   | &#x2717;                          | &#x2713;           | &#x2717;                    | &#x2717;                     | &#x2717;              | &#x2713;               | &#x2717;         | 
| [DCC](#dcc)                | &#x2717;           | &#x2713;                   | &#x2713;                          | &#x2713;           | &#x2717;                    | &#x2717;                     | &#x2717;              | &#x2713;               | &#x2717;         |
| [DHC](#dhc)                | &#x2717;           | &#x2713;                   | &#x2713;                          | &#x2713;           | &#x2717;                    | &#x2717;                     | &#x2717;              | &#x2713;               | &#x2717;         | 
| [RHCR](#rhcr)                   | &#x2717;           | &#x2717;                   | &#x2717;                          | &#x2717;           | &#x2717;                     | -                |



### MAMBA

| Paper      | Code        | 
|------------|-------------|
| [ArXiv](https://arxiv.org/abs/2205.15023)      | [GitHub](https://github.com/jbr-ai-labs/mamba)      |

This *model-based* MARL architecture is built on top [Dreamerv2](https://arxiv.org/abs/2010.02193). Its core is based on a recurrent state-space model (RSSM), that has two components --- stochastic and deterministic --- and each one is represented with own latent variable $$z_t$$ and $$h_t$$, consequentially. The latter is done with recurrent neural network and the former --- with a variational auto-encoder. The model is parameterised with a set of parameters $$\phi$$.
$$
\begin{align*}
    \text{RSSM} \begin{cases}
    \text{Recurrent model:} & h^i_t = f_\phi (h^i_{t-1}, \mathbf{z}_{t-1}, \mathbf{a}_{t-1}), \\
    \text{Representation model:} & z^i_t \sim q_\phi(z^i_t\vert h^i_t, o^i_t) , \\
    \text{Transition predictor:} & \hat{z}^i_t  \sim p_\phi(\hat{z}^i_t\vert h^i_t), \\
    \end{cases}
\end{align*}
$$
$$
\begin{align*}
    \text{Auxiliary components}\begin{cases}        
    \text{Observation predictor:} & \hat{o}^i_t  \sim p_\phi(\hat{o}^i_t\vert h^i_t, z^i_t), \\
    \text{Action predictor:} & \hat{a}^i_{t-1}  \sim p_\phi(\hat{a}^i_{t-1}\vert h^i_t, z^i_t), \\ 
    \text{Reward predictor:} & \hat{r}^i_t  \sim p_\phi(\hat{r}^i_t\vert h^i_t, z^i_t), \\
    \text{Discount predictor:} & \hat{\gamma}^i_t  \sim p_\phi(\hat{\gamma}^i_t\vert h^i_t, z^i_t)
    \end{cases}
\end{align*}
$$
The model accepts observations of the learning agent $$o^i_t$$ as input.The transition predictor tries to build the next model state only from the current model state about any information about transition, so the dynamics are learned implicitly, making it unnecessary to make and watch in the future for planning. The discount predictor estimates how likely an episode will end when learning behaviours from model predictions, and the reward predictor lets the model figure out the reward function.   

Loss for every agent $$i$$ function to achieve the objective is defined in the following way:
$$
\begin{align*}
    \begin{split}
    \max_\phi\mathbb{E}_{q_\phi(z_{1:T}\vert o_{1:T}, a_{0:T})} -\log p_\phi (o_t\vert h_t, z_t) - \log p_\phi (r_t\vert h_t, z_t)  \\ -\log p_\phi (\gamma_t\vert h_t, z_t) -\overbrace{\log p_\phi (a_{t-1}\vert h_t, z_t)}^{\text{information loss}} + \beta \cdot KL\left[ q_\phi(z_t\vert h_t, x_t) \vert \vert   p_\phi(z_t\vert h_t) \right] 
    \end{split}
\end{align*}
$$

and optimised with KL-balancing, which encourages learning an accurate prior
over increasing posterior entropy, so that the prior better approximates the aggregate posterior. 

The model then used for planning to learn a policy from trajectories imagined in the compact latent space. These trajectories start from posterior states computed during model training --- the authors call the approach “imagination MDP”. The actor and critic neural networks are supposed to learn behavioural policy from “imagined” trajectories of representations generated by the world model:
$$
\begin{align}
    \text{Actor: } a_t \sim \pi_\theta(a_t\vert [z_t, h_t]), && \text{Critic: }  v_\psi(R_t\vert [z_t, h_t]).
\end{align}
$$

For actor learning, the authors utilise REINFORCE with entropy exploration, whereas, for critic learning, $$TD(\lambda)$$ is used. 

Additionally, the method extends RSSM mentioned earlier with a communication block using stacked attention layers to obtain feature vectors 
$$
\begin{align*}
    e^i_t = Attn([z_{t-1}^1, a_{t-1}^1, \ldots, z_{t-1}^n, a_{t-1}^n])[i], \ i=1, \ldots, n,
\end{align*}
$$
also, there was added additional component to the loss function to maximise mutual information between the latent state and the previous action of the agent (action prediction):
$$
\begin{align*}
    \mathcal{L}_{\text{info}} = - I((h_t^i, z_t^i), a_t^i) \approx - \log p_\phi (a^i_{t-1} \vert  h_t^i, z_t^i)
\end{align*}
$$


### IQL

| Paper      | Code        | 
|------------|-------------|
| [ArXiv](https://arxiv.org/abs/1511.08779)      | [GitHub](https://github.com/oxwhirl/pymarl)      |


Denoting $$Q$$-function of the POSG as $$Q^\pi(s, \mathbf{a}) = \mathbb{E}_{\pi, P} \left[ \sum_{t=0}^{T_{\text{max}}} \gamma^t r_{t+1} \vert  s_0 = s, \mathbf{a}_t \sim \pi(\cdot\vert s)\right]$$, the goal is to compute the optimal $$Q^*$$-value: $$Q^*(s, \mathbb{a}) = \max_\pi Q^\pi(s, \mathbf{a})$$. This can we recursively rewritten, using a Bellman equation: $$Q^*(s, \mathbf{a}) = \mathbb{E}_{\pi, P} \left[ r + \gamma \max_{\mathbf{a'}} Q^\pi (s, \mathbf{a'})\right]$$. Deep $$Q$$-value network approach this problem of *value iteration* as regression of parameters $$\theta$$:

$$
\begin{align*}
\begin{split}
    \min_\theta \mathbb{E}_{\pi, P} \left[ (y - Q^\theta (s, \mathbf{a}))^2 \right], \\
    y = \mathbb{E}_{\pi, P} \left[ r + \gamma \max_{\mathbf{a'}} Q^{\theta^-} (s, \mathbf{a'})\right]
\end{split}
\end{align*}
$$

where $$\theta^-$$ are parameters of separate target networks that are periodically copied from $$\theta$$ and kept constant for a number of iterations . This is model-free, off-policy algorithm that has quite a lot of [modifications](https://arxiv.org/abs/1710.02298). To ensure the exploration, actions are picked with decaying $$\varepsilon$$-greedy strategy.

IQL is the simplest extention of a method that consists of using the autonomous [$$Q$$-learning algorithm](https://arxiv.org/abs/1312.5602) for each agent in the environment ($$\forall i \in \mathcal{I} \rightarrow Q^{\theta^i}(s, a^i)$$), thereby using the environment as the sole source of interaction between agents.

<!-- 
<details> 
  <summary>Deep Q-learning Network</summary>
</details> -->

### VDN

| Paper      | Code        | 
|------------|-------------|
| [ArXiv](https://arxiv.org/abs/1706.05296)      | [GitHub](https://github.com/Louiii/ValueDecomposition)      |

The authors introduce a novel learned additive value-decomposition approach over individual agents. Implicitly, the value decomposition network aims to learn an optimal linear value decomposition from the team reward signal, by back-propagating the total $$Q$$-gradient through deep neural networks representing the individual component value functions.

$$
\begin{align*}
    Q^\pi((h^1, \ldots, h^n), (a^1, \ldots, a^n)) \approx \sum_{i=1}^n \bar{Q}^{\pi^i}(h^i, a^i)
\end{align*}
$$

This additive value decomposition is specifically motivated by avoiding the spurious reward signals that emerge in purely independent learners.The implicit value function learned by each agent depends only on local observations, and so is more easily learned. Although learning requires a degree of centralisation, the learned agents can be deployed independently, since each agent acting greedily with respect to its local value $$\bar{Q}^{\pi^i}$$ is equivalent to a central arbiter choosing joint actions by, maximising the overall sum.

### QMIX

| Paper      | Code        | 
|------------|-------------|
| [ArXiv](https://arxiv.org/abs/1803.11485)      | [GitHub](https://github.com/oxwhirl/pymarl/)      |

This method extends [VDN](#vdn), introducing richer class of value function decompositions. The observation for the method is the insight that the full factorisation of VDN is not necessary in order to be able to extract decentralised policies that are fully consistent with their centralised counterpart. Thus, it could be only enough to ensure that a global argmax performed on $$\arg\max_{\mathbf{a}} Q^\pi$$ yields
the same result as a set of individual argmax operations performed on invidiual (per agent) $$Q$$-value (a.k.a. [Individual-Global-Max](https://arxiv.org/pdf/1905.05408)(IGM))): 
$$\arg\max_{\mathbf{a}} Q^\pi = \left( \arg\max_{a^1} \bar{Q}^{\pi^1}, \ldots,  \arg\max_{a^n} \bar{Q}^{\pi^n}\right)  $$ 

This allows each agent a to participate in a decentralised execution solely by choosing greedy actions with respect to its $$Q$$-value. To ensure monotonicity of the decompostion, with help of hypernetworks, the following constraint is applied to the weights but not biases of $$Q$$-functions: $$\frac{Q^\pi}{\bar{Q}^{\pi^i}} \geq 0$$. 


### QPLEX

| Paper      | Code        | 
|------------|-------------|
| [ArXiv](https://arxiv.org/abs/2008.01062)      | [GitHub](https://github.com/wjh720/QPLEX)      |

The method's goal is achieving the complete expressiveness of the IGM function class with effective scalability for cooperative MARL. The paper presents an aprroach, that takes a duplex dueling network architecture to factorise the joint action-value function into individual action-value functions. 

Dueling procedure:
$$
\begin{align*}
\begin{split}
\begin{cases}
    Q^\pi(h, \mathbf{a}) = V^\pi(h) + A^\pi(h, \mathbf{a}), \  V^\pi(h)=\max_{\mathbf{a'}}Q^\pi(h, \mathbf{a'}), & \text{Joint Duelling}, \\
    \bar{Q}^{\pi^i}(h^i, a^i) = \bar{V}^{\pi^i}(h^i) + \bar{A}^{\pi^i}(h^i, a), \  \bar{V}^{\pi^i}(h^i)=\max_{a'^{i}}\bar{Q}^{\pi^i}(h^i, a'^i), & \text{Individual Duelling},
\end{cases} \\
\text{s.t. } \arg\max_{\mathbf{a}} A^\pi = \left( \arg\max_{a^1} \bar{A}^{\pi^1}, \ldots,  \arg\max_{a^n} \bar{A}^{\pi^n}\right)
\end{split}
\end{align*}
$$
Compared with perious methods, losing the guarantee of exact IGM consistency due to approximation, QPLEX takes advantage of a duplex dueling architecture to encode it into the neural network structure and provide a guaranteed IGM consistency.

The overall architecture of QPLEX consists of two main components: 
1. an Individual Action-Value Function for each agent
2. a Duplex Dueling component that composes individual action-value functions into a joint action-value function under the advantage-based IGM constraint


### Learnable MAPF methods

Generally, MAPF approaches can be distinguished by access level of a method:
1. centralised approaches: each learning agent has the access to the others' locations and goals
2. decentralised approaches: each learning agent has the access to only own goal locations but can only observe other agents' locations if only they present in his field of view usually described as a neighbourhood of the agent's location.

There have been done a lot of preliminary work.
Notably, most of them employ complex training procedures, including multiple stage of imitation learning as well as reward shaping. The most common paradigm for these methods is *centralised fully-observable* MAPF, although, some ease the centralisation, assuming selective communication between the agents.  


### PICO

| Paper      | Code        | 
|------------|-------------|
| [ArXiv](https://arxiv.org/pdf/2202.03634)      | [GitHub](https://github.com/mail-ecnu/PICO)      |

The method incorporates the implicit planning priorities into the communication topology within the decentralised multi-agent reinforcement learning framework. Assembling with the classic coupled planners, the implicit priority learning module can be utilised to form the dynamic communication topology, which also builds an effective collision-avoiding mechanism. PICO performs significantly better in large-scale MAPF tasks in success rates and collision rates than state-of-the-art Elearning-based planners.

### Follower

| Paper      | Code        | 
|------------|-------------|
| [ArXiv](https://arxiv.org/abs/2310.01207)      | [GitHub](https://github.com/Cognitive-AI-Systems/learn-to-follow)      |

Follower is a *decentralised* learnable MAPF method, targeting lifelong MAPF, i.e. constantly re-assigning goals after the agents complete them. It is comprised of the two complementary modules combined into a coherent pipeline.
1. Heuristic Path Decider is used to construct an individual path to the goal, using breadth-first search (BFS) to get not the shortest but most dispersed paths.
2. Learnable Follower is invoked to discover colision avoidance policies for the planned paths.


### MATS-LP

| Paper      | Code        | 
|------------|-------------|
| [ArXiv](https://arxiv.org/abs/2312.15908)      | [GitHub](https://github.com/Cognitive-AI-Systems/mats-lp)      |

The method operates on the lifelong variant of MAPF (LMAPF), whereafter an agent reaches its goal, one is immediately assigned to another one (via an external assignment procedure) and has to continue moving to a new goal. Thus, LMAPF generally asks to find a set of $$K$$ initial plans and update each agent’s plan when it reaches the current goal and receives a new one.

The method combines two principal ingredients. 

1. [Monte-Carlo Tree Search (MCTS)](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) for an agent to reason about the possible future states of the environment and to choose the most promising action to be performed at the current time step, i.e., such action that, on the one hand, maximizes the chance of reaching the goal (eventually) and, on the other hand, decrease the chances of collisions and deadlocks with the other agents. 

2. A learnable policy, approximated by a neural network inside the MCTS simulation step and modelled by [PPO](https://arxiv.org/abs/1707.06347), tailored to accomplish MAPF tasks from the perspective of the single agent. Importantly, as this policy is extensively used in MCTS to simulate the future states of the environment, it should be computationally efficient (fast).

### Switcher

| Paper      | Code        | 
|------------|-------------|
| [ArXiv](https://grafft.github.io/assets/pdf/switcher2023.pdf)      | [GitHub](https://github.com/Cognitive-AI-Systems/when-to-switch)      |


The method introduces two novel and conceptually different policies for PO-MAPF. 

1. based on the search-based re-planning (REPLAN). At each timestep, an agent builds the shortest path to its goal using a history of the egocentric observations by a heuristic search algorithm. Other agents are considered as obstacles that need to be avoided. To mitigate the possible deadlocks and oscillating behaviour of the agents, it's augmented re-planning with additional decision-making procedures that pick a greedy or wait action under certain conditions.
2. based on a specifically designed reinforcement learning algorithm: Evolving Policy Optimization with Memory (EPOM). EPOM uses an actor-critic architecture with a recurrent neural network as a state approximator. One of the novel features of EPOM setting it apart from similar approaches, is the mechanism of augmenting the current observation with a patch of the previously observed and memorised map. Not only does this help stabilise learning, but it also contributes to higher performance of the policy. To determine the hyperparameters of the model during the learning process, a [population-based training approach](https://arxiv.org/abs/1711.09846) is utilised.

### SCRIMP

| Paper      | Code        | 
|------------|-------------|
| [ArXiv](https://arxiv.org/abs/2303.006053)      | [GitHub](https://github.com/marmotlab/SCRIMP)      |

This is a method where agents learn individual policies from even very small (down to 3x3) FOVs, by relying on a highly-scalable global communication mechanism based on a modified transformer. Futhther equipping the agents with a state-value-based tie-breaking strategy, improves performance in symmetric situations, and introducing intrinsic rewards to encourage exploration while mitigating the long-term credit assignment problem.

### DHC

| Paper      | Code        | 
|------------|-------------|
| [ArXiv](https://arxiv.org/abs/2106.11365)      | [GitHub](https://github.com/ZiyuanMa/DHC)      |


This paper combines communication with deep $$Q$$-learning to provide a novel learning based method for MAPF, where agents achieve cooperation via graph convolution. To guide RL algorithm on long-horizon goal-oriented tasks, the authors embed the potential choices of shortest paths from single source as heuristic guidance instead of using a specific path as in most existing works. The method treats each agent independently and trains the model from a single agent’s perspective. The final trained policy is applied to each agent for decentralized execution. The whole system is distributed during training and is trained under a curriculum learning strategy.

### DCC

| Paper      | Code        | 
|------------|-------------|
| [ArXiv](https://arxiv.org/abs/2109.05413)      | [GitHub](https://github.com/ZiyuanMa/DCC)      |

DCC is a simple yet efficient mechanism to reduce communication overhead in multi-agent systems. DCC can be instantiated by independent Q-learning [IQL](#iql), or any framework of centralized training and decentralized execution, such as [QMIX](#qmix) and [VDN](#vdn). The authors consider a request-reply scenario different from the traditional setup where each agent only sends out an almost information-less indicator, e.g. a scalar value, as the request signal. Instead, in the setup, each agent sends out a request signal with rich information including its own messages along with the relative positions of neighbors. In this way, after receiving the request, the agents being requested can immediately benefit from this query by collecting some information from the query agent.
<!-- 
### LaCAM

| Paper      | Code        | 
|------------|-------------|
| [ArXiv](https://arxiv.org/abs/2211.13432)      | [GitHub](https://github.com/Kei18/lacam)      |

This paper proposes a novel algorithm called lazy constraints addition search for MAPF (LaCAM). From the theoretical side, LaCAM is complete. It returns a solution for solvable instances, otherwise reports the non-existence. LaCAM has an easily extensible structure, which comprises a two-level search. At the high-level, it searches a sequence of configurations, where a configuration is a tuple of locations for all agents. At the low-level, it searches constraints that specify which agents go where in the next configuration. Successors at the high-level (i.e., configurations) are generated in a lazy manner while following constraints from the low-level, leading to a dramatic reduction of the search effort. -->

### RHCR

| Paper      | Code        | 
|------------|-------------|
| [ArXiv](https://arxiv.org/pdf/2005.07371)      | [GitHub](https://github.com/Jiaoyang-Li/RHCR)      |

This method approaches lifelong MAPF (LMAPF) problem. The framework is called Rolling- Horizon Collision Resolution (RHCR), and it decomposes lifelong MAPF into a sequence of Windowed MAPF instances and replan paths once every $$t$$ timestep(replanning period $$t$$ is user-specified) for interleaving planning and execution. A Windowed MAPF instance is different from a regular MAPF instance in the following ways:

1. it allows an agent to be assigned a sequence of goal locations within the same Windowed MAPF episode, and 
2. collisions need to be resolved only for the first $$w$$ timesteps (time horizon $$w \geq t$$ is user-specified)

The benefit of this decomposition is two-fold. First, it keeps the agents continually engaged, avoiding idle time, and thus increasing throughput. Second, it generates pliable plans that adapt to continually arriving new goal locations. In fact, resolving collisions in the entire time horizon (i.e., $$w=\infty$$) is often unnecessary since the paths of the agents can change as new goal locations arrive.
