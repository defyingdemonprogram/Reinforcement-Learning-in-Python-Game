## Reinforcement Learning (RL)

Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment. The goal is to learn a policy—a strategy for choosing actions—that maximizes a cumulative reward over time. Unlike supervised learning, where the model learns from labeled data, in RL the agent is not provided with explicit examples of correct actions but rather learns through trial and error, receiving feedback in the form of rewards or penalties.

The fundamental components of reinforcement learning are:
- **Agent**: The learner or decision maker.
- **Environment**: The external system the agent interacts with.
- **State**: A representation of the environment at a given time.
- **Action**: The choice made by the agent that affects the environment.
- **Reward**: A scalar feedback signal that tells the agent how good or bad an action was.
- **Policy**: A strategy that the agent follows to decide which action to take based on the current state.
- **Value function**: A function that estimates the expected future rewards the agent can obtain from a given state, helping the agent to decide which states are more valuable.

### Q-Learning

Q-Learning is one of the most popular model-free reinforcement learning algorithms. It is a type of **temporal-difference learning**, where an agent learns the value of each action in each state. The "Q" in Q-learning stands for the quality of an action taken in a given state.

In Q-learning, the agent updates a table called the **Q-table**, where each entry $Q(s, a)$ represents the value of taking action $a$ in state $s$. The Q-values are updated based on the observed rewards and the estimated future rewards that can be obtained by following the best policy. Over time, as the agent explores the environment, the Q-values converge to the optimal action values.

The Q-learning algorithm updates the Q-value using the following **Bellman equation**:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_a Q(s', a') - Q(s, a) \right]
$$

Where:
- $Q(s, a)$ is the current Q-value for taking action $a$ in state $s$,
- $\alpha$ is the learning rate (a value between 0 and 1 that determines how much new information overrides old information),
- $r$ is the reward received after taking action $a$ in state $s$,
- $\gamma$ is the discount factor (a value between 0 and 1 that models the agent's consideration of future rewards),
- $\max_a Q(s', a')$ is the maximum Q-value over all possible actions $a'$ in the next state $s'$.

The Q-learning algorithm converges to the optimal policy in the long run as the agent explores the environment and updates its Q-values.

### Bellman Equation

The Bellman equation is a fundamental recursive relationship used in dynamic programming and reinforcement learning. It expresses the value of a state in terms of the immediate reward and the value of the future states that the agent can transition to. It is used to update the value functions (such as the Q-values in Q-learning) and provides a way to break down decision-making into smaller sub-problems.

The **Bellman equation** for state values in the context of RL is:

$$
V(s) = \max_a \left[ r(s, a) + \gamma \sum_{s'} P(s' | s, a) V(s') \right]
$$

Where:
- $V(s)$ is the value of state $s$,
- $r(s, a)$ is the immediate reward received after taking action $a$ from state $s$,
- $P(s' | s, a)$ is the probability of transitioning to state $s'$ from state $s$ by taking action $a$,
- $\gamma$ is the discount factor,
- $\sum_{s'} P(s' | s, a) V(s')$ is the expected value of the future states weighted by their transition probabilities.

In Q-learning, the Bellman equation is adapted to update the Q-values:

$$
Q(s, a) = r(s, a) + \gamma \max_a Q(s', a')
$$

Here, the Bellman equation helps in computing the expected cumulative rewards by considering both immediate rewards and the future rewards from possible next states. The Bellman equation is critical to dynamic programming and is at the heart of many reinforcement learning algorithms like Q-learning and value iteration.

#### Relationship between Q-learning and the Bellman Equation

- The **Bellman equation** is a theoretical formulation that helps in defining the optimal value function.
- **Q-learning** is an algorithm that uses the Bellman equation to iteratively update its Q-values as it interacts with the environment.

### Snake Game Logic

In the Snake game, the output of the reinforcement learning is a one-hot encoded vector:

- **[1, 0, 0]**: Snake continues straight (no change in direction).
- **[0, 1, 0]**: Snake turns right (clockwise).
- **[0, 0, 1]**: Snake turns left (counterclockwise).

The input features for reinforcement learning consist of 11 values:
1. Danger straight ahead
2. Danger to the right
3. Danger to the left
4. Move left
5. Move right
6. Move up
7. Food to the left
8. Food to the right
9. Food above
10. Food below

These features help the model assess the current state of the game and decide the optimal action.

**References**
* [Q-learning - Wikipedia](https://en.wikipedia.org/wiki/Q-learning)
* [Introduction to Q-learning - DataCamp](https://www.datacamp.com/tutorial/introduction-q-learning-beginner-tutorial)
