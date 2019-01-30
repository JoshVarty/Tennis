# Report

## Learning Algorithm

My agent uses [Multi Agent Deep Deterministic Policy Gradients](https://arxiv.org/abs/1706.02275) (MADDPG). The bulk of the algorithm has been implemented in [Agent.step()](https://github.com/JoshVarty/Tennis/blob/e29c20069d085a5ab8fd03fadb5d1f948c7238d2/agent.py#L43-L51) which consists of two steps.

1. The agent stores `state`, `action`, `reward`, `next_state` and `done` information in a replay buffer
2. The agent samples experiences from this replay buffer in order to learn

## Sampling

Our agent starts off knowing nothing about its environment. It begins learning about its environment by taking primarily random actions. Randomness is introduced by both the random weights of our actor network and explict noise added to actions chosen by this network. After taking an action, the agent records the resulting tuple of `(state, action, reward, next_state, done)` in a replace buffer of size `1e6`.

As training progresses, the amount of noise added to the agent's chosen action is scaled downward. Initially the noise is scaled by a factor of `5`. This scaling factor is decayed at a rate of `1 / (300 * self.learning_rounds)` after each round of learning. This allows the agent to gradually make more deterministic decisions and controls the degree of exploration vs. exploitation.

## Learning

After there are enough samples to form a batch (`128`) the agent is ready to begin learning. Every `4` (`self.learn_every`) time steps, the agent randomly samples a batch of `128` experiences on which it can train and passes them to [`Agent.learn(self, experiences, gamma, agent_number)`](https://github.com/JoshVarty/Tennis/blob/e29c20069d085a5ab8fd03fadb5d1f948c7238d2/agent.py#L69). Randomly sampling experiences in this fashion allows our agent to avoid training on highly correllated experiences (which is a problem for neural networks).

When learning, our agent first computes the next set of actions using the `actor_target` network. The agent then uses the `critic_target` network to calculate a target value for the next set of states and actions. This value is discounted by `gamma` (`0.99`) and added to the current set of rewards giving us a target value for the current state. Next we use `critic_local` to calculate the expected value of the current state and actions. The mean-squared error between the target value of the current state and the expected value of the current state is our critic loss over which optimize.

After calculating critic loss, we also calculate actor loss using the `actor_local` network to generate actions and `critic_local` to place a value on the state and actions that were chosen.

After computing the losses and optimizing, we apply a "soft update" to the target networks. This soft update is governed by a `tau` parameter (`1e-3`).

## Hyperparameters

 - Buffer Size: `1e6`
  - The size of our replay buffer from which we will sample experiences

 - Batch Size: `128`
  - The size of batches of experience that we will sample from the replay buffer and train over

 - Learning Rate: `1e-4`
  - The parameter that governs the size of the updates we make to our networks during optimization.

 - Learn Every: `2`
  - Controls how often we perform a learning step while generating experiences

 - Learning Rounds: `4`
  - Controls how many learning steps we run inbetween generating experiences

 - Gamma: `0.99`
  - Controls the degree to which we discount future rewards. A high value allows us to prioritize long-term rewards

 - Tau: `1e-3`
  - Controls how quickly we update the target networks with weights from the local network in our "soft update" step

 - Epsilon: `5` down to `0`
  - Controls the degree to which we scale noise that is applied to actions chosen by the network. Scale down by a factor of `1 / (300 * self.learning_rounds)` at each learning step.

## Model Architecture

Each agent contains:

 - An `actor_local` network
 - An `actor_target` network
 - A `critical_local` network
 - A `critic_target` network

The actor networks have the following structure:

- A fully connected layer with `48` inputs and `256` outputs using ReLU activations
- A fully connected layer with `256` inputs and `256` outputs using ReLU activations
- A fully connected layer with `256` inputs and `2` outputs using ReLU activations

The critic networks have the following structure:

- A fully connected layer with `52` inputs and `256` outputs using ReLU activations
- A fully connected layer with `256` inputs and `256` outputs using ReLU activations
- A fully connected layer with `256` inputs and `1` output using ReLU activations

Each agent takes in both sets of state observations (`24` per agent). The `critic` network concatenates the chosen actions (`2` per agent) to its input.

## Results

After approximately two hours of training (3,900 episodes) on my local machine, the agent achieved a score greater than `0.5`. Its progress is shown below:

![https://i.imgur.com/Qyb4dtJ.png](https://i.imgur.com/Qyb4dtJ.png)





