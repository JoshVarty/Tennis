from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch
from agent import Agent

env = UnityEnvironment(file_name="Tennis_Linux_NoVis/Tennis.x86_64", worker_id=13)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

def maddpg(n_episodes=4000, max_t=1000, train_mode=True):
    all_scores = []
    scores_window = deque(maxlen=100)

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=train_mode)[brain_name]         
        states = np.reshape(env_info.vector_observations, (1,48)) 
        agent_0.reset()
        agent_1.reset()
        scores = np.zeros(num_agents)

        while True:
            actions = get_actions(states, add_noise=True)
            env_info = env.step(actions)[brain_name]
            next_states = np.reshape(env_info.vector_observations, (1, 48))
            rewards = env_info.rewards
            done = env_info.local_done
            agent_0.step(states, actions, rewards[0], next_states, done, 0)
            agent_1.step(states, actions, rewards[1], next_states, done, 1)
            scores += np.max(rewards)                         
            states = next_states

            if np.any(done):
                # we're done when the ball hit the ground or goes out of bounds
                scores_window.append(np.mean(scores))
                all_scores.append(np.mean(scores))
                break

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

        if np.mean(scores_window) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)), flush=True)
            # Save only the actor because that's all we need to run at test (visualization) time
            torch.save(agent_0.actor_local.state_dict(), 'checkpoint_actor_0.pth')
            torch.save(agent_1.actor_local.state_dict(), 'checkpoint_actor_1.pth')
            break
            
    return all_scores

def get_actions(states, add_noise=False):
    action_0 = agent_0.act(states, add_noise)
    action_1 = agent_1.act(states, add_noise)

    return np.stack((action_0, action_1), axis=0).flatten()

agent_0 = Agent(state_size, action_size)
agent_1 = Agent(state_size, action_size)

scores = maddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
