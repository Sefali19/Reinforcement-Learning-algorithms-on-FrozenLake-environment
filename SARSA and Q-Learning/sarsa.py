import numpy as np
from utils.policy import epsilon_greedy

def sarsa(env, alpha=0.1, gamma=0.9, epsilon=0.5, num_ep=int(1e4)):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    episode_lengths = np.zeros(num_ep)
    reward_array = np.zeros(num_ep)

    for i in range(num_ep):
        s = env.reset()
        done = False
        total_reward = 0
        episode_length = 0
        a = epsilon_greedy(Q, epsilon, env.action_space.n, s)
        while not done:
            s_, r, done, _ = env.step(a)
            total_reward += r
            episode_length += 1
            a_ = epsilon_greedy(Q, epsilon, env.action_space.n, s_)
            Q[s, a] += alpha * (r + (gamma * Q[s_, a_]) - Q[s, a])
            s, a = s_, a_
        reward_array[i] = total_reward
        episode_lengths[i] = episode_length

    return Q, reward_array, episode_lengths
