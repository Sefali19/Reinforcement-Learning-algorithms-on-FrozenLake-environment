import math
import numpy as np
from sarsa.choose_action import choose_action

def nstep_sarsa(env, n=8, alpha=0.2, gamma=0.9, epsilon=0.1, num_ep=int(1e4)):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards_per_ep = []
    for episodes in range(num_ep):
        T = math.inf
        t = 0
        states = []
        actions = []
        rewards = []
        ep_rewards = 0
        state = env.reset()
        action = choose_action(state, Q, epsilon, env)
        states.append(state)
        actions.append(action)
        rewards.append(0)
        while True:
            if t < T:
                next_state, reward, done, _ = env.step(action)
                states.append(next_state)
                rewards.append(reward)
                ep_rewards += reward
                if done:
                    T = t + 1
                else:
                    action = choose_action(next_state, Q, epsilon, env)
                    actions.append(action)
            tau = t - n + 1
            if tau >= 0:
                G = sum([gamma**(i - tau - 1) * rewards[i] for i in range(tau + 1, min(tau + n, T))])
                if tau + n < T:
                    G += gamma**n * Q[states[tau + n], actions[tau + n]]
                Q[states[tau], actions[tau]] += alpha * (G - Q[states[tau], actions[tau]])
            if tau == T - 1:
                break
            t += 1
            if t < T:
                state = next_state
                action = actions[t]
        rewards_per_ep.append(ep_rewards)
    return Q, rewards_per_ep
