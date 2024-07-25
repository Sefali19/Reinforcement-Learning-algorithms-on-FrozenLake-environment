import numpy as np

def value_policy(env, policy):
    n_states = env.observation_space.n
    r = np.zeros(n_states)  # Reward vector is zero everywhere except for the goal state (last state)
    r[-1] = 1.
    gamma = 0.8
    P = trans_matrix_for_policy(env, policy)
    v = np.linalg.inv(np.eye(n_states) - (gamma * P)) @ r
    return v

def trans_matrix_for_policy(env, policy):
    n_states = env.observation_space.n
    transitions = np.zeros((n_states, n_states))
    for s in range(n_states):
        probs = env.P[s][policy[s]]
        for el in probs:
            transitions[s, el[1]] += el[0]
    return transitions
