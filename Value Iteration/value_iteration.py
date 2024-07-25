import numpy as np

def value_iteration(env):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V_states = np.zeros(n_states)  # init values as zero
    theta = 1e-8
    gamma = 0.8
    num_iterations = 1000

    for i in range(num_iterations):
        updated_v_state = np.copy(V_states)

        for state in range(n_states):
            Q_values = [sum([prob * (r + gamma * updated_v_state[s_])
                             for prob, s_, r, _ in env.P[state][action]])
                        for action in range(n_actions)]

            V_states[state] = max(Q_values)

        if np.sum(np.fabs(updated_v_state - V_states)) <= theta:
            print('Value-iteration converged at iteration# %d.' % (i + 1))
            print("Optimal value function: %s" % V_states)
            break

    policy = np.zeros(n_states, dtype=int)

    for s in range(n_states):
        Q_values = [sum([prob * (r + gamma * V_states[s_])
                         for prob, s_, r, _ in env.P[s][a]])
                    for a in range(n_actions)]

        policy[s] = np.argmax(np.array(Q_values))

    return policy
