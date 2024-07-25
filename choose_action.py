import numpy as np

def choose_action(state, Q, epsilon, env):
    if np.random.rand() < epsilon:
        return np.random.choice(np.arange(env.action_space.n))
    else:
        return np.argmax(Q[state])
