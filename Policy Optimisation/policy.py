import numpy as np
import itertools
from utils.value_function import value_policy

def bruteforce_policies(env):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    terms = terminals(env)
    optimal_policies = []
    optimal_value = np.zeros(n_states)

    for policy in itertools.product(range(n_actions), repeat=n_states):
        env.render()
        v = value_policy(env, policy)
        optimal = True
        for s in range(n_states):
            if s in terms:
                continue
            if optimal_value[s] <= v[s]:
                if optimal_value[s] < v[s]:
                    optimal_value[s] = v[s]
                    optimal_policies = []
            else:
                optimal = False
        if optimal:
            optimal_policies.append(policy)

    print("Optimal value function:")
    print(optimal_value)
    print("Number of optimal policies:")
    print(len(optimal_policies))
    print("Optimal policies:")
    print(np.array(optimal_policies))
    return optimal_policies

def terminals(env):
    n_states = env.observation_space.n
    terms = []
    for s in range(n_states):
        if env.P[s][0][0][0] == 1.0 and env.P[s][0][0][3] is True:
            terms.append(s)
    return terms

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
