import gym
import numpy as np
from utils.environment import setup_environment
from utils.policy import bruteforce_policies, value_policy


def main():
    env, n_states = setup_environment()

    print("Current environment: ")
    env.render()
    print("")

    # Here a policy is just an array with the action for a state as an element
    policy_left = np.zeros(n_states, dtype=np.intc)  # 0 for all states
    policy_right = np.ones(n_states, dtype=np.intc) * 2  # 2 for all states

    # Value functions:
    print("Value function for policy_left (always going left):")
    print(value_policy(env, policy_left))
    print("Value function for policy_right (always going right):")
    print(value_policy(env, policy_right))

    optimal_policies = bruteforce_policies(env)

    # This code can be used to "rollout" a policy in the environment:
    """
    print("Rollout policy:")
    max_iter = 100
    state = env.reset()
    for
