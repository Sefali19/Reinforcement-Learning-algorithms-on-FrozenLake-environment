import gym
import numpy as np
from utils.environment import create_environment
from utils.policy import print_policy
from utils.value_iteration import value_iteration


def main():
    env = create_environment()
    
    # print the environment
    print("current environment: ")
    env.reset()
    env.render()
    dims = env.desc.shape
    print()

    # run the value iteration
    policy = value_iteration(env)
    print("Computed policy: ")
    print(policy.reshape(dims))
    print_policy(policy, env)


if __name__ == "__main__":
    main()
