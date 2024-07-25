import gym
import numpy as np

def setup_environment():
    custom_map3x3 = [
        'SFF',
        'FFF',
        'FHG',
    ]
    env = gym.make("FrozenLake-v0", desc=custom_map3x3)
    n_states = env.observation_space.n
    return env, n_states
