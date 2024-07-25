import gym

def create_environment():
    return gym.make('FrozenLake-v1', is_slippery=False)
