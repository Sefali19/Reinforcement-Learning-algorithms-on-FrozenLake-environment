import gym

def create_environment():
    custom_map3x3 = [
        'SFF',
        'FFF',
        'FHG',
    ]
    return gym.make("FrozenLake-v1", desc=custom_map3x3, is_slippery=False)
