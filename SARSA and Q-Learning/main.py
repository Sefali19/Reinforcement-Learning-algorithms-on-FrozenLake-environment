import gym
import numpy as np
from utils.environment import create_environment
from utils.policy import print_policy, epsilon_greedy
from utils.sarsa import sarsa
from utils.qlearning import qlearning
from utils.plotting import plot_V, plot_Q, plot_average_episode_length, plot_average_reward

def main():
    env = create_environment()
    print("current environment: ")
    env.reset()
    env.render()
    print()

    print("Running SARSA...")
    Q, reward_array, episode_lengths = sarsa(env)
    plot_V(Q, env)
    plot_Q(Q, env)
    print_policy(Q, env)
    plot_average_episode_length(episode_lengths) # Plot the average episode length
    plot_average_reward(reward_array)

    print("\nRunning Q-learning")
    Q, reward_array, episode_lengths = qlearning(env)
    plot_V(Q, env)
    plot_Q(Q, env)
    print_policy(Q, env)
    plot_average_episode_length(episode_lengths) # Plot the average episode length
    plot_average_reward(reward_array)

if __name__ == "__main__":
    main()
