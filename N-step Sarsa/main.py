import gym
from sarsa.nstep_sarsa import nstep_sarsa
from sarsa.plot_results import plot_results

def main():
    env = gym.make('FrozenLake-v0', map_name="8x8")
    alphas = [0.2, 0.5, 0.7]
    n_steps = [2, 4, 8]
    gamma = 0.9  # Discount factor
    epsilon = 0.1  # Exploration rate
    num_episodes = 5000  # Number of episodes for training

    all_rewards = {}

    for alpha in alphas:
        for n in n_steps:
            print(f'Training with α={alpha}, n={n}')
            Q, rewards = nstep_sarsa(env, n, alpha, gamma, epsilon, num_episodes)
            all_rewards[(alpha, n)] = rewards

    plot_results(all_rewards, alphas, n_steps)

if __name__ == "__main__":
    main()
