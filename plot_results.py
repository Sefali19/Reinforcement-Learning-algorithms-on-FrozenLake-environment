import matplotlib.pyplot as plt
import numpy as np

def plot_results(rewards, alphas, n_steps):
    plt.figure(figsize=(12, 8))
    for n in n_steps:
        avg_rewards = [np.mean(rewards[(alpha, n)][-100:]) for alpha in alphas]
        plt.plot(alphas, avg_rewards, marker='o', label=f'n={n}')
    plt.xlabel('Alpha (Learning Rate)')
    plt.ylabel('Average Reward (last 100 episodes)')
    plt.title('n-step SARSA Performance on FrozenLake-v0')
    plt.legend()
    plt.show()
