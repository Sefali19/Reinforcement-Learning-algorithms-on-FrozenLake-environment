import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from matplotlib import colors, patches

def plot_V(Q, env):
    fig = plt.figure()
    if not hasattr(env, 'desc'):
        env = env.env
    dims = env.desc.shape
    V = np.zeros(dims)
    for s in range(len(Q)):
        idx = np.unravel_index(s, dims)
        V[idx] = np.max(Q[s])
        if env.desc[idx] in ['H', 'G']:
            V[idx] = 0.
    plt.imshow(V, origin='upper', extent=[0, dims[0], 0, dims[1]], vmin=.0, vmax=.6, cmap=plt.cm.RdYlGn, interpolation='none')
    for x, y in product(range(dims[0]), range(dims[1])):
        plt.text(y + 0.5, dims[0] - x - 0.5, '{:.3f}'.format(V[x, y]),
                 horizontalalignment='center', verticalalignment='center')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def plot_Q(Q, env):
    fig = plt.figure()
    ax = fig.gca()
    if not hasattr(env, 'desc'):
        env = env.env
    dims = env.desc.shape

    up = np.array([[0, 1], [0.5, 0.5], [1, 1]])
    down = np.array([[0, 0], [0.5, 0.5], [1, 0]])
    left = np.array([[0, 0], [0.5, 0.5], [0, 1]])
    right = np.array([[1, 0], [0.5, 0.5], [1, 1]])
    tri = [left, down, right, up]
    pos = [[0.2, 0.5], [0.5, 0.2], [0.8, 0.5], [0.5, 0.8]]

    cmap = plt.cm.RdYlGn
    norm = colors.Normalize(vmin=.0, vmax=.6)

    ax.imshow(np.zeros(dims), origin='upper', extent=[0, dims[0], 0, dims[1]], vmin=.0, vmax=.6, cmap=cmap)
    ax.grid(which='major', color='black', linestyle='-', linewidth=2)

    for s in range(len(Q)):
        idx = np.unravel_index(s, dims)
        x, y = idx
        if env.desc[idx] in ['H', 'G']:
            ax.add_patch(patches.Rectangle((y, 3 - x), 1, 1, color=cmap(.0)))
            plt.text(y + 0.5, dims[0] - x - 0.5, '{:.2f}'.format(.0),
                     horizontalalignment='center',
                     verticalalignment='center')
            continue
        for a in range(len(tri)):
            ax.add_patch(patches.Polygon(tri[a] + np.array([y, 3 - x]), color=cmap(Q[s][a])))
            plt.text(y + pos[a][0], dims[0] - 1 - x + pos[a][1], '{:.2f}'.format(Q[s][a]),
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=9, fontweight=('bold' if Q[s][a] == np.max(Q[s]) else 'normal'))

    plt.xticks([])
    plt.yticks([])
    plt.show()

def plot_average_episode_length(episode_lengths, window=100):
    avg_lengths = np.convolve(episode_lengths, np.ones(window) / window, mode='valid')
    plt.plot(avg_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Average Episode Length')
    plt.title('Average Episode Length Over Time')
    plt.show()

def plot_average_reward(reward_array, window=100):
    avg_rewards = np.convolve(reward_array, np.ones(window) / window, mode='valid')
    plt.plot(avg_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Average Reward Over Time')
    plt.show()
