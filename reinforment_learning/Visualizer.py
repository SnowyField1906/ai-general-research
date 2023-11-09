import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from time import time
from World import World
from Helpers import ACTION as A

default_fig_size = (8, 6)
default_n = 100
default_mode = True
default_start_pos = (2, 0)

class Visualizer:
    def __init__(self, world: World):
        self.world = world

    def random_start_policy(self, policy, start_pos=default_start_pos, n=default_n, fig_size=default_fig_size, plot=default_mode):
        """
        Repeatedly execute the given policy for n times.
        """
        w = self.world

        start_time = int(round(time() * 1000))
        overtime = False
        scores = np.zeros(n)
        i = 0
        while i < n:
            temp = w.execute_policy(policy, start_pos)
            print(f'i = {i} Random start result: {temp}')
            if temp > float('-inf'):
                scores[i] = temp
                i += 1
            cur_time = int(round(time() * 1000)) - start_time
            if cur_time > n * w.time_limit:
                overtime = True
                break

        print(f'max = {np.max(scores)}')
        print(f'min = {np.min(scores)}')
        print(f'mean = {np.mean(scores)}')
        print(f'std = {np.std(scores)}')

        if overtime is False and plot is True:
            _, ax = plt.subplots(1, 1, figsize=fig_size)
            ax.set_xlabel('Total rewards in a single game')
            ax.set_ylabel('Frequency')
            ax.hist(scores, bins=100, color='#1f77b4', edgecolor='black')
            plt.show()

        if overtime is True:
            print('Overtime!')
            return None
        else:
            return np.max(scores), np.min(scores), np.mean(scores)

    def plot_map(self, fig_size=default_fig_size):
        """
        Visualize the map of the Grid World.
        """
        w = self.world

        unit = min(fig_size[1] // w.n_rows, fig_size[0] // w.n_cols)
        unit = max(1, unit)
        _, ax = plt.subplots(1, 1, figsize=fig_size)
        ax.axis('off')

        for i in range(w.n_cols + 1):
            if i == 0 or i == w.n_cols:
                ax.plot([i * unit, i * unit], [0, w.n_rows * unit], color='black')
            else:
                ax.plot([i * unit, i * unit], [0, w.n_rows * unit], alpha=0.7, color='grey', linestyle='dashed')
        for i in range(w.n_rows + 1):
            if i == 0 or i == w.n_rows:
                ax.plot([0, w.n_cols * unit], [i * unit, i * unit], color='black')
            else:
                ax.plot([0, w.n_cols * unit], [i * unit, i * unit], alpha=0.7, color='grey', linestyle='dashed')

        for i in range(w.n_rows):
            for j in range(w.n_cols):
                y = (w.n_rows - 1 - i) * unit
                x = j * unit
                if w.map[i, j] == 3:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='black', alpha=0.6)
                    ax.add_patch(rect)
                elif w.map[i, j] == 2:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='red', alpha=0.6)
                    ax.add_patch(rect)
                elif w.map[i, j] == 1:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='green', alpha=0.6)
                    ax.add_patch(rect)

        plt.tight_layout()
        plt.show()

    def plot_policy(self, policy, fig_size=default_fig_size):
        """
        Visualize the given policy.
        """
        w = self.world

        unit = min(fig_size[1] // w.n_rows, fig_size[0] // w.n_cols)
        unit = max(1, unit)
        _, ax = plt.subplots(1, 1, figsize=fig_size)
        ax.axis('off')
        for i in range(w.n_cols + 1):
            if i == 0 or i == w.n_cols:
                ax.plot([i * unit, i * unit], [0, w.n_rows * unit], color='black')
            else:
                ax.plot([i * unit, i * unit], [0, w.n_rows * unit], alpha=0.7, color='grey', linestyle='dashed')
        for i in range(w.n_rows + 1):
            if i == 0 or i == w.n_rows:
                ax.plot([0, w.n_cols * unit], [i * unit, i * unit], color='black')
            else:
                ax.plot([0, w.n_cols * unit], [i * unit, i * unit], alpha=0.7, color='grey', linestyle='dashed')

        for i in range(w.n_rows):
            for j in range(w.n_cols):
                y = (w.n_rows - 1 - i) * unit
                x = j * unit
                if w.map[i, j] == 3:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='black', alpha=0.6)
                    ax.add_patch(rect)
                elif w.map[i, j] == 2:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='red', alpha=0.6)
                    ax.add_patch(rect)
                elif w.map[i, j] == 1:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='green', alpha=0.6)
                    ax.add_patch(rect)
                s = w.get_state_from_pos((i, j))
                if w.map[i, j] == 0:
                    a = policy[s]
                    ax.plot([x + 0.5 * unit], [y + 0.5 * unit], marker=A.SYMBOLS[a], linestyle='none', markersize=max(fig_size)*unit, color='#1f77b4')

        plt.tight_layout()
        plt.show()

    def visualize_value_policy(self, policy, utilities, fig_size=default_fig_size):
        """
        Visualize the given policy and utility utilities.
        """
        w = self.world

        unit = min(fig_size[1] // w.n_rows, fig_size[0] // w.n_cols)
        unit = max(1, unit)
        _, ax = plt.subplots(1, 1, figsize=fig_size)
        ax.axis('off')

        for i in range(w.n_cols + 1):
            if i == 0 or i == w.n_cols:
                ax.plot([i * unit, i * unit], [0, w.n_rows * unit], color='black')
            else:
                ax.plot([i * unit, i * unit], [0, w.n_rows * unit], alpha=0.7, color='grey', linestyle='dashed')
        for i in range(w.n_rows + 1):
            if i == 0 or i == w.n_rows:
                ax.plot([0, w.n_cols * unit], [i * unit, i * unit], color='black')
            else:
                ax.plot([0, w.n_cols * unit], [i * unit, i * unit], alpha=0.7, color='grey', linestyle='dashed')

        for i in range(w.n_rows):
            for j in range(w.n_cols):
                curr_pos = (i, j)
                y = (w.n_rows - 1 - i) * unit
                x = j * unit
                s = w.get_state_from_pos(curr_pos)
                if w.map[curr_pos] == 3:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='black', alpha=0.6)
                    ax.add_patch(rect)
                elif w.map[curr_pos] == 2:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='red', alpha=0.6)
                    ax.add_patch(rect)
                elif w.map[curr_pos] == 1:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='green', alpha=0.6)
                    ax.add_patch(rect)
                if w.map[curr_pos] != 3:
                    ax.text(x + 0.5 * unit, y + 0.5 * unit, f'{utilities[s]:.4f}', horizontalalignment='center', verticalalignment='center', fontsize=max(fig_size)*unit*0.6)
                if policy is not None:
                    if w.map[curr_pos] == 0:
                        a = policy[s]
                        ax.plot([x + 0.5 * unit], [y + 0.5 * unit], marker=A.SYMBOLS[a], alpha=0.4, linestyle='none', markersize=max(fig_size)*unit, color='#1f77b4')

        plt.tight_layout()
        plt.show()