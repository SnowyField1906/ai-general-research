import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from World import World
from Constants import ACTION as A, VISUALIZATION as V

class Visualizer:
    def __init__(self, world: World):
        self.world = world

    def plot_map(self):
        """
        Visualize the map of the Grid World.
        """
        w = self.world

        unit = min(V.FIG_SIZE[1] // w.n_rows, V.FIG_SIZE[0] // w.n_cols)
        unit = max(1, unit)
        _, ax = plt.subplots(1, 1, figsize=V.FIG_SIZE)
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
                ax.text(x + 0.5 * unit, y + 0.5 * unit, f's = {s}\nr = {w.reward_function[s]}', horizontalalignment='center', verticalalignment='center', fontsize=V.FONT_SIZE)

        plt.tight_layout()
        plt.show()

    def plot_policy(self, policy):
        """
        Visualize the given policy.
        """
        w = self.world

        unit = min(V.FIG_SIZE[1] // w.n_rows, V.FIG_SIZE[0] // w.n_cols)
        unit = max(1, unit)
        _, ax = plt.subplots(1, 1, figsize=V.FIG_SIZE)
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

                if w.map[i, j] == 0:
                    s = w.get_state_from_pos((i, j))
                    a = policy[s]
                    ax.plot([x + 0.5 * unit], [y + 0.5 * unit], marker=A.SYMBOLS[a], markersize=V.MARKER_SIZE, linestyle='none', color='#1f77b4')


        plt.tight_layout()
        plt.show()

    def visualize_value_policy(self, policy, values):
        """
        Visualize the given policy and value values.
        """
        w = self.world

        unit = min(V.FIG_SIZE[1] // w.n_rows, V.FIG_SIZE[0] // w.n_cols)
        unit = max(1, unit)
        _, ax = plt.subplots(1, 1, figsize=V.FIG_SIZE)
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
                    ax.text(x + 0.5 * unit, y + 0.5 * unit, f'{values[s]:.4f}', horizontalalignment='center', verticalalignment='center', fontsize=V.FONT_SIZE)
                if policy is not None:
                    if w.map[curr_pos] == 0:
                        a = policy[s]
                        ax.plot([x + 0.5 * unit], [y + 0.5 * unit], marker=A.SYMBOLS[a], alpha=0.4, linestyle='none', markersize=V.MARKER_SIZE, color='#1f77b4')

        plt.tight_layout()
        plt.show()

    def random_start_policy(self, policy):
        """
        Repeatedly execute the given policy for n times.
        """
        w = self.world

        scores = np.zeros(V.EXECUTION_LIMIT)
        i = 0
        while i < V.EXECUTION_LIMIT:
            temp = w.execute_policy(policy, V.START_POS)
            print(f'i = {i} Random start result: {temp}')
            if temp > float('-inf'):
                scores[i] = temp
                i += 1

        print(f'max = {np.max(scores)}')
        print(f'min = {np.min(scores)}')
        print(f'mean = {np.mean(scores)}')
        print(f'std = {np.std(scores)}')

        _, ax = plt.subplots(1, 1, figsize=V.FIG_SIZE)
        ax.set_xlabel('Total rewards in a single game')
        ax.set_ylabel('Frequency')
        ax.hist(scores, bins=100, color='#1f77b4', edgecolor='black')
        plt.show()

        return np.max(scores), np.min(scores), np.mean(scores)
