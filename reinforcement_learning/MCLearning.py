import numpy as np
import matplotlib.pyplot as plt
from time import time
from Constants import ACTION as A, TRAIN as T, VISUALIZATION as V

class MCLearner:
    def __init__(self, n_states, blackbox_move):
        self.n_states = n_states
        self.transition_threshold = T.TRANSITION_THRESHOLD

        self.policy = np.random.randint(A.LEN, size=n_states)
        self.q_table = np.ones((n_states, A.LEN))
        self.c_table = np.zeros((n_states, A.LEN))

        self.visited = np.zeros((n_states, A.LEN))
        self.count_table = np.zeros((n_states, A.LEN))

        self.blackbox_move = blackbox_move

    def actuate(self, next_state):
        """
        Return the next action for the agent based on currently learned policy for state s'.
        
        ### Parameters
            - next_state: The next state.

        ### Algorithm
        With probability ξ, the agent will choose a random action (exploration).
        Otherwise, the agent will choose the action based on the learned policy (exploitation).

        ### Return
            - The action to take.
        """
        if np.random.uniform() <= self.transition_threshold:
            return np.random.randint(A.LEN)
        else:
            return self.policy[next_state]

    def percept(self, state, action, reward):
        """
        Update the learned reward and transition model after each step moved in MDP from the given (s, a, s', r) associated with that step.

        ### Parameters
            - state: The current state.
            - action: The current action.
            - reward: The reward of the current state.

        ### Algorithm
        Every state-action pair that has been visited will be added with the current reward.
        """
        if self.visited[state, action] == 0:
            self.visited[state, action] = 1
        
        visited = self.visited == 1
        self.c_table[visited] += reward

    def one_evaluation(self, state):
        """
        Perform one episode of evaluation.

        ### Parameters
            - state: The starting state.

        ### Algorithm
        For each step, the agent will:
            - Actuate the next action based on the current state.
            - Percept the next state and reward based on the current state and action.
            - Update the learned reward and transition model.
            - Update the learned policy.
        Until:
            - The agent reaches the goal.
            - The agent falls into the pit.

        ### Return
            - The total reward of the episode.
            - Whether the agent wins the game.
        """
        win = 0
        reward_game = 0

        while True:
            action = self.actuate(state)
            next_state, reward = self.blackbox_move(state, action)
            self.percept(state, action, reward)
            reward_game += reward

            if reward in T.TERMINAL:
                win = reward == T.WIN
                break
            else:
                state = next_state

        return reward_game, win

    def policy_improvement(self):
        """
        Update the learned policy after each episode.

        ### Algorithm
        Use the learned C-Table to update the Q-Table by Monte Carlo method.
        """

        visited = self.visited == 1
        self.count_table[visited] += 1

        self.q_table[visited] += (self.c_table[visited] - self.q_table[visited]) / self.count_table[visited]

        for state in range(self.n_states):
            self.policy[state] = np.argmax(self.q_table[state])

        self.c_table = np.zeros((self.n_states, A.LEN))
        self.visited = np.zeros((self.n_states, A.LEN))
        self.transition_threshold *= T.DECAY_FACTOR

    def train(self, plot=True):
        reward_history = np.zeros(T.IMPROVEMENT_LIMIT)
        total_reward_history = np.zeros(T.IMPROVEMENT_LIMIT)
        total_reward = 0
        game_win = np.zeros(T.IMPROVEMENT_LIMIT)

        time_start = int(round(time() * 1000))

        for i in range(T.IMPROVEMENT_LIMIT):
            print(f'Training epoch {i + 1}')

            reward_episode, win_episode = self.one_evaluation(0)
            self.policy_improvement()

            total_reward += reward_episode
            game_win[i] = win_episode
            reward_history[i] = reward_episode
            total_reward_history[i] = total_reward

        time_end = int(round(time() * 1000))

        print(f'Time used = {time_end - time_start}')
        print(f'Final reward = {total_reward}')

        segment = 10
        game_win = game_win.reshape((segment, T.IMPROVEMENT_LIMIT // segment))
        game_win = np.sum(game_win, axis=1)

        print(f'Winning percentage = {game_win / (T.IMPROVEMENT_LIMIT // segment)}')

        if plot:
            _, axes = plt.subplots(2, 1, figsize=V.FIG_SIZE, sharex='all')

            axes[0].plot(
                np.arange(len(total_reward_history)),
                total_reward_history,
                label=r'Trained with $\xi$' + f' = {T.DECAY_FACTOR}'
            )
            axes[0].set_ylabel('Total rewards')
            axes[0].legend()

            axes[1].plot(
                np.arange(len(reward_history)),
                reward_history,
                marker='o',
                linestyle='none'
            )
            axes[1].set_xlabel('Episode')
            axes[1].set_ylabel('Reward from\na single game')
            axes[0].grid(axis='x')
            axes[1].grid(axis='x')

            plt.tight_layout()
            plt.show()

            _, ax = plt.subplots(1, 1, figsize=V.FIG_SIZE)
            ax.plot(
                np.arange(1, segment + 1) * (T.IMPROVEMENT_LIMIT // segment),
                game_win / (T.IMPROVEMENT_LIMIT // segment),
                marker='o',
                label=r'$\xi$ = ' + f'{T.DECAY_FACTOR}'
            )
            ax.set_ylabel('Winning percentage')
            ax.set_xlabel('Episode')
            ax.grid(axis='x')
            ax.grid(axis='x')

            plt.tight_layout()
            plt.show()

        return total_reward