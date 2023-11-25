import numpy as np
import matplotlib.pyplot as plt
from time import time
from Constants import ACTION as A, TRAIN as T, VISUALIZATION as V

class QLearning:
    def __init__(self, n_states, blackbox_move):
        self.n_states = n_states
        self.transition_threshold = T.TRANSITION_THRESHOLD

        self.q_table = np.zeros((n_states, A.LEN))
        self.policy = np.random.randint(A.LEN, size=n_states)

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
    
    def percept(self, state, action, next_state, reward):
        """
        Update the learned reward and transition model after each step moved in MDP from the given (s, a, s', r) associated with that step.

        ### Parameters
            - state: The current state.
            - action: The current action.
            - next_state: The next state.
            - reward: The reward of the current state.

        ### Algorithm
        Q-Table will be update on every step with given learning rate. Policy will be updated as well.
        """
        td_error = reward + T.DISCOUNT_FACTOR * np.max(self.q_table[next_state]) - self.q_table[state, action]
        self.q_table[state, action] += T.LEARNING_RATE * td_error
        self.policy[state] = np.argmax(self.q_table[state])
    
    def one_evaluation(self, state):
        win = 0
        reward_game = 0

        while True:
            action = self.actuate(state)
            next_state, reward = self.blackbox_move(state, action)
            self.percept(state, action, next_state, reward)
            reward_game += reward

            if reward in T.TERMINAL:
                win = reward == T.WIN
                break
            else:
                state = next_state

        self.transition_threshold *= T.DECAY_FACTOR        
        
        return reward_game, win

    def train(self, plot=True):
        reward_history = np.zeros(T.IMPROVEMENT_LIMIT)
        total_reward_history = np.zeros(T.IMPROVEMENT_LIMIT)
        total_reward = 0
        game_win = np.zeros(T.IMPROVEMENT_LIMIT)

        time_start = int(round(time() * 1000))

        for i in range(T.IMPROVEMENT_LIMIT):
            print(f'Training epoch {i + 1}')

            reward_episode, win_episode = self.one_evaluation(0)

            total_reward += reward_episode
            game_win[i] = win_episode
            reward_history[i] = reward_episode
            total_reward_history[i] = total_reward

        time_end = int(round(time() * 1000))

        print(f'time used = {time_end - time_start}')
        print(f'final reward = {total_reward}')

        segment = 10
        game_win = game_win.reshape((segment, T.IMPROVEMENT_LIMIT // segment))
        game_win = np.sum(game_win, axis=1)

        print(f'winning percentage = {game_win / (T.IMPROVEMENT_LIMIT // segment)}')

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