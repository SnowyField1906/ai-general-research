import numpy as np
import matplotlib.pyplot as plt
from Constants import ACTION as A, TRAIN as T, VISUALIZATION as V

class ValueIteration:
    def __init__(self, reward_function, transition_model, init_value=None):
        self.n_states = transition_model.shape[0]
        self.reward_function = np.nan_to_num(reward_function)
        self.transition_model = transition_model
    
        self.policy = (np.ones(self.n_states) * -1).astype(int)

        if init_value is None:
            self.values = np.zeros(self.n_states)
        else:
            self.values = init_value

    def one_evaluation(self):
        """
        Perform one iteration of value evaluation.

        ### Algorithm
        For each State, its new Value is calculated from:
            - The current Value.
            - The expectation following its random_rate distribution to next States based on the current Policy.
            - Reward of the current State.
            - Discount factor gamma.

        v(s) = r(s) + gamma * max(p(s, a, s') * v(s'))

        ### Return
            - The maximum change in Value.
        """
        old = self.values
        new = np.zeros(self.n_states)

        for state in range(self.n_states):
            values = np.zeros(A.LEN)
            reward = self.reward_function[state]

            for action in A.ACTIONS:
                probability = self.transition_model[state, action]
                values[action] = reward + T.DISCOUNT_FACTOR * np.inner(probability, self.values)

            new[state] = max(values)

        self.values = new
        delta = np.max(np.abs(old - new))

        return delta

    def policy_improvement(self):
        """
        Perform one Policy improvement.

        ### Algorithm
        For each State, its new Policy is calculated from:
            - The highest Value between all of its next States.

        π(s) = argmax(p(s, a, s') * v(s'))
        """
        for state in range(self.n_states):
            next_values = np.zeros(A.LEN)

            for action in A.ACTIONS:
                probability = self.transition_model[state, action]
                next_values[action] = np.inner(probability, self.values)

            self.policy[state] = np.argmax(next_values)

    def train(self, plot=True):
        """
        Perform sweeps of Value evaluation iteratively with a stop criterion or epoch limit.

        ### Algorithm
        For each sweep, update the Values, until:
            - The highest change between updates is less than tol.
            - The number of sweeps exceeds epoch_limit.

        ### Parameters
            - plot        --  Whether to plot learning curves showing number of evaluation sweeps.
        """
        delta = float('inf')
        delta_history = []

        while delta > T.STOP_CRITERION and len(delta_history) < T.EVALUATION_LIMIT:
            delta = self.one_evaluation()
            delta_history.append(delta)

        self.policy_improvement()

        if plot is True:
            _, ax = plt.subplots(1, 1, figsize=V.FIG_SIZE)
            ax.plot(
                np.arange(len(delta_history)) + 1,
                delta_history,
                marker='o',
                label=f'Sweeps in Value evaluation with $\gamma= $' + f'{T.DISCOUNT_FACTOR}'
            )
            ax.set_xticks(np.arange(len(delta_history)))
            ax.set_xlabel('Sweeps')
            ax.set_ylabel('Delta')
            ax.legend()

            plt.tight_layout()
            plt.show()