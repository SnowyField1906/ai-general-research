import numpy as np
import matplotlib.pyplot as plt
from Constants import ACTION as A, TRAIN as T, VISUALIZATION as V

class ValueIteration:
    def __init__(self, reward_function, transition_model, gamma=T.GAMMA, init_value=None):
        self.n_states = transition_model.shape[0]
        self.reward_function = np.nan_to_num(reward_function)
        self.transition_model = transition_model
        self.gamma = gamma
    
        self.policy = (np.ones(self.n_states) * -1).astype(int)

        if init_value is None:
            self.values = np.zeros(self.n_states)
        else:
            self.values = init_value

    def one_value_evaluation(self):
        """
        Perform one iteration of value evaluation.

        ### Algorithm
        For each State, its new Value is calculated from:
            - The current Value.
            - The expectation following its random_rate distribution to neighbor States based on the current Policy.
            - Reward of the current State.
            - Discount factor gamma.

        v(s) = r(s) + gamma * max(sum(p(s, a, s') * v(s')))

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
                values[action] = reward + self.gamma * np.inner(probability, self.values)

            new[state] = max(values)

        self.values = new
        delta = np.max(np.abs(old - new))

        return delta

    def run_policy_improvement(self):
        """
        Perform one Policy improvement.

        ### Algorithm
        For each State, its new Policy is calculated from:
            - The highest Value between all of its neighbor States.

        π(s) = argmax_a(v(s))
        """
        for state in range(self.n_states):
            neighbor_values = np.zeros(A.LEN)

            for action in A.ACTIONS:
                probability = self.transition_model[state, action]
                neighbor_values[action] = self.reward_function[state] + self.gamma * np.inner(probability, self.values)

            self.policy[state] = np.argmax(neighbor_values)

    def train(self, tol=T.TOL, epoch_limit=T.EVALUATION_LIMIT, plot=True):
        """
        Perform sweeps of Value evaluation iteratively with a stop criterion of the given tol or epoch_limit.

        ### Algorithm
        For each sweep, update the Values, until:
            - The highest change between updates is less than tol.
            - The number of sweeps exceeds epoch_limit.

        ### Parameters
            - tol         -- The stop criterion.
            - epoch_limit -- The maximum number of sweeps.
            - plot        --  Whether to plot learning curves showing number of evaluation sweeps.
        """
        delta = float('inf')
        delta_history = []

        while delta > tol and len(delta_history) < epoch_limit:
            delta = self.one_value_evaluation()
            delta_history.append(delta)

        self.run_policy_improvement()

        if plot is True:
            _, axe = plt.subplots(1, 1, figsize=V.FIG_SIZE)
            axe.plot(
                np.arange(len(delta_history)) + 1,
                delta_history,
                marker='o',
                label=f'Sweeps in Value evaluation with $\gamma= $' + f'{self.gamma}'
            )
            axe.set_xticks(np.arange(len(delta_history)))
            axe.set_xlabel('Sweeps')
            axe.set_ylabel('Delta')
            axe.legend()

            plt.tight_layout()
            plt.show()