import numpy as np
import matplotlib.pyplot as plt
from Constants import ACTION as A, TRAIN as T, VISUALIZATION as V

class PolicyIteration:
    def __init__(self, reward_function, transition_model, gamma=T.GAMMA, init_policy=None, init_value=None):
        self.n_states = transition_model.shape[0]
        self.reward_function = np.nan_to_num(reward_function)
        self.transition_model = transition_model
        self.gamma = gamma
    
        if init_policy is None:
            self.policy = np.random.randint(0, A.LEN, self.n_states)
        else:
            self.policy = init_policy
        if init_value is None:
            self.values = np.zeros(self.n_states)
        else:
            self.values = init_value

    def one_policy_evaluation(self):
        """
        Perform one sweep of Policy evaluation.
        
        ### Algorithm
        For each State, its new Value is calculated from:
            - The current Value.
            - The expectation following its random_rate distribution to neighbor States based on the current Policy.
            - Reward of the current State.
            - Discount factor gamma.

        v(s) = r(s) + gamma * sum(p(s, a, s') * v(s'))

        ### Return
            - The maximum change in Value.
        """
        old = self.values
        new = np.zeros(self.n_states)

        # Approach 1
        for state in range(self.n_states):
            action = self.policy[state]
            probability = self.transition_model[state, action]
            reward = self.reward_function[state]
            new[state] = reward + self.gamma * np.inner(probability, old)
        
        # Approach 2
        # A = np.eye(self.n_states) - self.gamma * self.transition_model[range(self.n_states), self.policy]
        # b = self.reward_function
        # new = np.linalg.solve(A, b)

        self.values = new
        delta = np.max(np.abs(old - new))

        return delta

    def run_policy_evaluation(self, tol=T.TOL, epoch_limit=T.EVALUATION_LIMIT):
        """
        Perform sweeps of Policy evaluation iteratively with a stop criterion of the given tol or epoch_limit.

        ### Algorithm
        For each sweep, update the Values, until:
            - The highest change between updates is less than tol.
            - The number of sweeps exceeds epoch_limit.

        ### Parameters
            - tol         -- The stop criterion.
            - epoch_limit -- The maximum number of sweeps.

        ### Return
            - The number of sweeps.
        """
        delta = float('inf')
        delta_history = []

        while delta > tol and len(delta_history) < epoch_limit:
            delta = self.one_policy_evaluation()
            delta_history.append(delta)

        return len(delta_history)

    def run_policy_improvement(self):
        """
        Perform one Policy improvement.

        ### Algorithm
        For each State, its new Policy is calculated from:
            - The highest Value between all of its neighbor States.

        π(s) = argmax_a(v(s))

        ### Return
            - The number of States whose Policy has been changed.
        """
        update_policy_count = 0

        for state in range(self.n_states):
            temp = self.policy[state]
            neighbor_values = np.zeros(A.LEN)
            
            for action in A.ACTIONS:
                probability = self.transition_model[state, action]
                neighbor_values[action] = np.inner(probability, self.values)

            self.policy[state] = np.argmax(neighbor_values)

            if temp != self.policy[state]:
                update_policy_count += 1

        return update_policy_count

    def train(self, epoch_limit=T.IMPROVEMENT_LIMIT, plot=True):
        """
        Perform Policy iteration by iteratively alternates Policy evaluation and Policy improvement.

        ### Algorithm
        For each iteration, store into history:
            - The number of Evaluation sweeps.
            - The number of Policy updates.
        Until:
            - The Policy is unchanged.
            - The number of iterations exceeds epoch_limit.

        ### Parameters
            - epoch_limit -- The maximum number of iterations.
            - plot        -- Whether to plot learning curves showing number of evaluation sweeps and Policy updates in each iteration.
        """
        policy_changes = float('inf')
        eval_sweeps_history = []
        policy_changes_history = []

        while policy_changes != 0 and len(eval_sweeps_history) < epoch_limit:
            eval_sweeps = self.run_policy_evaluation()
            policy_changes = self.run_policy_improvement()
            eval_sweeps_history.append(eval_sweeps)
            policy_changes_history.append(policy_changes)

        if plot is True:
            _, axes = plt.subplots(2, 1, figsize=V.FIG_SIZE)

            axes[0].plot(
                np.arange(len(eval_sweeps_history)),
                eval_sweeps_history,
                marker='o',
                label=f'Sweeps in Policy evaluation with $\gamma =$ {self.gamma}'
            )
            axes[0].set_xticks(np.arange(len(eval_sweeps_history)))
            axes[0].set_xlabel('Sweeps')
            axes[0].set_ylabel('Delta')
            axes[0].legend()

            axes[1].plot(
                np.arange(len(policy_changes_history)),
                policy_changes_history,
                marker='o',
                label=f'Updates in Policy improvement with $\gamma =$ {self.gamma}'
            )
            axes[1].set_xticks(np.arange(len(policy_changes_history)))
            axes[1].set_xlabel('Updates')
            axes[1].set_ylabel('Delta')
            axes[1].legend()

            plt.tight_layout()
            plt.show()