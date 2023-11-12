import numpy as np
import matplotlib.pyplot as plt
from Helpers import ACTION as A

default_tol = 1e-3
default_gamma = 0.9
default_evaluation_limit = 100
default_training_limit = 100

class PolicyIteration:
    def __init__(self, reward_function, transition_model, gamma=default_gamma, init_policy=None, init_value=None):
        self.num_states = transition_model.shape[0]
        self.reward_function = np.nan_to_num(reward_function)
        self.transition_model = transition_model
        self.gamma = gamma
    
        if init_policy is None:
            self.policy = np.random.randint(0, A.LEN, self.num_states)
        else:
            self.policy = init_policy
        if init_value is None:
            self.values = np.zeros(self.num_states)
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

        u(s) = r(s) + gamma * sum(p(s, a, s') * u(s'))

        ### Return
            - The maximum change in Value.
        """
        old = self.values
        new = np.zeros(self.num_states)

        # for state in range(self.num_states):
        #     action = self.policy[state]
        #     probability = self.transition_model[state, action]
        #     reward = self.reward_function[state]
        #     new[state] = reward + self.gamma * np.inner(probability, old)
        
        A = np.eye(self.num_states) - self.gamma * self.transition_model[range(self.num_states), self.policy]
        b = self.reward_function
        new = np.linalg.solve(A, b)

        self.values = new
        delta = np.max(np.abs(old - new))

        return delta

    def run_policy_evaluation(self, tol=default_tol, epoch_limit=default_evaluation_limit):
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

        print("Delta history:", delta_history)

        return len(delta_history)

    def run_policy_improvement(self):
        """
        Perform one Policy improvement.

        ### Algorithm
        For each State, its new Policy is calculated from:
            - The highest Value between all of its neighbor States.

        π(s) = argmax(sum(p(s, a, s') * u(s')))

        ### Return
            - The number of States whose Policy has been changed.
        """
        update_policy_count = 0

        for s in range(self.num_states):
            temp = self.policy[s]
            neighbor_values = np.zeros(A.LEN)
            
            for a in A.ACTIONS:
                p = self.transition_model[s, a]
                neighbor_values[a] = np.inner(p, self.values)

            self.policy[s] = np.argmax(neighbor_values)

            if temp != self.policy[s]:
                update_policy_count += 1

        return update_policy_count

    def train(self, epoch_limit=default_training_limit, plot=True):
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
            _, axes = plt.subplots(2, 1)

            axes[0].plot(
                np.arange(len(eval_sweeps_history)),
                eval_sweeps_history,
                label=f'Sweeps in Policy evaluation with $\gamma =$ {self.gamma}'
            )
            axes[0].set_xticks(np.arange(len(eval_sweeps_history)))
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Sweeps')
            axes[0].legend()

            axes[1].plot(
                np.arange(len(policy_changes_history)),
                policy_changes_history,
                label=f'Updates in Policy improvement with $\gamma =$ {self.gamma}'
            )
            axes[1].set_xticks(np.arange(len(policy_changes_history)))
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Updates')
            axes[1].legend()

            plt.tight_layout()
            plt.show()