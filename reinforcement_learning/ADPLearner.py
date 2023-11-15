import numpy as np
from PolicyIteration import PolicyIteration
from Constants import ACTION as A, TRAIN as T

class ADPLearner:
    def __init__(self, n_states, gamma=T.GAMMA, epsilon=T.EPSILON, xi=T.XI):
        self.n_states = n_states
        self.gamma = gamma
        self.epsilon = epsilon
        self.xi = xi

        self.u_table = np.zeros(n_states)
        self.r_table = np.zeros(n_states)
        self.p_table = np.zeros((n_states, A.LEN, n_states))
        self.cur_policy = np.random.randint(A.LEN, size=n_states)
        self.visited_state = np.zeros(n_states)
        self.count_action = np.zeros((n_states, A.LEN))
        self.count_outcome = np.zeros((n_states, A.LEN, n_states))

    def percept(self, s, a, s_prime, r):
        if self.visited_state[s_prime] == 0:
            self.u_table[s_prime] = r
            self.r_table[s_prime] = r
            self.visited_state[s_prime] = 1

        self.count_action[s, a] += 1
        self.count_outcome[s, a, s_prime] += 1
        self.p_table[s, a] = self.count_outcome[s, a] / self.count_action[s, a]

    def actuate(self, s_prime):
        if np.random.uniform() <= self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return self.cur_policy[s_prime]

    def policy_update(self):
        solver = PolicyIteration(
            self.r_table,
            self.p_table,
            self.gamma,
            init_policy=self.cur_policy,
            init_value=self.u_table
        )
        solver.train()
        self.cur_policy = solver.policy
        self.epsilon *= self.xi