import numpy as np
from time import time
from Constants import ACTION as A, TRAIN as T

class World:
    def __init__(self, filename):
        file = open(filename)
        self.map = np.array(
            [list(map(float, s.strip().split(","))) for s in file.readlines()]
        )
        file.close()

        self.n_rows = self.map.shape[0]
        self.n_cols = self.map.shape[1]
        self.n_states = self.n_rows * self.n_cols
        self.reward_function = self.get_reward_function()
        self.transition_model = self.get_transition_model()

    def get_state_from_pos(self, pos):
        """
        Transfer a position in the World into an integer representing state.

        ### Parameters
            - pos: A tuple of two integers representing the position in the World.
        
        ### Return
            - An integer representing the state.
        """
        return pos[0] * self.n_cols + pos[1]

    def get_pos_from_state(self, state):
        """
        Transfer an integer representing state back into a position in the World.

        ### Parameters
            - state: An integer representing the state.
        ### Return
            - A tuple of two integers representing the position in the World.
        """
        return state // self.n_cols, state % self.n_cols
    
    def get_next_pos(self, pos, action):
        """
        Get the next position given the current position and the action.

        ### Parameters
            - pos: A tuple of two integers representing the current position.
            - action: An integer representing the action.

        ### Return
            - A tuple of two integers representing the next position.
        """
        if action == A.LEFT:
            return pos[0], max(pos[1] - 1, 0)
        elif action == A.RIGHT:
            return pos[0], min(pos[1] + 1, self.n_cols - 1)
        elif action == A.UP:
            return max(pos[0] - 1, 0), pos[1]
        elif action == A.DOWN:
            return min(pos[0] + 1, self.n_rows - 1), pos[1]
        
    def get_likely_action(self, action):
        """
        Get the main action and its two likely actions.

        ### Parameters
            - action: An integer representing the main action.

        ### Return
            - A list of three integers representing the main action and its two likely actions.
        """
        return [action, (action + 1) % A.LEN, (action - 1) % A.LEN]

    def get_reward_function(self):
        """
        Calculate the reward function R(s) of MDP.
        """
        reward_table = np.zeros(self.n_states)

        for r in range(self.n_rows):
            for c in range(self.n_cols):
                pos = (r, c)
                state = self.get_state_from_pos(pos)
                reward_table[state] = T.REWARD[self.map[pos]]

        return reward_table

    def get_transition_model(self):
        """
        Calculate the transitional model T(s'|s, a) of MDP.

        ### Return
            - A 3-dimensional numpy array with shape (n_states, A.LEN, n_states).
        """
        transition_model = np.zeros((self.n_states, A.LEN, self.n_states))

        for r in range(self.n_rows):
            for c in range(self.n_cols):
                pos = (r, c)
                state = self.get_state_from_pos(pos)
                next_state = np.zeros(A.LEN)

                if self.map[pos] == 0:
                    for action in A.ACTIONS:
                        next_pos = self.get_next_pos(pos, action)
                        next_state[action] = self.get_state_from_pos(next_pos)
                else:
                    next_state = np.ones(A.LEN) * state

                for action in A.ACTIONS:
                    main, likely1, likely2 = self.get_likely_action(action)
                    transition_model[state, action, int(next_state[main])] += 1 - T.RANDOM_RATE
                    transition_model[state, action, int(next_state[likely1])] += T.RANDOM_RATE / 2.0
                    transition_model[state, action, int(next_state[likely2])] += T.RANDOM_RATE / 2.0

        return transition_model

    def generate_random_policy(self):
        """
        Initialize a policy of random actions.

        ### Return
            - A numpy array with shape (n_states, ) representing the policy.
        """
        return np.random.randint(A.LEN, size=self.n_states)

    def execute_policy(self, policy, start_pos, time_limit=T.EXECUTION_TIME_LIMIT):
        """
        Get the total reward starting from the start_pos following the given policy.

        ### Parameters
            - policy    : A numpy array with shape (n_states, ) representing the policy.
            - start_pos : A tuple of two integers representing the starting position.
            - time_limit: An integer representing the time limit of each execution.

        ### Return
            - The total reward.
        """
        state = self.get_state_from_pos(start_pos)
        reward = self.reward_function[state]
        total_reward = reward

        start_time = int(round(time() * 1000))
        overtime = False

        while reward not in T.TERMINAL:
            state = np.random.choice(self.n_states, p=self.transition_model[state, policy[state]])
            reward = self.reward_function[state]
            total_reward += reward
            cur_time = int(round(time() * 1000)) - start_time

            if cur_time > time_limit:
                overtime = True
                break

        if overtime is True:
            return float('-inf')
        else:
            return total_reward

    def blackbox_move(self, state, action):
        """
        Simulate an environment where the agent can not access the reward function and transition model. The agent provides the current state s and an action a, this function returns the next state s' of the agent and the reward assigned to the agent through this move.

        ### Parameters
            - state: The current state.
            - action: The action.

        ### Return
            - The next state of the agent.
            - The reward assigned to the agent through this move.
        """
        temp = self.transition_model[state, action]

        while True:
            next_state = np.random.choice(self.n_states, p=temp)
            reward = self.reward_function[next_state]

            if np.isfinite(reward):
                break

        return next_state, reward