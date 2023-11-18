import numpy as np
from Constants import ACTION as A
from World import World
from Visualizer import Visualizer

problem = World(
    'data/world00.txt'
)

policy=[2, 2, 3, 3, 3, 0, 3, 3, 2, 1, 2, 3, 1, 0, 2, 0]
values=[2.68545065, 3.43905213, 4.23985797, 4.91403074, 2.64765564
, 0., 5.30410547, 6.21248421, 3.25924498, 4.17825261
, 6.36382476, 7.80098376, 2.67025469 , -10., 7.57579258
, 10.        ]

def policy_improvement():
    """
    Perform one Policy improvement.

    ### Algorithm
    For each State, its new Policy is calculated from:
        - The highest Value between all of its next States.

    π(s) = argmax(T(s' | s, a) * V(s'))

    ### Return
        - The number of States whose Policy has been changed.
    """
    for state in range(problem.n_states):
        next_values = np.zeros(A.LEN)
        
        for action in A.ACTIONS:
            probability = problem.transition_model[state, action]
            next_values[action] = np.inner(probability, values)

        policy[state] = np.argmax(next_values)

policy_improvement()
visualizer = Visualizer(problem)
visualizer.plot_map()
visualizer.plot_policy(policy)
visualizer.visualize_value_policy(policy, values)