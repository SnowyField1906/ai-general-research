import numpy as np

from World import World
from Visualizer import Visualizer
from Constants import ACTION as A

problem = World('data/world00.txt')
init_policy = problem.generate_random_policy()

visualizer = Visualizer(problem)
visualizer.plot_map()
visualizer.plot_policy(init_policy)
visualizer.visualize_value_policy(init_policy, np.zeros(problem.n_states))
visualizer.random_start_policy(init_policy)

reward_function = problem.reward_function
print(f'reward function =')
for s in range(len(reward_function)):
    print(f'State s = {s}, Reward R({s}) = {reward_function[s]}')

transition_model = problem.transition_model
print(f'transition model =')
for s in range(transition_model.shape[0]): # for each state
    print('======================================')
    for a in range(transition_model.shape[1]): # for each action
        print('--------------------------------------')
        for s_prime in range(transition_model.shape[2]): # for each next state
            print(f'{s}-{A.NAMES[a]} => {s_prime}-{100*transition_model[s, a, s_prime]}%')