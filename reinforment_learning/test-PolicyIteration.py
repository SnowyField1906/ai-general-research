from Visualizer import Visualizer
from World import World
from PolicyIteration import PolicyIteration

problem = World('data/world00.csv')

solver = PolicyIteration(problem.reward_function, problem.transition_model)
solver.train()

visualizer = Visualizer(problem)
visualizer.visualize_value_policy(solver.policy, solver.utilities)
visualizer.random_start_policy(solver.policy)
