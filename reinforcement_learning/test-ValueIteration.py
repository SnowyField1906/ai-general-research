from World import World
from ValueIteration import ValueIteration
from Visualizer import Visualizer

problem = World('data/world00.txt')

solver = ValueIteration(problem.reward_function, problem.transition_model)
solver.train()

visualizer = Visualizer(problem)
visualizer.plot_policy(solver.policy)
visualizer.visualize_value_policy(solver.policy, solver.values)
visualizer.random_start_policy(solver.policy)