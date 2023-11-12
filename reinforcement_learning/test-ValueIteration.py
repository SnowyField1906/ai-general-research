from World import World
from ValueIteration import ValueIteration
from Visualizer import Visualizer

problem = World('data/world00.txt')

solver = ValueIteration(problem.reward_function, problem.transition_model, gamma=0.9)
solver.train()

visualizer = Visualizer(problem)
visualizer.visualize_value_policy(policy=solver.policy, values=solver.values)
visualizer.plot_policy(policy=solver.policy)

