from World import World
from ValueIteration import ValueIteration
from Visualizer import Visualizer

problem = World('data/world00.csv')

solver = ValueIteration(problem.reward_function, problem.transition_model, gamma=0.9)
solver.train()

visualizer = Visualizer(problem)
visualizer.visualize_value_policy(policy=solver.policy, utilities=solver.utilities)
visualizer.plot_policy(policy=solver.policy)

