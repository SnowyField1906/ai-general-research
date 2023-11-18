
from World import World
from ADPLearning import ADPLearner
from Visualizer import Visualizer

problem = World('data/world00.txt')

solver = ADPLearner(problem.n_states, problem.blackbox_move)
solver.train()

visualizer = Visualizer(problem)
visualizer.plot_policy(solver.policy)
visualizer.visualize_value_policy(solver.policy, solver.values)
visualizer.random_start_policy(solver.policy)
