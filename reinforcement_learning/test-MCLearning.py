
from World import World
from MCLearning import MCLearner
from Visualizer import Visualizer

problem = World('data/world00.txt')

solver = MCLearner(problem.n_states, problem.blackbox_move)
solver.train()

visualizer = Visualizer(problem)
visualizer.plot_policy(solver.policy)
visualizer.random_start_policy(solver.policy)
