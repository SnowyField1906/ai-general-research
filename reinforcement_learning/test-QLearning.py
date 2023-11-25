from World import World
from QLearning import QLearning
from Visualizer import Visualizer

problem = World('world.txt')

solver = QLearning(problem.n_states, problem.blackbox_move)
solver.train()

visualizer = Visualizer(problem)
visualizer.plot_policy(solver.policy)
visualizer.random_start_policy(solver.policy)
