import numpy as np

class ACTION:
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

    ACTIONS = [LEFT, UP, RIGHT, DOWN]
    LEN = 4

    SYMBOLS = ['<', '^', '>', 'v']
    NAMES = ['LEFT', 'UP', 'RIGHT', 'DOWN']

class OBJECT:
    EMPTY = 0
    GOAL = 1
    PIT = 2
    WALL = 3

class TRAIN:
    TOL = 1e-3
    GAMMA = 0.85
    EPSILON = 0.9
    XI = 0.99
    RANDOM_RATE = 0.2
    EVALUATION_LIMIT = 100
    IMPROVEMENT_LIMIT = 100
    EXECUTION_TIME_LIMIT = 200
    REWARD = {
        OBJECT.EMPTY: -0.1,
        OBJECT.GOAL: 10.0,
        OBJECT.PIT: -10.0,
        OBJECT.WALL: np.NaN
    }

class VISUALIZATION:
    FIG_SIZE = (6, 6)
    MARKER_SIZE = 20
    FONT_SIZE = 10

    EXECUTION_LIMIT = 100
    START_POS = (0, 0)