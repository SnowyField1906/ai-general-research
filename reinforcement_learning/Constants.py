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
    REWARD = {
        OBJECT.EMPTY: -0.1,
        OBJECT.GOAL: 10.0,
        OBJECT.PIT: -10.0,
        OBJECT.WALL: np.NaN
    }
    WIN = REWARD[OBJECT.GOAL]
    LOSE = REWARD[OBJECT.PIT]
    TERMINAL = [WIN, LOSE]

    RANDOM_RATE = 0.2
    DISCOUNT_FACTOR = 0.9
    DECAY_FACTOR = 0.99
    TRANSITION_THRESHOLD = 0.95
    LEARNING_RATE = 0.2

    STOP_CRITERION = 1e-3
    EVALUATION_LIMIT = 100
    IMPROVEMENT_LIMIT = 1000
    EXECUTION_TIME_LIMIT = 200

class VISUALIZATION:
    FIG_SIZE = (6, 6)
    MARKER_SIZE = 20
    FONT_SIZE = 10

    EXECUTION_LIMIT = 100
    START_POS = (0, 0)