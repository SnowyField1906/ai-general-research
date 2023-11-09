class ACTION:
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

    ACTIONS = [LEFT, UP, RIGHT, DOWN]
    LEN = 4

    SYMBOLS = ['<', '^', '>', 'v']
    NAMES = ['LEFT', 'UP', 'RIGHT', 'DOWN']

class WORLD:
    EMPTY = 0
    GOAL = 1
    PIT = 2
    WALL = 3