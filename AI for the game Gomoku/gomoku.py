# ********
# This file is individualized for NetID dli106.
# ********
# No other imports are allowed
import sys
import math # in case you need it
import itertools as it
import numpy as np
from scipy.signal import correlate

# enum for grid cell contents
EMPTY = 0
MAX = +1
MIN = -1

class GomokuDomain:

    def __init__(self, board_size, win_size):
        self.board_size = board_size
        self.win_size = win_size

    def initial_state(self):
        return np.full((self.board_size, self.board_size), EMPTY)

    def string_of(self, state):
        symbols = np.array(tuple("o.x"))
        return "\n".join("".join(row) for row in symbols[state+1])

    def is_max_turn_in(self, state):
        return (state == MAX).sum() == (state == MIN).sum()

    def current_player_in(self, state):
        return MAX if self.is_max_turn_in(state) else MIN

    def valid_actions_in(self, state):
        return list(zip(*np.nonzero(state == EMPTY)))

    def perform(self, action, state):
        new_state = state.copy()
        new_state[action] = MAX if self.is_max_turn_in(state) else MIN
        return new_state

    def score_in(self, state):

        win_patterns = [
            np.rot90(np.ones((1, self.win_size))), # vertical
            np.ones((1,self.win_size)),  # horizontal
            np.eye(self.win_size),  # diagonal
            np.rot90(np.eye(self.win_size)) #anti_diagonal
            # TODO: add three more win patterns for horizontal, diagonal, and anti-diagonal
            # the functions np.rot90, np.ones, and np.eye may be helpful
        ]

        for pattern in win_patterns:
            matches = correlate(state, pattern, mode='valid', method='direct')
            if (matches == +self.win_size).any(): return +1
            if (matches == -self.win_size).any(): return -1

        return 0

    def is_over_in(self, state):
        draw = (state != EMPTY).all()
        return draw or self.score_in(state) != 0

def human_turn(game, state):
    # helper to run a human-controlled turn

    # Show human valid actions
    valid_actions = game.valid_actions_in(state)
    print('actions', valid_actions)

    # Ask human for move (repeat if their input is invalid)
    while True:
        try:
            action = tuple(map(int, input("Enter action in format '<row>,<col>': ").split(",")))
            if action not in valid_actions: raise ValueError
            break
        except KeyboardInterrupt:
            sys.exit(0)
        except:
            print("Invalid action, try again.")

    # return new state after turn
    state = game.perform(action, state)
    return state

if __name__ == "__main__":

    game = GomokuDomain(5,5)
    state = game.initial_state()

    while True:

        # show current state
        print(game.string_of(state))
        print('score', game.score_in(state))

        # end if game over
        if game.is_over_in(state):
            print("Game over.")
            break

        # take human-controlled move
        state = human_turn(game, state)

