# ********
# This file is individualized for NetID dli106.
# ********
# No other imports are allowed
import itertools as it # in case you need it
import math # in case you need it
import numpy as np
import gomoku as gm
import scipy as sp
from minimax import minimax

def simple_evaluator(game, state):
    # always estimates 0 utility for non-game-over states at the depth limit
    return 0

def better_evaluator(game, state):
    score = 0

    for row in range(game.board_size):
        for col in range(game.board_size):
            if state[row, col] != gm.EMPTY:
                player = state[row, col]
                
                # Check horizontal lines (within bounds)
                if col + game.win_size <= game.board_size:
                    score += evaluate_line(state[row, col:col + game.win_size])
                
                # Check vertical lines (within bounds)
                if row + game.win_size <= game.board_size:
                    score += evaluate_line(state[row:row + game.win_size, col])
                
                # Check diagonal (top-left to bottom-right)
                if row + game.win_size <= game.board_size and col + game.win_size <= game.board_size:
                    score += evaluate_line([state[row + i, col + i] for i in range(game.win_size)])
                
                # Check diagonal (bottom-left to top-right)
                if row - game.win_size >= -1 and col + game.win_size <= game.board_size:
                    score += evaluate_line([state[row - i, col + i] for i in range(game.win_size)])
                
    return score

def evaluate_line(line):
    score = 0
    for i in range(len(line)):
        if line[i] == gm.MAX:
            score += 1
        elif line[i] == gm.MIN:
            score -= 1
    return score


if __name__ == "__main__":

    board_size = 6
    win_size = 5
    max_depth = 3

    game = gm.GomokuDomain(board_size, win_size)
    state = game.initial_state()

    choice = input("Enter S to play against simple_evaluator, B to play against better_evaluator: ")
    evaluation_fn = better_evaluator if choice == "B" else simple_evaluator

    # loop until game over
    print("Starting game, AI's first move may take several seconds...")
    while True:

        # print current game state
        print(game.string_of(state))
        print('score', game.score_in(state))

        # stop if current game is over
        if game.is_over_in(state):
            print("Game over.")
            break

        # AI controls max player
        if game.is_max_turn_in(state):

            # select next state with minimax search
            state, _, node_count = minimax(game, state, max_depth, evaluation_fn)
            print("Total nodes processed:", node_count)

        # Human input for min player
        else:

            state = gm.human_turn(game, state)


