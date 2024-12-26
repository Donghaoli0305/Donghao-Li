# ********
# This file is individualized for NetID dli106.
# ********
# No other imports are allowed
import itertools as it # in case you need it
import math # in case you need it
import sys
import numpy as np
import gomoku as gm

def minimax(game, state, max_depth=-1, evaluation_function=None, alpha=-np.inf, beta=np.inf):
    """
    depth-limited minimax with alpha-beta pruning and evaluation function
    default max_depth = -1 will not impose any depth limit
    default evaluation_function assigns zero to all states
    custom evaluation_function should accept game, state as input and return a number
    minimax returns (child state, child utility, node_count), where:
    - child_state is optimal child
    - child_utility is its utility (also the utility of the parent)
    - node_count is the total number of game states processed by the recursion
    """

    # default evaluation
    if evaluation_function is None: evaluation_function = (lambda g, s: 0)

    # base cases
    node_count = 1 # at least one game state (the current one) is processed
    if game.is_over_in(state): return None, game.score_in(state), node_count
    if max_depth == 0: return None, evaluation_function(game, state), node_count

    # setup alpha-beta pruning variables
    is_max = game.is_max_turn_in(state)
    bound = -np.inf if is_max else np.inf # bound is "v" in the minimax slides

    # recursive case
    children = []
    utilities = []
    for action in game.valid_actions_in(state):

        # recursively calculate child state utilities and node counts
        child_state = game.perform(action, state)
        _, utility, child_count = minimax(game, child_state, max_depth-1, evaluation_function, alpha, beta)

        # save results 
        children.append(child_state)
        utilities.append(utility)

        # TODO: add code here to update node_count based on the recursive call
        node_count += child_count

        # TODO: add code here for alpha-beta pruning
        # You can use similar code to the minimax slides, except:
        # - do not call minimax again, use the utility computed recursively above
        # - break the loop instead of returning from the function
        if is_max:
            bound = max(bound, utility)
            if bound >= beta:
                break
            alpha= max(alpha,bound)
        else:
            bound = min(bound, utility)
            if bound <= alpha:
                break
            beta = min(beta,bound)

        
    # return the results
    if is_max:
        best_index = np.argmax(utilities)
    else:
        best_index = np.argmin(utilities)

    return children[best_index], utilities[best_index], node_count

if __name__ == "__main__":

    game = gm.GomokuDomain(3, 3)
    state = game.initial_state()

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
            state, _, node_count = minimax(game, state)
            print("Total nodes processed:", node_count)

        # Human input for min player
        else:

            state = gm.human_turn(game, state)

