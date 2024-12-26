from time import perf_counter
import itertools as it
import numpy as np
import gomoku as gm
from minimax import minimax
from evaluation import simple_evaluator, better_evaluator

# runs a competitive game between two AIs
def compete(verbose=False):

    # do not change these parameters, they are used for grading in the instructor's copy
    board_size = 6
    win_size = 5
    num_reps = 1
    max_depth = 3
    evaluators = {gm.MAX: simple_evaluator, gm.MIN: better_evaluator}

    game = gm.GomokuDomain(board_size, win_size)
    timing = {player: np.zeros(num_reps) for player in (gm.MIN, gm.MAX)}

    for rep in range(num_reps):

        if verbose: print(f"Starting game {rep}...")
        state = game.initial_state()

        for turn in it.count():
    
            # print current state
            if verbose: print(game.string_of(state))

            # stop if game over
            if game.is_over_in(state): break

            # get current player and evaluation function
            player = game.current_player_in(state)
            evaluation_fn = evaluators[player]
            
            # select next state with current player's evaluator, time the turn
            turn_start = perf_counter()
            state, utility, node_count = minimax(game, state, max_depth, evaluation_fn)
            turn_duration = perf_counter() - turn_start

            # update metrics
            timing[player][rep] += turn_duration
            if verbose: print(f"Turn {turn}: Processed {node_count} nodes in {turn_duration} seconds")

        score = game.score_in(state)
        if verbose: print(f"Game {rep} over, score = {score}")

    if verbose: print("All games over")
    if verbose: print(f"Max player: {timing[gm.MAX].min()} seconds total")
    if verbose: print(f"Min player: {timing[gm.MIN].min()} seconds total")
        
    return score, timing

if __name__ == "__main__":

    score, timing = compete(verbose=True)

    if score < 0:
        print("Better evaluator wins, test PASSED (eligible for bonus credit)")
    else:
        print("Better evaluator does not win, test FAILED (ineligible for bonus credit)")

