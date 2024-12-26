import numpy as np
import matplotlib.pyplot as pt
import catmouse_helpers as ch

# Domain API
# state (mx, my, cx, cy) are xy positions of mouse and cat
# Each can move one unit vertically, horizontally, or diagonally
# The cat moves uniformly at random, the mouse is the agent
# Reward is maximized by staying far from the cat

# The mouse and cat live in a discrete rectangular grid
# If they try to move outside the grid, they stay where they are
grid_rows, grid_cols = 10, 10
# grid_rows, grid_cols = 25, 25 # TODO: try larger grid size
num_states = (grid_rows*grid_cols)**2 # Total number of possible states

# The mouse agent has 9 actions: one unit in each direction, or staying in place
# dmx and dmy represent changes to position mx and my
actions = []
for dmx in [-1,0,1]:
    for dmy in [-1,0,1]:
        actions.append((dmx, dmy))

# Mathematical notation
N, K = num_states, len(actions)

### State-index mapping
# To use the MDP formalism, each state must be assigned a unique index
# This determines how to fill out the reward, probability, and utility arrays
# For this domain, we can treat mouse and cat coordinates as digits
# The base of the digits depends on the grid size
# For example, in a 10x10 grid, the coordinates are digits in a base-10 number:
# state mx, my, cx, cy = 0, 0, 0, 0 <-> index 0000
# state mx, my, cx, cy = 1, 0, 0, 0 <-> index 1000

def state_to_index(state):
    """
    This assigns each state a unique index
    Works essentially like algorithms that convert binary strings to ints,
    except that a non-square grid doesn't have a uniform base for every digit
    The coef variable is analogous to the powers of the base
    The elements of the state tuple are analogous to the digits
    """
    factors = [grid_cols, grid_rows, grid_cols, grid_rows]
    idx = 0
    coef = 1
    for i in range(4):
        digit = state[i]
        idx += digit*coef
        coef *= factors[i]
    return int(idx)

def index_to_state(idx):
    """
    This method is the inverse of "state_to_index":
    Given an integer index, it reconstructs the corresponding state.
    Works essentially like algorithms that convert ints to binary strings 
    """
    factors = [grid_cols, grid_rows, grid_cols, grid_rows]
    state = []
    for i in range(4):
        digit = idx % factors[i]
        idx = (idx - digit) / factors[i]
        state.append(digit)
    return tuple(state)

### Reward function
# Uses the distance from the cat as a reward
# This will make the mouse stay as far as possible from the cat
# Since diagonal motions are allowed, this uses the Chebyshev (chessboard) distance:
# https://en.wikipedia.org/wiki/Chebyshev_distance 
def reward(state):
    mx, my, cx, cy = state
    return max(np.fabs(mx-cx), np.fabs(my-cy))

### Reward array
r = np.array([reward(index_to_state(i)) for i in range(N)])

### Discount factor
ɣ = ch.get_discount_factor()

def plot_state(state):
    """
    state = (mx, my, cx, cy)
    Visualize a state with matplotlib:
        Blue circle is mouse
        Red circle is cat
    """
    mx, my, cx, cy = state
    pt.grid()
    pt.scatter(cx, cy, s=600, c='r')
    pt.scatter(mx, my, s=200, c='b')
    pt.xlim([-1, grid_cols])
    pt.ylim([-1, grid_rows])

### Performing actions
# Change the mouse coordinates by mdx and mdy
# Likewise for the cat
# Cat arguments default to None, in which case the cat motion is randomized
def move(state, mdx, mdy, cdx=None, cdy=None):

    # Randomize cat motions if not provided
    if cdx is None:
        cdx, cdy = np.random.choice([-1,0,1],size=2)

    # Unpack mouse and cat coordinates in the current state
    mx, my, cx, cy = state
 
    # animals stay at the same place if they try to move past the grid bounds
    mx = min(max(0, mx+mdx), grid_cols-1)
    my = min(max(0, my+mdy), grid_rows-1)
    cx = min(max(0, cx+cdx), grid_cols-1)
    cy = min(max(0, cy+cdy), grid_rows-1)

    return (mx, my, cx, cy)

"""
Using TD Q learning when probabilities and optimal utilities are not accessible
"""

# Initial Q estimates and counts
Q = np.zeros((N, K)) # Repeatedly updated during TD learning
visit_counts = np.zeros(N) # Tracks how many times each state was visited
choice_counts = np.zeros((N, K)) # How many times each action was done in each state

# Reward curve: tracks actual rewards received during learning (should improve)
reward_curve = []
display_period = 30000 # number of time-steps between visualizations
display_window = 10 # how long to animate the animals each visualization
num_buckets = 20 # how many buckets to use for averaging reward in the visualization

# Arbitrary initial state when learning begins
state = (0, 0, 3, 0)

# Total number of time-steps for learning
num_timesteps = 10**6
fig, axs = pt.subplots(1, 2, figsize=(7,3)) # subplots for visualization
for t in range(num_timesteps):

    # Get current state index and update visit count
    i = state_to_index(state)
    visit_counts[i] += 1

    # Update the reward curve with the current reward
    reward_curve.append(r[i])

    # Choose next action
    k = ch.choose_action(t, Q[i], choice_counts[i])

    # Update the action count and perform the chosen action
    choice_counts[i, k] += 1
    mdx, mdy = actions[k]
    state = move(state, mdx, mdy)

    # j is the new state index after the current action is performed
    j = state_to_index(state)

    # TD update rule
    # α is the learning rate, should get smaller over time
    α = ch.choose_learning_rate(t, k, Q[i], choice_counts[i], Q[j], choice_counts[j])
    Q[i,k] = (1-α) * Q[i,k] + α * (r[i] + ɣ * Q[j,:].max())

    # Visualize progress
    pt.ion()
    if False or (t % display_period < display_window):

        # Render the current state
        pt.sca(axs[0])
        pt.cla()
        plot_state(state)
        pt.title("Time-step %d" % t)
        
        # Plot the reward curve
        # For a smoother plot, average the reward curve in buckets
        pt.sca(axs[1])
        pt.cla()
        if t < num_buckets:
            pt.plot(reward_curve, 'k-')
        else:
            trim = int(t / num_buckets) * num_buckets
            reward_buckets = np.array(reward_curve[:trim]).reshape(num_buckets,-1).mean(axis=1)
            pt.plot(np.arange(0, trim, int(t / num_buckets)), reward_buckets, 'k-')
        pt.xlabel("Time-step")
        pt.ylabel("Reward")
        pt.title("Avg reward = %f" % np.mean(reward_curve))
        
        pt.pause(0.01)
        
        

