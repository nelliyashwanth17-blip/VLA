import numpy as np


def heuristic_action(observation, grid_size=6):
    """Simple non-learning reference policy using the same state/instruction inputs."""
    state = observation["state"]
    red = np.rint(state[:2] * (grid_size - 1)).astype(int)
    blue = np.rint(state[2:] * (grid_size - 1)).astype(int)
    move_red = observation["instruction"][0] > 0.5

    moving = red if move_red else blue
    fixed = blue if move_red else red

    # Choose the legal one-cell move that minimizes Manhattan distance.
    moves = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1),
    }
    best_action, best_distance = 0, 10**9
    for action, delta in moves.items():
        candidate = np.clip(moving + np.array(delta), 0, grid_size - 1)
        if np.array_equal(candidate, fixed):
            continue
        distance = int(np.abs(candidate - fixed).sum())
        if distance < best_distance:
            best_action, best_distance = action, distance
    return best_action
