import numpy as np
import random


# columns represent states, rows represent actions
grid_size = 6
pattern_num = grid_size
n_states = grid_size * grid_size
to_be = 42
n_actions = grid_size * grid_size
q_tables = np.zeros((pattern_num, n_states, n_actions))
actions = [x for x in range(n_actions)]
pattern_rng = random.Random(to_be)

patterns = {pattern_rng.randint(0, grid_size*grid_size - 1): round(pattern_rng.uniform(0.6, 0.8), 2)
            for _ in range(grid_size * grid_size // 2)}
print(patterns)
