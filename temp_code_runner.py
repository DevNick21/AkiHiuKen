import numpy as np
import random
import time

# columns represent states, rows represent actions
grid_size = 3
pattern_num = grid_size * grid_size // 2
n_states = grid_size * grid_size
to_be = 42
n_actions = grid_size * grid_size
q_tables = np.zeros((pattern_num, n_states, n_actions))
actions = [x for x in range(n_actions)]
pattern_rng = random.Random(to_be)

patterns = [pattern_rng.randint(0, grid_size*grid_size - 1)
            for _ in range(pattern_num)]

start = time.time()
time.sleep(1)  # Simulating some computationally expensive task
end = time.time()
print(end - start)
