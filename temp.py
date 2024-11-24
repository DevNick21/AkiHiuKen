import numpy as np
import random


# columns represent states, rows represent actions
grid_size = 6
pattern_num = grid_size * grid_size // 2
n_states = grid_size * grid_size
n_actions = grid_size * grid_size
q_tables = np.zeros((pattern_num, n_states, n_actions))
learning_rate = 0.1
discount_factor = 0.5
epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.1
episodes = 1000
REWARD = 1.0
IDLE_PENALTY = -0.5
PENALTY = -1.0

to_be = 42
# Using Psuedo random number generator for reproducibility
pattern_rng = random.Random(to_be)
exploration_rng = random.Random(to_be * 2)
state_rng = random.Random((to_be * 2) * 2)

# Patterns
patterns = [pattern_rng.randint(0, grid_size*grid_size - 1)
            for _ in range(pattern_num)]

# print(q_table)

actions = [x for x in range(0, n_actions)]


# grid = np.zeros((3, 3))
# print(grid)


def take_action(state, action, pattern):
    new_position = actions[action]
    # print(new_x, new_y)
    match_index = False
    reward_gotten = 0.0

    # Checking if the action matches the pattern state
    if new_position == pattern:
        reward_gotten += REWARD
        match_index = True
    elif new_position == state:
        reward_gotten += IDLE_PENALTY
    else:
        reward_gotten += PENALTY

    return new_position, reward_gotten, match_index


stats = {f"mole {i+1}": {"match_count": 0, "match_fail_count": 0}
         for i in range(len(patterns))}


for episode in range(episodes):
    state = state_rng.randint(0, n_states - 1)
    current_pattern_index = 0
    done = False

    while not done:
        pattern_position = patterns[current_pattern_index]
        # Choosing action based on epsilon-greedy policy
        # Randomly choose action if epsilon is higher than the random num, otherwise choose action with highest Q-value for current state and pattern.
        if exploration_rng.uniform(0, 1) < epsilon:
            action = exploration_rng.choice(actions)  # Exploration
        else:
            action = int(np.argmax(
                q_tables[current_pattern_index, state]))  # Exploitation
        # Take action function to get new state, reward, and whether the action matches the pattern state or not
        position, reward_obtained, match = take_action(
            state, action, pattern_position)
        # print(
        #     f"State: {state}, Action: {action}, Reward: {reward_obtained}, Match: {match}")

        # Update Q-table
        # Accesses the Q-value of the current state and action
        old_value = q_tables[current_pattern_index, state, action]

        next_max = np.max(q_tables[current_pattern_index, position, :])

        new_value = old_value + learning_rate * \
            (reward_obtained + discount_factor * next_max - old_value)

        q_tables[current_pattern_index, position] = new_value

        if match:
            stats[f"mole {current_pattern_index + 1}"]["match_count"] += 1
            current_pattern_index += 1
            state = state_rng.randint(0, n_states - 1)
            if current_pattern_index == len(patterns):
                done = True
        else:
            state = position
            stats[f"mole {current_pattern_index+1}"]["match_fail_count"] += 1

    print(f"Episode {episode+1}: {stats}\n")
    # Reduces epsilon for next episode
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
