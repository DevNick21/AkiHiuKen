# Import necessary libraries
import numpy as np
import random
import pygame
import pandas as pd
import time
import csv


# columns represent states, rows represent actions
# Grid size is 6x6
GRID_SIZE = 6

# Reward and penalty constants
REWARD = 1.0
PENALTY = -1.0

# Idle penalty incase agent decides to take no action
IDLE_PENALTY = -0.5
SCREEN_SIZE = 600
CELL_SIZE = SCREEN_SIZE // GRID_SIZE
BACKGROUND_COLOR = (185, 122, 87)  # Light brown
HOLE_COLOR = (139, 69, 19)         # Dark brown
HIT_COLOR = (0, 255, 0)            # Green for hit
MISS_COLOR = (255, 0, 0)

#  No. of moles in the grid
pattern_num = GRID_SIZE * GRID_SIZE // 2

# N_states representing states and actions (Indexes)
n_states = GRID_SIZE * GRID_SIZE


# q_table represents Q-values for each state-action pair
# initialized as a 3D array, where the first dimension represents the pattern number
# or number of moles
# the second dimension represents the states and the third dimension represents the actions
# in or cause the states and actions are the same number because in a whack a mole game,
# moving from one cell to another is considered an action and also a change of state
q_tables = np.zeros((pattern_num, n_states, n_states))

# Learning rate determines how quickly the Q-values change or how fast the agent decides to
# consider its current Q-value and the Q-value of the next state (Memory)
learning_rate = 0.1

# Discount factor determines how much the agent values future rewards compared to immediate rewards
discount_factor = 0.5

# Epsilon determines the probability of choosing a random action instead of the best action
epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.01

# Number of episodes to train the agent
episodes = 5000

# Seed for reproducibility
to_be = 42

#! EXOGENOIUS INFORMATION, The randomness is introduced in this code to make the agent's behavior more unpredictable
#! and the environment more challenging
# Using Psuedo random number generator for reproducibility
pattern_rng = random.Random(to_be)
exploration_rng = random.Random(to_be * 2)
state_rng = random.Random((to_be * 2) * 2)

# Patterns (Mole positions in the sequence)
patterns = [pattern_rng.randint(0, GRID_SIZE*GRID_SIZE - 1)
            for _ in range(pattern_num)]

# States (Indexes representing positions in the grid)
states = [x for x in range(0, n_states)]

# Helper function load image from a file and resize it if necessary


def load_image(image_path, size=None, default_color=None):
    """
    Load an image from the specified path, and optionally resize it and fill it with a default color if the image loading fails.

    Parameters:
    image_path (str): The path to the image file.
    size (tuple, optional): The desired size of the image. If not provided, the image will be loaded in its original size.
    default_color (tuple, optional): The color to fill the image if the loading fails. If not provided, the image will not be filled.

    Returns:
    pygame.Surface: The loaded and possibly resized image. If the loading fails and a default color is provided, a filled surface will be returned.
    """
    try:
        img = pygame.image.load(image_path)
        if size:
            img = pygame.transform.scale(img, size)
        return img
    except pygame.error:
        if default_color:
            img = pygame.Surface((CELL_SIZE, CELL_SIZE))
            img.fill(default_color)
        return img


def take_action(state, action, pattern):
    """
    Simulates taking an action in the Whack-A-Mole game and calculates the reward.

    This function determines the new state, implements the reward, and checks if the action
    matches the current pattern (mole position).

    Parameters:
    state (int): The current state (position) of the agent.
    action (int): The action (new position) chosen by the agent.
    pattern (int): The current pattern (mole position) to match.

    Returns:
    tuple: A tuple containing three elements:
        - new_state (int): The new state after taking the action (same as the action).
        - reward_gotten (float): The reward obtained from the action.
        - match_index (bool): True if the action matches the pattern, False otherwise.
    """
    new_state = action
    match_index = False
    reward_gotten = 0.0
    # Checking if the action matches the pattern state
    if new_state == pattern:
        reward_gotten += REWARD
        match_index = True
    elif new_state == state:
        reward_gotten += IDLE_PENALTY
    else:
        reward_gotten += PENALTY

    return new_state, reward_gotten, match_index


# Loads the image of the Mole with a blue placeholder
DIGLETT_IMAGE = load_image(
    "red.png", (CELL_SIZE // 2, CELL_SIZE // 2), (0, 0, 255))

# Cell class representing a hole in the Whack-A-Mole game
# It has method to draw the cell on the game screen and
# show a mole if the cell contains a mole
# It also has methods to toggle the cell based on whether
# it contains a mole and whether it has been hit or missed


class Cell:
    def __init__(self, x, y):
        """
        Initialize a Cell object representing a hole in the Whack-A-Mole game.

        This function sets up the initial state of a cell, including its position,
        dimensions, and various attributes related to the game mechanics.

        Parameters:
        x (int): The x-coordinate of the top-left corner of the cell.
        y (int): The y-coordinate of the top-left corner of the cell.

        Attributes:
        rect (pygame.Rect): A rectangle representing the cell's boundaries.
        hole_center (tuple): The (x, y) coordinates of the center of the hole.
        has_mole (bool): Indicates whether the cell currently contains a mole.
        selected_color (None or tuple): The color to use when the cell is selected (hit or miss).
        cell_index (int): A unique index for the cell based on its position in the grid.

        Returns:
        None
        """
        self.x = x
        self.y = y
        self.rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
        self.hole_center = (x + CELL_SIZE // 2, y + CELL_SIZE // 2)
        self.has_mole = False
        self.selected_color = None
        self.cell_index = int((self.x * GRID_SIZE + self.y)/100)

    def draw(self):
        """
        Draw the cell on the game screen.

        This method renders the cell as a circle representing a hole, and if the cell
        contains a mole, it also draws the mole image.

        Parameters:
        None

        Returns:
        None

        Side effects:
        - Draws a circle on the game screen to represent the hole.
        - If the cell has a mole, draws the mole image on top of the hole.
        """
        color = self.selected_color if self.selected_color else HOLE_COLOR
        pygame.draw.circle(screen, color, self.hole_center, CELL_SIZE // 3)
        if self.has_mole:
            mole_pos = (self.hole_center[0] - DIGLETT_IMAGE.get_width() //
                        2, self.hole_center[1] - DIGLETT_IMAGE.get_height() // 2)
            screen.blit(DIGLETT_IMAGE, mole_pos)

    # Toggles mole status, which helps the draw method
    def toggle_mole(self):
        self.has_mole = not self.has_mole

    # Set cell color based on hit or miss
    def set_hit(self):
        self.selected_color = HIT_COLOR

    def set_miss(self):
        self.selected_color = MISS_COLOR

    # Reset cell color
    def reset_color(self):
        self.selected_color = None


# Create grid of cells
cells = [[Cell(x * CELL_SIZE, y * CELL_SIZE)
          for x in range(GRID_SIZE)] for y in range(GRID_SIZE)]

# using the clock to control the frame rate
clock = pygame.time.Clock()
mole_timer = 0
all_stats = []

# Boolean flag to indicate whether the game is running or not
running = True

# Initialize screen, fill it with the background color, and set the caption
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("AkiHiuKen")
screen.fill(BACKGROUND_COLOR)

# Game loop (Including visual and AI logic)
while running:
    screen.fill(BACKGROUND_COLOR)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

# Iterations to improve the agent's performance
    for episode in range(episodes):

        #! STATE VARIABLE: Initial state of the agent which is pseudo-randomly generated at the start of each episode
        # Initialize agent state and pattern index at the start of each episode
        state = state_rng.randint(0, n_states - 1)
        current_pattern_index = 0
        start_time = time.time()

        # Objective function results to store the total reward obtained in an episode,
        # the number of times the agent failed to match the pattern,
        # and the current epsilon value
        #! OBJECTIVE FUNCTION: Total reward obtained in an episode and Match failure count
        stats = {"Episode": episode + 1, "match_fail_count": 0, "reward_per_episode": 0.0,
                 "epsilon": round(epsilon, 3), "time_per_episode": 0.0}

        # Boolean flag to indicate whether an episode is done or not
        done = False

        # Main training loop per episode
        while not done:
            # Get the current mole position in the pattern list
            pattern_position = patterns[current_pattern_index]

            # TODO: Visualization breaks in speeds, so the learning can be seen
            # if episode < 3 or episode > 4500:
            #     mole_timer += clock.tick(30)
            # else:  # Frame rate adjustment
            #     mole_timer += clock.tick(0)  # Frame rate adjustment

            # Resetting the Moles per frame rate
            for row in cells:
                for cell in row:
                    cell.has_mole = False
                    cell.reset_color()

            # Toggling the current mole in the pattern
            for row in cells:
                for cell in row:
                    if cell.cell_index == patterns[current_pattern_index]:
                        cell.toggle_mole()

            # Get the current agent state from outside the training loop
            for row in cells:
                for cell in row:
                    if cell.cell_index == states[state]:
                        current_agent_cell = cell

            # Choosing action based on epsilon-greedy policy
            # Randomly choose action if epsilon is higher than the random num,
            # otherwise choose action with highest Q-value for current state and pattern.
            #! DECISION VARIABLE: THE AGENT ACT ON THE EPISON-GREEDY POLICY
            if exploration_rng.uniform(0, 1) < epsilon:
                action = exploration_rng.choice(states)  # Exploration
            else:
                action = int(np.argmax(
                    q_tables[current_pattern_index, state]))  # Exploitation

            # Take action function to get new state, reward,
            # and whether the action matches the pattern state or not
            new_state, reward_obtained, match = take_action(
                state, action, pattern_position)

            # Adding the stats to the episode statistics
            stats["reward_per_episode"] += reward_obtained

            # Updating Q-table

            # Accesses the Q-value of the current state and action
            old_value = q_tables[current_pattern_index, state, action]

            # Compute the maximum Q-value for the next state
            next_max = np.max(q_tables[current_pattern_index, new_state, :])

            #! TRANSITION FUNCTION: THE AGENT LEARNS FROM THE REWARD AND UPDATE THE Q-TABLE
            # Update the Q-value for the current state and action using the Q-learning update rule
            q_tables[current_pattern_index, state, action] = old_value + learning_rate * \
                (reward_obtained + discount_factor * next_max - old_value)

            # resetting the current agent cell color and position for the change (hit or miss)
            if current_agent_cell:
                current_agent_cell.reset_color()

            # Setting the new state of the agent to the new state
            for row in cells:
                for cell in row:
                    if cell.cell_index == states[new_state]:
                        current_agent_cell = cell

            # Checking if the action matches the pattern state or not,
            # and updating the state accordingly.
            # also setting the current agent cell color and position for the change (hit or miss)
            if match:
                current_agent_cell.set_hit()
                current_pattern_index += 1
                state = state_rng.randint(0, n_states - 1)
                if current_pattern_index == len(patterns):
                    done = True
            else:
                state = new_state
                current_agent_cell.set_miss()
                stats["match_fail_count"] += 1

            # Updating the game screen
            for row in cells:
                for cell in row:
                    cell.draw()

            pygame.display.flip()

        # Printing episode statistics and epsilon value
        stats["time_per_episode"] = round(time.time() - start_time, 4)
        all_stats.append(stats)
        print(f"{stats}\n")

        # Reducing epsilon for next episode
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Early stopping condition: if min epsilon is reached, stop the training
        if epsilon == min_epsilon:
            running = False
            break

pygame.quit()

# Saving the Q-table to a CSV file for analysis
data = []
for pattern_index in range(q_tables.shape[0]):
    for state in range(q_tables.shape[1]):
        for action in range(q_tables.shape[2]):
            data.append([pattern_index, state, action,
                        q_tables[pattern_index, state, action]])

q_tables_df = pd.DataFrame(
    data, columns=["Pattern Index", "State", "Action", "Q-Value"])

q_tables_df.to_csv("q_tables.csv", index=False)

# Saving the episode statistics to a CSV file for analysis
csv_file = "episode_stats.csv"
with open(csv_file, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=stats.keys())

    # Write the header row
    writer.writeheader()

    # Write each episode's stats as a new row
    for stats in all_stats:
        writer.writerow(stats)
