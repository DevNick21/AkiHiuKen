import numpy as np
import random
import pygame


# columns represent states, rows represent actions
GRID_SIZE = 6
pattern_num = GRID_SIZE * GRID_SIZE // 2
n_states = GRID_SIZE * GRID_SIZE
q_tables = np.zeros((pattern_num, n_states, n_states))
learning_rate = 0.1
discount_factor = 0.5
epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.01
episodes = 5000
REWARD = 1.0
IDLE_PENALTY = -0.5
PENALTY = -1.0

SCREEN_SIZE = 600
CELL_SIZE = SCREEN_SIZE // GRID_SIZE
BACKGROUND_COLOR = (185, 122, 87)  # Light brown
HOLE_COLOR = (139, 69, 19)         # Dark brown
HIT_COLOR = (0, 255, 0)            # Green for hit
MISS_COLOR = (255, 0, 0)

to_be = 42
# Using Psuedo random number generator for reproducibility
pattern_rng = random.Random(to_be)
exploration_rng = random.Random(to_be * 2)
state_rng = random.Random((to_be * 2) * 2)

# print(q_table)
# Patterns
patterns = [pattern_rng.randint(0, GRID_SIZE*GRID_SIZE - 1)
            for _ in range(pattern_num)]
states = [x for x in range(0, n_states)]


def load_image(image_path, size=None, default_color=None):
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


DIGLETT_IMAGE = load_image(
    "red.png", (CELL_SIZE // 2, CELL_SIZE // 2), (255, 0, 0))  # Red placeholder
# HAMMER_IMAGE = load_image("hammer.png", (CELL_SIZE, CELL_SIZE), (0, 0, 255))  # Blue placeholder

# Initialize screen
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("AkiHiuKen")
screen.fill(BACKGROUND_COLOR)


class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
        self.hole_center = (x + CELL_SIZE // 2, y + CELL_SIZE // 2)
        self.has_mole = False
        self.selected_color = None
        self.cell_index = int((self.x * GRID_SIZE + self.y)/100)

    def draw(self):
        color = self.selected_color if self.selected_color else HOLE_COLOR
        pygame.draw.circle(screen, color, self.hole_center, CELL_SIZE // 3)
        if self.has_mole:
            mole_pos = (self.hole_center[0] - DIGLETT_IMAGE.get_width() //
                        2, self.hole_center[1] - DIGLETT_IMAGE.get_height() // 2)
            screen.blit(DIGLETT_IMAGE, mole_pos)

    def toggle_mole(self):
        self.has_mole = not self.has_mole

    def set_hit(self):
        self.selected_color = HIT_COLOR

    def set_miss(self):
        self.selected_color = MISS_COLOR

    def reset_color(self):
        self.selected_color = None


# Create grid of cells
cells = [[Cell(x * CELL_SIZE, y * CELL_SIZE)
          for x in range(GRID_SIZE)] for y in range(GRID_SIZE)]

clock = pygame.time.Clock()
mole_timer = 0
# hammer_timer = 0
# mole_interval = 10  # Mole pops every second
# hammer_interval = 1000

running = True


while running:
    screen.fill(BACKGROUND_COLOR)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False


# TODO : Match Fail count against episodes
# TODO : Epsilon
# Main loop for the episodes
    for episode in range(episodes):
        state = state_rng.randint(0, n_states - 1)
        current_pattern_index = 0

        stats = {"match_fail_count": 0, "reward_per_episode": 0.0,
                 "epsilon": round(epsilon, 3)}
        done = False
        while not done:
            pattern_position = patterns[current_pattern_index]
            for row in cells:
                for cell in row:
                    if cell.cell_index == states[state]:
                        current_hammer_cell = cell
            if episode > 1500:
                mole_timer += clock.tick(50)  # Frame rate adjustment
            mole_timer += clock.tick(0)  # Frame rate adjustment
            # if mole_timer > mole_interval:
            #     mole_timer = 0
            for row in cells:
                for cell in row:
                    cell.has_mole = False
                    cell.reset_color()
            for row in cells:
                for cell in row:
                    if cell.cell_index == patterns[current_pattern_index]:
                        cell.toggle_mole()
            # Choosing action based on epsilon-greedy policy
            # Randomly choose action if epsilon is higher than the random num, otherwise choose action with highest Q-value for current state and pattern.
            if exploration_rng.uniform(0, 1) < epsilon:
                action = exploration_rng.choice(states)  # Exploration
            else:
                action = int(np.argmax(
                    q_tables[current_pattern_index, state]))  # Exploitation

            # Take action function to get new state, reward, and whether the action matches the pattern state or not
            new_state, reward_obtained, match = take_action(
                state, action, pattern_position)

            stats["reward_per_episode"] += reward_obtained
            # Update Q-table
            # Accesses the Q-value of the current state and action
            old_value = q_tables[current_pattern_index, state, action]

            next_max = np.max(q_tables[current_pattern_index, new_state, :])

            q_tables[current_pattern_index, state, action] = old_value + learning_rate * \
                (reward_obtained + discount_factor * next_max - old_value)

            # hammer_timer += clock.get_time()
            # # hammer_timer += clock.tick(30)
            # if hammer_timer > hammer_interval:
            #     hammer_timer = 0
            if current_hammer_cell:
                current_hammer_cell.reset_color()
            for row in cells:
                for cell in row:
                    if cell.cell_index == states[new_state]:
                        current_hammer_cell = cell

            if match:
                current_hammer_cell.set_hit()
                current_pattern_index += 1
                state = state_rng.randint(0, n_states - 1)
                if current_pattern_index == len(patterns):
                    done = True
            else:
                state = new_state
                current_hammer_cell.set_miss()
                stats["match_fail_count"] += 1

            for row in cells:
                for cell in row:
                    cell.draw()
            pygame.display.flip()

        print(f"Episode {episode+1}: {stats}\n, {round(epsilon, 2)}")
        # Reduces epsilon for next episode
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Early stopping condition: if min epsilon is reached, stop the training
        if epsilon == min_epsilon:
            running = False
            break

pygame.quit()
