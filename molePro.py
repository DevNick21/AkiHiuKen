import pygame

pygame.init()

SCREEN_SIZE = 600
GRID_SIZE = 3
CELL_SIZE = SCREEN_SIZE // GRID_SIZE

BACKGROUND_COLOR = (150, 200, 100)  # Light green
SQUARE_COLOR = (185, 122, 87)       # Light brown for squares
CIRCLE_COLOR = (139, 69, 19)        # Dark brown for circles

screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("AkiHiuKen")


screen.fill(BACKGROUND_COLOR)

for x in range(GRID_SIZE):
    for y in range(GRID_SIZE):
        # Calculate the center of each cell
        center_x = x * CELL_SIZE + CELL_SIZE // 2
        center_y = y * CELL_SIZE + CELL_SIZE // 2

        # Draw a square for each cell
        pygame.draw.rect(screen, SQUARE_COLOR, (center_x - CELL_SIZE //
                         2, center_y - CELL_SIZE // 2, CELL_SIZE, CELL_SIZE))

        # Draw a circle inside each square
        pygame.draw.circle(screen, CIRCLE_COLOR,
                           (center_x, center_y), CELL_SIZE // 3)

# Update the display once
pygame.display.flip()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
