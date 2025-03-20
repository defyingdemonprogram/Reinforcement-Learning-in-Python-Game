import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
try:
    font = pygame.font.SysFont("../assets/fonts/iosevka-regular.ttf", 25)
except Exception as err:
    print(f"[ERROR]: Unable to load font due to {err}")
    font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# RGB Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# Game Constants
BLOCK_SIZE = 20
SPEED = 40

class SnakeGameAI:

    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """Reset the game state to initial values."""
        self.direction = Direction.RIGHT
        self.head = Point(self.width // 2, self.height // 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        """Randomly place the food on the grid, ensuring it's not on the snake."""
        x = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        """Run one step of the game based on the action."""
        self.frame_iteration += 1

        # Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Move the snake
        self._move(action)
        self.snake.insert(0, self.head)

        # Check if the game is over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # Place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # Update the display and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # Return the current state
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        """Check if the snake collides with the wall or itself."""
        if pt is None:
            pt = self.head
        # Check boundary collisions
        if pt.x >= self.width or pt.x < 0 or pt.y >= self.height or pt.y < 0:
            return True
        # Check self-collision
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        """Update the user interface (display)."""
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        score_text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(score_text, [10, 10])
        pygame.display.flip()

    def _move(self, action):
        """Update the snake's direction based on the action."""
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        current_idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_direction = clock_wise[current_idx]  # No change
        elif np.array_equal(action, [0, 1, 0]):
            new_direction = clock_wise[(current_idx + 1) % 4]  # Right turn
        else:  # [0, 0, 1]
            new_direction = clock_wise[(current_idx - 1) % 4]  # Left turn

        self.direction = new_direction
        self.head = self._calculate_new_head_position()

    def _calculate_new_head_position(self):
        """Calculate the new position of the snake's head based on the current direction."""
        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        return Point(x, y)
