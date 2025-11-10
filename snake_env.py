import pygame
import random
import numpy as np

# --- Nokia logical screen ---
SCREEN_W = 84
SCREEN_H = 48
UI_H     = 8

# --- Playfield ---
PLAY_W = 72
PLAY_H = 32
MARGIN_X_PLAY = (SCREEN_W - PLAY_W) // 2
MARGIN_Y_PLAY = (SCREEN_H - UI_H - PLAY_H) // 2
MARGIN_X = MARGIN_X_PLAY
MARGIN_Y = UI_H + MARGIN_Y_PLAY

SCALE  = 8
WIDTH  = SCREEN_W * SCALE
HEIGHT = SCREEN_H * SCALE

# --- Colors ---
OUT_BG_COLOR      = (5, 20, 5)
PLAY_BG_COLOR     = (15, 56, 15)
UI_BG_COLOR       = (10, 35, 10)
BORDER_COLOR      = (0, 0, 0)
SNAKE_BODY_COLOR  = (48, 98, 48)
SNAKE_HEAD_COLOR  = (139, 172, 15)
FOOD_COLOR        = (0, 0, 0)
TEXT_COLOR        = (200, 200, 200)

SNAKE_SPEED = 12  # only used when rendering


class SnakeEnv:
    def __init__(self, render=False):
        """Simple Snake env for DQN."""
        self.render_mode = render
        self.action_space = 4  # left, right, up, down

        # pygame init only if render
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Snake - Nokia RL")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("bahnschrift", 16)
        else:
            self.screen = None
            self.clock = None
            self.font = None

        self.reset()

    # ---------- drawing helpers ----------

    def grid_to_screen(self, gx, gy):
        sx = (MARGIN_X + gx) * SCALE
        sy = (MARGIN_Y + gy) * SCALE
        return sx, sy

    def _draw_snake(self):
        if not self.snake:
            return
        # body
        for gx, gy in self.snake[:-1]:
            sx, sy = self.grid_to_screen(gx, gy)
            pygame.draw.rect(self.screen, SNAKE_BODY_COLOR,
                             [sx, sy, SCALE, SCALE])
        # head
        hx, hy = self.snake[-1]
        sx, sy = self.grid_to_screen(hx, hy)
        pygame.draw.rect(self.screen, SNAKE_HEAD_COLOR,
                         [sx, sy, SCALE, SCALE])

    def _draw_food(self):
        fx, fy = self.food
        sx, sy = self.grid_to_screen(fx, fy)
        pygame.draw.rect(self.screen, FOOD_COLOR,
                         [sx, sy, SCALE, SCALE])

    def _draw_ui(self):
        pygame.draw.rect(self.screen, UI_BG_COLOR,
                         [0, 0, WIDTH, UI_H * SCALE])
        score = len(self.snake) - 1
        text = self.font.render(f"Score: {score}", True, TEXT_COLOR)
        self.screen.blit(text, (4, 2))

    def _draw_playfield_bg_and_border(self):
        x = MARGIN_X * SCALE
        y = MARGIN_Y * SCALE
        w = PLAY_W * SCALE
        h = PLAY_H * SCALE
        pygame.draw.rect(self.screen, PLAY_BG_COLOR, [x, y, w, h])
        pygame.draw.rect(self.screen, BORDER_COLOR, [x, y, w, h], width=1)

    # ---------- RL API ----------

    def reset(self):
        """Reset env and return state."""
        self.gx = PLAY_W // 2
        self.gy = PLAY_H // 2
        self.dx, self.dy = 1, 0  # moving right
        self.snake = [[self.gx, self.gy]]
        self.food = self._random_food_position()
        self.done = False
        return self._get_state()

    def step(self, action):
        """
        action: 0=LEFT,1=RIGHT,2=UP,3=DOWN
        returns: state, reward, done, info
        """
        if self.done:
            raise ValueError("Call reset() before step() after done=True")

        # update direction, avoid reverse
        if action == 0 and not (self.dx == 1 and self.dy == 0):
            self.dx, self.dy = -1, 0
        elif action == 1 and not (self.dx == -1 and self.dy == 0):
            self.dx, self.dy = 1, 0
        elif action == 2 and not (self.dx == 0 and self.dy == 1):
            self.dx, self.dy = 0, -1
        elif action == 3 and not (self.dx == 0 and self.dy == -1):
            self.dx, self.dy = 0, 1

        # move head
        self.gx += self.dx
        self.gy += self.dy

        reward = -0.01  # small step penalty

        # wall collision
        if self.gx < 0 or self.gx >= PLAY_W or self.gy < 0 or self.gy >= PLAY_H:
            self.done = True
            reward = -1.0
        else:
            new_head = [self.gx, self.gy]
            # self-collision
            if new_head in self.snake:
                self.done = True
                reward = -1.0
            else:
                # normal move
                self.snake.append(new_head)
                # food
                if self.gx == self.food[0] and self.gy == self.food[1]:
                    reward = 1.0
                    self.food = self._random_food_position()
                else:
                    # remove tail
                    del self.snake[0]

        state = self._get_state()
        info = {}
        return state, reward, self.done, info

    def render(self):
        """Render one frame (only if render_mode=True)."""
        if not self.render_mode:
            return
        # handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill(OUT_BG_COLOR)
        self._draw_ui()
        self._draw_playfield_bg_and_border()
        self._draw_food()
        self._draw_snake()
        pygame.display.update()
        self.clock.tick(SNAKE_SPEED)

    def close(self):
        if self.render_mode:
            pygame.quit()

    # ---------- helpers internal ----------

    def _random_food_position(self):
        while True:
            fx = random.randrange(0, PLAY_W)
            fy = random.randrange(0, PLAY_H)
            if [fx, fy] not in self.snake:
                return [fx, fy]

    def _get_state(self):
        """Build state vector for DQN."""
        head_x_norm = self.gx / PLAY_W
        head_y_norm = self.gy / PLAY_H
        food_x_norm = self.food[0] / PLAY_W
        food_y_norm = self.food[1] / PLAY_H

        # distances to walls (normalized)
        dist_left  = self.gx / PLAY_W
        dist_right = (PLAY_W - 1 - self.gx) / PLAY_W
        dist_up    = self.gy / PLAY_H
        dist_down  = (PLAY_H - 1 - self.gy) / PLAY_H

        state = np.array([
            head_x_norm, head_y_norm,
            food_x_norm, food_y_norm,
            self.dx, self.dy,
            dist_left, dist_right, dist_up, dist_down
        ], dtype=np.float32)

        return state
