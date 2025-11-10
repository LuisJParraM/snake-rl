import pygame
import random

pygame.init()

# --- Nokia logical screen ---
SCREEN_W = 84   # total width (Nokia)
SCREEN_H = 48   # total height (Nokia)
UI_H     = 8    # top bar height (score area)

# --- Playfield (centered inside screen) ---
PLAY_W = 72     # playfield width (cells)
PLAY_H = 32     # playfield height (cells)

MARGIN_X_PLAY = (SCREEN_W - PLAY_W) // 2             # left/right margin
MARGIN_Y_PLAY = (SCREEN_H - UI_H - PLAY_H) // 2      # margin between UI and bottom
MARGIN_X = MARGIN_X_PLAY
MARGIN_Y = UI_H + MARGIN_Y_PLAY                      # top of playfield

# --- Scale factor (for PC window) ---
SCALE  = 8
WIDTH  = SCREEN_W * SCALE
HEIGHT = SCREEN_H * SCALE
SNAKE_SPEED = 12

# --- Colors ---
OUT_BG_COLOR      = (5, 20, 5)       # outside playfield
PLAY_BG_COLOR     = (15, 56, 15)     # inside playfield
UI_BG_COLOR       = (10, 35, 10)     # top bar
BORDER_COLOR      = (0, 0, 0)
SNAKE_BODY_COLOR  = (48, 98, 48)
SNAKE_HEAD_COLOR  = (139, 172, 15)
FOOD_COLOR        = (0, 0, 0)
TEXT_COLOR        = (200, 200, 200)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake - Nokia style")

clock = pygame.time.Clock()
font_style = pygame.font.SysFont("bahnschrift", 16)


# --- Helpers ---

def grid_to_screen(gx, gy):
    """Grid -> screen coords (scaled)."""
    sx = (MARGIN_X + gx) * SCALE
    sy = (MARGIN_Y + gy) * SCALE
    return sx, sy


def draw_snake(snake_list):
    """Draw snake body and head."""
    if not snake_list:
        return
    # body
    for gx, gy in snake_list[:-1]:
        sx, sy = grid_to_screen(gx, gy)
        pygame.draw.rect(screen, SNAKE_BODY_COLOR, [sx, sy, SCALE, SCALE])
    # head
    hx, hy = snake_list[-1]
    sx, sy = grid_to_screen(hx, hy)
    pygame.draw.rect(screen, SNAKE_HEAD_COLOR, [sx, sy, SCALE, SCALE])


def draw_food(fx, fy):
    """Draw food."""
    sx, sy = grid_to_screen(fx, fy)
    pygame.draw.rect(screen, FOOD_COLOR, [sx, sy, SCALE, SCALE])


def draw_ui(score):
    """Draw top bar and score."""
    pygame.draw.rect(screen, UI_BG_COLOR, [0, 0, WIDTH, UI_H * SCALE])
    text = font_style.render(f"Score: {score}", True, TEXT_COLOR)
    screen.blit(text, (4, 2))


def draw_playfield_bg_and_border():
    """Draw playfield background and border."""
    # playfield background (lighter green)
    x = MARGIN_X * SCALE
    y = MARGIN_Y * SCALE
    w = PLAY_W * SCALE
    h = PLAY_H * SCALE
    pygame.draw.rect(screen, PLAY_BG_COLOR, [x, y, w, h])
    # border (black rectangle)
    pygame.draw.rect(screen, BORDER_COLOR, [x, y, w, h], width=1)


def message_center(msg):
    """Draw centered message."""
    mesg = font_style.render(msg, True, TEXT_COLOR)
    rect = mesg.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(mesg, rect)


def random_food_position():
    """Random cell inside playfield."""
    fx = random.randrange(0, PLAY_W)
    fy = random.randrange(0, PLAY_H)
    return fx, fy


# --- Main game loop ---

def gameLoop():
    game_over = False
    game_close = False

    # snake head in grid coords (playfield)
    gx = PLAY_W // 2
    gy = PLAY_H // 2

    dx, dy = 1, 0         # moving right
    direction = "RIGHT"

    snake_list = []
    snake_len = 1

    food_x, food_y = random_food_position()

    while not game_over:

        while game_close:
            screen.fill(OUT_BG_COLOR)
            draw_ui(score=snake_len - 1)
            draw_playfield_bg_and_border()
            message_center("You lost! Q=Quit, C=Retry")
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_over = True
                    game_close = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_over = True
                        game_close = False
                    if event.key == pygame.K_c:
                        gameLoop()  # restart

        # --- Input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True

            if event.type == pygame.KEYDOWN:
                # no reverse (no 180° turn)
                if event.key == pygame.K_LEFT and direction != "RIGHT":
                    dx, dy = -1, 0
                    direction = "LEFT"
                elif event.key == pygame.K_RIGHT and direction != "LEFT":
                    dx, dy = 1, 0
                    direction = "RIGHT"
                elif event.key == pygame.K_UP and direction != "DOWN":
                    dx, dy = 0, -1
                    direction = "UP"
                elif event.key == pygame.K_DOWN and direction != "UP":
                    dx, dy = 0, 1
                    direction = "DOWN"

        # --- Move snake in grid ---
        gx += dx
        gy += dy

        # wall collision (playfield)
        if gx < 0 or gx >= PLAY_W or gy < 0 or gy >= PLAY_H:
            game_close = True

        # draw outside bg + UI
        screen.fill(OUT_BG_COLOR)
        draw_ui(score=snake_len - 1)

        # playfield background + border
        draw_playfield_bg_and_border()

        # food
        draw_food(food_x, food_y)

        # snake update
        head = [gx, gy]
        snake_list.append(head)
        if len(snake_list) > snake_len:
            del snake_list[0]

        # self collision
        for segment in snake_list[:-1]:
            if segment == head:
                game_close = True

        # draw snake
        draw_snake(snake_list)

        pygame.display.update()

        # eat food
        if gx == food_x and gy == food_y:
            food_x, food_y = random_food_position()
            snake_len += 1

        clock.tick(SNAKE_SPEED)

    pygame.quit()
    quit()


if __name__ == "__main__":
    gameLoop()