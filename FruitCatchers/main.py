# Human playable fruit catchers game


import pygame
import random
import time
pygame.init()

# Screen setup 
screen_w = 800
screen_h = 600
screen = pygame.display.set_mode((screen_w, screen_h))
pygame.display.set_caption("Fruit Catchers with Bombs & Power-Up")

# Environent Colours
SKY_BLUE = (135, 206, 235)
GRASS_GREEN = (34, 139, 34)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)

# Basket settings
basket_w = 100
basket_h = 50
basket_x = screen_w // 2 - basket_w // 2
basket_y = screen_h - basket_h - 40
basket_speed = 12  

# Game state
score = 0
timer = pygame.time.Clock()

# Power-up settings
powerup_active = False
powerup_duration = 3
powerup_cooldown = 8
last_powerup_time = -powerup_cooldown

# Gravity and difficulty 
base_gravity = 0.10
gravity = base_gravity
speed_multiplier = 1.0

# Bomb settings 
bomb_spawn_chance = 0.002
bombs = []


# Fruit class 
class Fruit:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = random.randint(0, screen_w - 50)
        self.y = random.randint(-300, -50)
        self.base_speed = random.uniform(2, 3.5)
        self.speed = self.base_speed * speed_multiplier
        self.color = random.choice([RED, ORANGE, YELLOW])

    def update(self, slow_factor=1.0):
        # dynamically scale speed by current power-up slowdown
        self.y += self.speed * slow_factor

    def draw(self):
        pygame.draw.ellipse(screen, self.color, [self.x, self.y, 50, 50])


# Bomb class 
class Bomb:
    def __init__(self):
        self.x = random.randint(0, screen_w - 40)
        self.y = -40
        self.vy = random.uniform(1.5, 3) * speed_multiplier

    def update(self, slow_factor=1.0):
        global gravity
        self.vy += gravity / 3
        self.y += self.vy * slow_factor

    def draw(self):
        pygame.draw.circle(screen, BLACK, (int(self.x + 20), int(self.y + 20)), 20)


# Helper functions 
def draw_background():
    screen.fill(SKY_BLUE)
    pygame.draw.rect(screen, GRASS_GREEN, [0, screen_h - 40, screen_w, 40])

def draw_basket(x, y):
    pygame.draw.rect(screen, BLACK, [x, y, basket_w, basket_h])

def display_score(score):
    font = pygame.font.SysFont(None, 36)
    text = font.render(f"Score: {score}", True, BLACK)
    screen.blit(text, (10, 10))

def display_powerup_status(remaining_cooldown):
    font = pygame.font.SysFont(None, 28)
    if powerup_active:
        text = font.render("POWER-UP ACTIVE!", True, BLUE)
    elif remaining_cooldown > 0:
        text = font.render(f"Cooldown: {remaining_cooldown:.1f}s", True, BLACK)
    else:
        text = font.render("Press SPACE for Power-Up", True, BLACK)
    screen.blit(text, (10, 50))

def game_over():
    font = pygame.font.SysFont(None, 72)
    text = font.render("Game Over!", True, BLACK)
    screen.blit(text, (screen_w // 2 - 180, screen_h // 2 - 36))
    pygame.display.update()
    pygame.time.delay(2000)
    pygame.quit()
    quit()


# Main Game loop
def game_loop():
    global basket_x, basket_y, score, powerup_active, gravity, last_powerup_time, speed_multiplier

    fruits = [Fruit()]      
    next_fruit_spawned = False

    running = True
    while running:
        draw_background()
        current_time = time.time()

        # Randomly spawn bombs 
        if random.random() < bomb_spawn_chance and len(bombs) < 3:
            bombs.append(Bomb())

        # Handle quit event 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Basket controls (left, right, up, and down)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            basket_x -= basket_speed
        if keys[pygame.K_RIGHT]:
            basket_x += basket_speed
        if keys[pygame.K_UP]:
            basket_y -= 8
        if keys[pygame.K_DOWN]:
            basket_y += 8

        basket_x = max(0, min(screen_w - basket_w, basket_x))
        basket_y = max(0, min(screen_h - basket_h - 40, basket_y))

        # Power-Up 
        if keys[pygame.K_SPACE] and not powerup_active and (current_time - last_powerup_time >= powerup_cooldown):
            powerup_active = True
            last_powerup_time = current_time

        if powerup_active and (current_time - last_powerup_time >= powerup_duration):
            powerup_active = False

        slow_factor = 0.5 if powerup_active else 1.0

        # Difficulty ramp 
        if score >= 10 and speed_multiplier == 1.0:
            speed_multiplier = 1.3
            for f in fruits:
                f.speed = f.base_speed * speed_multiplier
            for b in bombs:
                b.vy *= 1.3
            print("⚡ Level up! Fruits and bombs are faster!")

        # Fruit logic 
        if len(fruits) == 1 and fruits[0].y >= screen_h / 2 and not next_fruit_spawned:
            fruits.append(Fruit())
            next_fruit_spawned = True

        for fruit in fruits[:]:
            fruit.update(slow_factor)
            fruit.draw()

            if basket_y < fruit.y + 50 and basket_y + basket_h > fruit.y and basket_x < fruit.x + 25 < basket_x + basket_w:
                score += 1
                fruit.reset()

            if fruit.y + 50 >= screen_h - 40:
                game_over()

        # Bomb logic 
        for bomb in bombs[:]:
            bomb.update(slow_factor)
            bomb.draw()
            if basket_x < bomb.x + 20 < basket_x + basket_w and basket_y < bomb.y + 20 < basket_y + basket_h:
                game_over()
            if bomb.y > screen_h:
                bombs.remove(bomb)

        # Reset stagger once both fruits are gone 
        if len(fruits) == 2:
            if all(f.y < -50 or f.y > screen_h for f in fruits):
                fruits = [Fruit()]
                next_fruit_spawned = False

        # UI 
        draw_basket(basket_x, basket_y)
        display_score(score)
        remaining_cooldown = max(0, powerup_cooldown - (current_time - last_powerup_time))
        display_powerup_status(remaining_cooldown)

        pygame.display.update()
        timer.tick(60)

if __name__ == "__main__":
    game_loop()
