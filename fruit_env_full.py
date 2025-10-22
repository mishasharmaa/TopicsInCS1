# fruit_env_full.py // AI playing the game
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import time

# Import everything from main.py 
from main import (
    Fruit, Bomb,
    screen_w, screen_h,
    draw_background, draw_basket, display_score, display_powerup_status,
    basket_w, basket_h, base_gravity, basket_speed
)


class FruitCatchFullEnv(gym.Env):
    """
    Full RL environment that mirrors the playable Fruit Catchers game.
    Controls:
      0 = left, 1 = right, 2 = up, 3 = down, 4 = power-up
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=False, persona="survivor"):
        super().__init__()
        self.persona = persona
        pygame.init()
        self.render_mode = render_mode
        if self.render_mode:
            self.screen = pygame.display.set_mode((screen_w, screen_h))
            pygame.display.set_caption("AI Playing - Fruit Catchers")
        else:
            self.screen = pygame.Surface((screen_w, screen_h))

        # Define action & observation spaces 
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )

        # Clock and initial reset
        self.clock = pygame.time.Clock()
        self.reset()

    # Methods
    def _get_obs(self):
        f1x = np.clip(self.fruits[0].x / screen_w, 0.0, 1.0)
        f1y = np.clip(self.fruits[0].y / screen_h, 0.0, 1.0)
        f2x = np.clip(self.fruits[1].x / screen_w, 0.0, 1.0)
        f2y = np.clip(self.fruits[1].y / screen_h, 0.0, 1.0)
        bx = np.clip(self.basket_x / screen_w, 0.0, 1.0)
        by = np.clip(self.basket_y / screen_h, 0.0, 1.0)
        cooldown = np.clip((time.time() - self.last_powerup_time) / self.powerup_cooldown, 0.0, 1.0)
        active = 1.0 if self.powerup_active else 0.0
        return np.array([f1x, f1y, f2x, f2y, bx, by, cooldown, active], dtype=np.float32)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.basket_x = screen_w // 2 - basket_w // 2
        self.basket_y = screen_h - basket_h - 40
        self.score = 0
        self.done = False

        # game physics
        self.gravity = base_gravity
        self.speed_multiplier = 1.0
        self.powerup_active = False
        self.powerup_duration = 3
        self.powerup_cooldown = 8
        self.last_powerup_time = -self.powerup_cooldown

        # objects
        self.fruits = [Fruit(), Fruit()]
        for f in self.fruits:
            f.y = random.randint(-250, -100)  # start the fruits higher up
            f.speed = random.uniform(2.5, 3.5)  # slower descent of the fruits 
        self.bombs = []

        return self._get_obs(), {}

    def step(self, action):
        reward = 0.0
        info = {}
        current_time = time.time()
        
        # Basket movement 
        move_speed = 10  # tuned for AI training so it can train properly
        if action == 0:
            self.basket_x -= move_speed
        elif action == 1:
            self.basket_x += move_speed
        elif action == 2:
            self.basket_y -= 8
        elif action == 3:
            self.basket_y += 8
        elif action == 4:
            
        # Activate power-up if available
            if not self.powerup_active and (current_time - self.last_powerup_time >= self.powerup_cooldown):
                self.powerup_active = True
                self.last_powerup_time = current_time
                self.gravity = base_gravity / 3
                for f in self.fruits:
                    f.speed /= 2
                for b in self.bombs:
                    b.vy /= 2

        # Clamp basket to screen
        self.basket_x = np.clip(self.basket_x, 0, screen_w - basket_w)
        min_y = screen_h - 200  # about 200px from bottom
        self.basket_y = np.clip(self.basket_y, min_y, screen_h - basket_h - 40)

        # Power-up timeout 
        if self.powerup_active and (current_time - self.last_powerup_time >= self.powerup_duration):
            self.powerup_active = False
            self.gravity = base_gravity
            for f in self.fruits:
                f.speed *= 2
            for b in self.bombs:
                b.vy *= 2
            
        # Bomb logic 
        if random.random() < 0.002 and len(self.bombs) < 2:
            self.bombs.append(Bomb())

        for bomb in self.bombs[:]:
            bomb.update()

        # Bomb hits basket
            if (self.basket_x < bomb.x + 20 < self.basket_x + basket_w) and \
            (self.basket_y < bomb.y + 20 < self.basket_y + basket_h):
                reward -= 10.0
                self.done = True

        # Remove bomb off-screen
            elif bomb.y > screen_h:
                self.bombs.remove(bomb)

        # Fruit logic 
        caught = False
        for fruit in self.fruits:
            fruit.update()

        # Catching fruit
        if (self.basket_y < fruit.y + 50 < self.basket_y + basket_h) and \
            (self.basket_x < fruit.x + 25 < self.basket_x + basket_w):
            reward += 25.0
            self.score += 1
            caught = True
            fruit.reset()

        # Fruit hits the ground
        elif fruit.y + 50 >= screen_h - 40:
            reward -= 3.0
            fruit.reset()
            self.score = max(0, self.score - 1)

        # Difficulty scaling 
        if self.score >= 10 and self.speed_multiplier == 1.0:
            self.speed_multiplier = 1.3
            for f in self.fruits:
                f.speed *= 1.3
            for b in self.bombs:
                b.vy *= 1.3

        # Alignment reward 
        target = min(self.fruits, key=lambda f: f.y)
        basket_center = self.basket_x + basket_w / 2
        fruit_center = target.x + 25
        dist_x = abs(fruit_center - basket_center)

        # Encourage aligning under fruit
        reward += max(0, 1 - dist_x / (screen_w / 2)) * 0.5

        # Gentle penalty for moving too far unnecessarily
        if action in [0, 1, 2, 3]:
            reward -= 0.02

        # Keeping the basket below lower half of screen so it doesn't "cheat"
        if self.basket_y < screen_h - 250:
            reward -= 2.0
            
        # Encourage survival each frame
        reward += 0.1

        # Slight penalty if near a bomb
        for bomb in self.bombs:
            dist = abs((bomb.x + 20) - basket_center)
            if dist < 100:
                reward -= 0.5

        if not self.done:
            reward += 0.1  # survival reward only if still alive


        if self.persona == "survivor":
            # Encourages staying alive, penalizes dying early
            if not self.done:
                reward += 0.2
            else:
                reward -= 5.0
        elif self.persona == "collector":
            
            # Encourages catching more fruits quickly
            reward += 0.5 * self.score
            
        obs = self._get_obs()
        if self.render_mode:
            self.render()

        return obs, reward, self.done, False, info



    def render(self):
        if not self.render_mode:
            return

        draw_background()
        for f in self.fruits:
            f.draw()
        for b in self.bombs:
            b.draw()
        draw_basket(self.basket_x, self.basket_y)
        display_score(self.score)
        remaining = max(0, self.powerup_cooldown - (time.time() - self.last_powerup_time))
        display_powerup_status(remaining)
        

        if self.done:
            font = pygame.font.SysFont(None, 72)
            text = font.render("Game Over!", True, (0, 0, 0))
            self.screen.blit(text, (screen_w // 2 - 180, screen_h // 2 - 36))

        pygame.display.update()
        self.clock.tick(60)
        
    def close(self):
        pygame.quit()
