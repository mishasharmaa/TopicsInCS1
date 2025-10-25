from __future__ import annotations
import random
from typing import Optional, Tuple, List

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
            self,
            render_mode: Optional[str] = None,
            seed: Optional[int] = None,
            reward_mode: str = "survival",
            max_steps: int = 5000,
            frame_size_x: int = 720,
            frame_size_y: int = 480,
    ):
        super().__init__()
        self.render_mode = render_mode
        self._rnd = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self.reward_mode = reward_mode
        self.max_steps = max_steps

        self.frame_size_x = frame_size_x
        self.frame_size_y = frame_size_y
        self.grid_size = 10  # Snake moves in 10-pixel increments
        self.grid_width = frame_size_x // self.grid_size
        self.grid_height = frame_size_y // self.grid_size

        # Observation space:
        # [head_x, head_y, food_x, food_y, food_dist_x, food_dist_y,
        #  danger_up, danger_down, danger_left, danger_right,
        #  snake_length, direction_up, direction_down, direction_left, direction_right]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(15,),
            dtype=np.float32
        )

        # Action space: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.action_space = spaces.Discrete(4)

        self.reset(seed=seed)

        # Pygame for rendering
        self._pygame = None
        self._screen = None
        self._clock = None

        # Colors
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rnd.seed(seed)
            self._np_rng = np.random.default_rng(seed)

        # Center
        start_x = 100
        start_y = 50
        self.snake_pos = [start_x, start_y]
        self.snake_body = [
            [start_x, start_y],
            [start_x - 10, start_y],
            [start_x - 20, start_y]
        ]

        # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.direction = 3  # Start moving RIGHT

        self.food_pos = self._spawn_food()

        self.steps = 0
        self.score = 0
        self.steps_since_food = 0

        self.prev_distance_to_food = self._distance_to_food()

        obs = self._get_obs()
        info = {"score": self.score, "length": len(self.snake_body)}
        return obs, info

    def step(self, action: int):
        self.steps += 1
        self.steps_since_food += 1

        new_direction = int(action)

        if new_direction == 0 and self.direction != 1:  # UP (not if going DOWN)
            self.direction = 0
        elif new_direction == 1 and self.direction != 0:  # DOWN (not if going UP)
            self.direction = 1
        elif new_direction == 2 and self.direction != 3:  # LEFT (not if going RIGHT)
            self.direction = 2
        elif new_direction == 3 and self.direction != 2:  # RIGHT (not if going LEFT)
            self.direction = 3

        if self.direction == 0:  # UP
            self.snake_pos[1] -= self.grid_size
        elif self.direction == 1:  # DOWN
            self.snake_pos[1] += self.grid_size
        elif self.direction == 2:  # LEFT
            self.snake_pos[0] -= self.grid_size
        elif self.direction == 3:  # RIGHT
            self.snake_pos[0] += self.grid_size

        self.snake_body.insert(0, list(self.snake_pos))

        ate_food = False
        if self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]:
            self.score += 1
            ate_food = True
            self.steps_since_food = 0
            self.food_pos = self._spawn_food()
        else:
            self.snake_body.pop()

        terminated = self._check_collision()

        truncated = False
        if self.steps >= self.max_steps or self.steps_since_food > 100 * len(self.snake_body):
            truncated = True

        reward = self._calculate_reward(ate_food, terminated)

        obs = self._get_obs()
        info = {
            "score": self.score,
            "length": len(self.snake_body),
            "steps_since_food": self.steps_since_food
        }

        if self.render_mode == "human":
            self._render_human()

        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self.render_mode == "human":
            self._render_human()
        else:
            return None

    def close(self):
        if self._pygame:
            pygame.quit()
            self._pygame = None
            self._screen = None
            self._clock = None

    def _spawn_food(self) -> List[int]:
        while True:
            x = self._rnd.randint(1, self.grid_width - 1) * self.grid_size
            y = self._rnd.randint(1, self.grid_height - 1) * self.grid_size
            if [x, y] not in self.snake_body:
                return [x, y]

    def _distance_to_food(self) -> float:
        return abs(self.snake_pos[0] - self.food_pos[0]) + abs(self.snake_pos[1] - self.food_pos[1])

    def _check_collision(self) -> bool:
        if (self.snake_pos[0] < 0 or self.snake_pos[0] >= self.frame_size_x or
                self.snake_pos[1] < 0 or self.snake_pos[1] >= self.frame_size_y):
            return True

        for block in self.snake_body[1:]:
            if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
                return True

        return False

    def _is_danger(self, direction: int) -> bool:
        test_pos = list(self.snake_pos)

        if direction == 0:  # UP
            test_pos[1] -= self.grid_size
        elif direction == 1:  # DOWN
            test_pos[1] += self.grid_size
        elif direction == 2:  # LEFT
            test_pos[0] -= self.grid_size
        elif direction == 3:  # RIGHT
            test_pos[0] += self.grid_size

        if (test_pos[0] < 0 or test_pos[0] >= self.frame_size_x or
                test_pos[1] < 0 or test_pos[1] >= self.frame_size_y):
            return True

        if test_pos in self.snake_body:
            return True

        return False

    def _calculate_reward(self, ate_food: bool, terminated: bool) -> float:
        reward = 0.0

        if terminated:
            reward -= 10.0
            return reward

        if ate_food:
            if self.reward_mode == "survival":
                reward += 10.0
            elif self.reward_mode == "length":
                reward += 10.0 + len(self.snake_body)  # Bonus for length
        else:
            current_distance = self._distance_to_food()

            if current_distance < self.prev_distance_to_food:
                reward += 0.1
            else:
                reward -= 0.15

            self.prev_distance_to_food = current_distance

            reward += 0.01

        return reward

    def _get_obs(self) -> np.ndarray:
        head_x = self.snake_pos[0] / self.frame_size_x
        head_y = self.snake_pos[1] / self.frame_size_y
        food_x = self.food_pos[0] / self.frame_size_x
        food_y = self.food_pos[1] / self.frame_size_y

        food_dist_x = (self.food_pos[0] - self.snake_pos[0]) / self.frame_size_x
        food_dist_y = (self.food_pos[1] - self.snake_pos[1]) / self.frame_size_y

        danger_up = float(self._is_danger(0))
        danger_down = float(self._is_danger(1))
        danger_left = float(self._is_danger(2))
        danger_right = float(self._is_danger(3))

        max_length = self.grid_width * self.grid_height
        snake_length = len(self.snake_body) / max_length

        dir_up = float(self.direction == 0)
        dir_down = float(self.direction == 1)
        dir_left = float(self.direction == 2)
        dir_right = float(self.direction == 3)

        obs = np.array([
            head_x, head_y, food_x, food_y,
            food_dist_x + 0.5, food_dist_y + 0.5,
            danger_up, danger_down, danger_left, danger_right,
            snake_length,
            dir_up, dir_down, dir_left, dir_right
        ], dtype=np.float32)

        return obs

    def _lazy_pygame(self):
        if self._pygame is None:
            import pygame
            self._pygame = pygame
            pygame.init()
            self._screen = pygame.display.set_mode((self.frame_size_x, self.frame_size_y))
            self._clock = pygame.time.Clock()
            pygame.display.set_caption("Snake RL")

    def _render_human(self):
        self._lazy_pygame()
        pygame = self._pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        self._draw_scene()
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def _draw_scene(self):
        pygame = self._pygame

        self._screen.fill(self.black)

        for pos in self.snake_body:
            pygame.draw.rect(
                self._screen,
                self.green,
                pygame.Rect(pos[0], pos[1], self.grid_size, self.grid_size)
            )

        # Draw food
        pygame.draw.rect(
            self._screen,
            self.white,
            pygame.Rect(self.food_pos[0], self.food_pos[1], self.grid_size, self.grid_size)
        )

        font = pygame.font.SysFont('consolas', 20)
        score_surface = font.render(f'Score: {self.score}', True, self.white)
        score_rect = score_surface.get_rect()
        score_rect.midtop = (self.frame_size_x / 10, 15)
        self._screen.blit(score_surface, score_rect)