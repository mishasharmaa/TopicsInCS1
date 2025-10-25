from __future__ import annotations
import math
import random
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class AimTrainerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
            self,
            render_mode: Optional[str] = None,
            seed: Optional[int] = None,
            reward_mode: str= "survival",
            max_steps: int = 5000,
            width: int = 1280,
            height: int = 720
    ):
        super().__init__()
        self.mouse_x = None
        self.mouse_y = None
        self.render_mode = render_mode
        self._rnd = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.width = width
        self.height = height
        self.steps = 0
        self.score = 0
        self.clicks = 0
        self.hits = 0
        self.misses = 0
        self.total_distance_to_target = 0.0
        self.targets_spawned = 0
        self.reward_breakdown = {"hit_bonus": 0, "miss_penalty": 0, "death": 0, "proximity_bonus": 0,
                                 "survival_bonus": 0, "death_penalty": 0}
        self.best_distance = None
        self.reward_mode = reward_mode

        # Game constants
        self.growth_speed = 0.10
        self.max_ball_size = 150
        self.min_ball_size = 5
        self.max_initial_ball_size = 30
        self.target_margin = 100  # Keep targets away from edges

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        self.reset(seed=seed)

        self._pygame = None
        self._screen = None
        self._clock = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rnd.seed(seed)
            self._np_rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.clicks = 0
        self.hits = 0
        self.misses = 0

        self.mouse_x = self.width / 2
        self.mouse_y = self.height / 2

        self._spawn_new_target()

        self.total_distance_to_target = 0.0
        self.targets_spawned = 0

        obs = self._get_obs()
        info = {
            "score": self.score,
            "accuracy": 0.0,
            "hits": self.hits,
            "misses": self.misses
        }
        return obs, info

    def step(self, action: np.ndarray):

        click_x = float(action[0] * self.width)
        click_y = float(action[1] * self.height)

        self.mouse_x = click_x
        self.mouse_y = click_y
        self.clicks += 1
        self.steps += 1

        distance = math.sqrt((click_x - self.target_x) ** 2 + (click_y - self.target_y) ** 2)
        hit = distance <= self.ball_size

        reward = 0.0
        terminated = False

        if hit:
            self.hits += 1
            self.score += 1

            base_hit_reward = 2.0

            accuracy_bonus = max(0.0, 1.0 - (distance / self.ball_size))
            accuracy_reward = accuracy_bonus * 1.0

            size_difficulty = (self.max_ball_size - self.ball_size) / self.max_ball_size
            size_reward = size_difficulty * 1.5

            total_hit_reward = base_hit_reward + accuracy_reward + size_reward
            self.reward_breakdown["hit_bonus"] += total_hit_reward
            reward += total_hit_reward

            self._spawn_new_target()

        else:
            self.misses += 1

            max_distance = math.sqrt(self.width ** 2 + self.height ** 2)
            distance_penalty = (distance / max_distance) * 0.5
            base_miss_penalty = -0.2
            total_miss_penalty = base_miss_penalty - distance_penalty

            self.reward_breakdown["miss_penalty"] += total_miss_penalty
            reward += total_miss_penalty

            self.ball_size += self.growth_speed

            if self.ball_size >= self.max_ball_size:
                terminated = True
                death_penalty = -3.0
                self.reward_breakdown["death_penalty"] += death_penalty
                reward += death_penalty

        survival_bonus = 0.01
        if self.reward_mode == "survival":
            survival_bonus *= 5

        self.reward_breakdown["survival_bonus"] += survival_bonus
        reward += survival_bonus

        current_distance = math.sqrt((self.mouse_x - self.target_x) ** 2 + (self.mouse_y - self.target_y) ** 2)
        max_distance = math.sqrt(self.width ** 2 + self.height ** 2)

        proximity_bonus = 0.05 * (1.0 - current_distance / max_distance)
        if self.reward_mode == "survival":
            proximity_bonus *= 2.5
        self.reward_breakdown["proximity_bonus"] += proximity_bonus
        reward += proximity_bonus

        reward = np.clip(reward, -5.0, 5.0)

        truncated = self.steps >= self.max_steps

        obs = self._get_obs()
        accuracy = (self.hits / self.clicks) if self.clicks > 0 else 0.0
        info = {
            "score": self.score,
            "accuracy": accuracy,
            "hits": self.hits,
            "misses": self.misses,
            "distance_to_target": current_distance,
            "reward_breakdown": dict(self.reward_breakdown)
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

    def _spawn_new_target(self):
        margin = self.target_margin
        self.target_x = self._rnd.randint(margin, self.width - margin)
        self.target_y = self._rnd.randint(margin, self.height - margin)
        self.ball_size = self._rnd.randint(self.min_ball_size, self.max_initial_ball_size)
        self.targets_spawned += 1

    def _get_obs(self) -> np.ndarray:
        # Have to normalize to width of screen
        obs = np.array([
            self.mouse_x / self.width,
            self.mouse_y / self.height,
            self.target_x / self.width,
            self.target_y / self.height,
            self.ball_size / self.max_ball_size,
            self.growth_speed / 1.0
        ], dtype=np.float32)
        return obs

    def _lazy_pygame(self):
        if self._pygame is None:
            import pygame
            self._pygame = pygame
            pygame.init()
            self._screen = pygame.display.set_mode((self.width, self.height))
            self._clock = pygame.time.Clock()
            pygame.display.set_caption("Aim Trainer RL")

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

        self._screen.fill((50, 50, 50))

        pygame.draw.circle(
            self._screen,
            (255, 35, 12),
            (int(self.target_x), int(self.target_y)),
            int(self.ball_size)
        )

        pygame.draw.circle(
            self._screen,
            (0, 255, 0),
            (int(self.mouse_x), int(self.mouse_y)),
            3
        )

        pygame.draw.line(
            self._screen,
            (0, 255, 0),
            (int(self.mouse_x) - 10, int(self.mouse_y)),
            (int(self.mouse_x) + 10, int(self.mouse_y)),
            2
        )
        pygame.draw.line(
            self._screen,
            (0, 255, 0),
            (int(self.mouse_x), int(self.mouse_y) - 10),
            (int(self.mouse_x), int(self.mouse_y) + 10),
            2
        )

        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        location_text = font.render(f"X: {round(self.mouse_x, 6)} Y {round(self.mouse_y, 6)}", True, (255, 255, 255))

        self._screen.blit(score_text, (10, 10))
        self._screen.blit(location_text, (10, 50))

        size_text = pygame.font.Font(None, 24).render(
            f"Ball Size: {int(self.ball_size)}/{int(self.max_ball_size)}",
            True,
            (255, 255, 255)
        )
        self._screen.blit(size_text, (10, 90))