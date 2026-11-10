import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image, ImageDraw


class VLA2DEnv(gym.Env):
    """Lightweight image-and-language-conditioned tabletop manipulation simulator.

    The agent moves either the red or blue block one grid cell per action.
    Success means the selected block is horizontally or vertically adjacent
    to the other block.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 8}

    ACTIONS = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1),   # right
    }

    def __init__(self, grid_size=6, max_steps=30, render_mode=None):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.cell_px = 14
        self.image_size = grid_size * self.cell_px

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255,
                shape=(self.image_size, self.image_size, 3),
                dtype=np.uint8
            ),
            # Compact semantic language encoding:
            # [1, 0] = move red beside blue; [0, 1] = move blue beside red.
            "instruction": spaces.Box(
                low=0.0, high=1.0, shape=(2,), dtype=np.float32
            ),
            # Normalized positions: red_x, red_y, blue_x, blue_y.
            "state": spaces.Box(
                low=0.0, high=1.0, shape=(4,), dtype=np.float32
            ),
        })

        self.red = np.array([0, 0], dtype=np.int32)
        self.blue = np.array([0, 0], dtype=np.int32)
        self.move_red = True
        self.steps = 0
        self._last_frame = None

    def _distance(self):
        moving = self.red if self.move_red else self.blue
        fixed = self.blue if self.move_red else self.red
        return int(np.abs(moving - fixed).sum())

    def _get_obs(self):
        image = self.render()
        scale = float(self.grid_size - 1)
        state = np.array([
            self.red[0] / scale, self.red[1] / scale,
            self.blue[0] / scale, self.blue[1] / scale
        ], dtype=np.float32)
        instruction = np.array(
            [1.0, 0.0] if self.move_red else [0.0, 1.0],
            dtype=np.float32
        )
        return {"image": image, "instruction": instruction, "state": state}

    def _get_info(self, success=False):
        return {
            "success": bool(success),
            "distance": self._distance(),
            "instruction_text": (
                "Move the red block next to the blue block"
                if self.move_red else
                "Move the blue block next to the red block"
            ),
            "steps": self.steps,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        cells = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
        ]
        first, second = self.np_random.choice(len(cells), size=2, replace=False)
        self.red = np.array(cells[int(first)], dtype=np.int32)
        self.blue = np.array(cells[int(second)], dtype=np.int32)
        self.move_red = bool(self.np_random.integers(0, 2))
        while self._distance() <= 1:
            first, second = self.np_random.choice(len(cells), size=2, replace=False)
            self.red = np.array(cells[int(first)], dtype=np.int32)
            self.blue = np.array(cells[int(second)], dtype=np.int32)
        self.steps = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        before = self._distance()
        delta = np.array(self.ACTIONS[int(action)], dtype=np.int32)
        moving = self.red if self.move_red else self.blue
        fixed = self.blue if self.move_red else self.red

        candidate = np.clip(moving + delta, 0, self.grid_size - 1)
        if not np.array_equal(candidate, fixed):
            moving[:] = candidate

        self.steps += 1
        after = self._distance()
        success = after == 1
        reward = 0.20 * (before - after) - 0.02
        if success:
            reward += 10.0

        terminated = success
        truncated = self.steps >= self.max_steps and not terminated
        return self._get_obs(), float(reward), terminated, truncated, self._get_info(success)

    def render(self):
        size = self.image_size
        image = Image.new("RGB", (size, size), (245, 245, 245))
        draw = ImageDraw.Draw(image)

        for i in range(self.grid_size + 1):
            p = i * self.cell_px
            draw.line((p, 0, p, size), fill=(205, 205, 205), width=1)
            draw.line((0, p, size, p), fill=(205, 205, 205), width=1)

        def draw_block(position, color):
            x, y = int(position[0]), int(position[1])
            pad = 2
            x0, y0 = y * self.cell_px + pad, x * self.cell_px + pad
            x1, y1 = (y + 1) * self.cell_px - pad, (x + 1) * self.cell_px - pad
            draw.rounded_rectangle((x0, y0, x1, y1), radius=3, fill=color)

        draw_block(self.red, (220, 45, 45))
        draw_block(self.blue, (40, 95, 220))
        return np.asarray(image, dtype=np.uint8)

    def close(self):
        pass
