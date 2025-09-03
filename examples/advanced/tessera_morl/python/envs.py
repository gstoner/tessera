import numpy as np

class DeepSeaTreasure:
    """Classic MORL env: 10x10 grid, two objectives: (treasure value, -time).
    State: (x,y), start (0,0), goal treasures at bottom row (y=9) with values.
    Actions: 0:up 1:right 2:down 3:left. Episode length 50.
    """
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.W, self.H = 10, 10
        self.max_steps = 50
        self.reset()

    def reset(self):
        self.x, self.y = 0, 0
        self.t = 0
        self.done = False
        return self._obs()

    def _obs(self):
        return np.array([self.x, self.y, self.t], dtype=np.float32)

    def step(self, a):
        if self.done:
            raise RuntimeError("step after done")
        dx = [0,1,0,-1][a]
        dy = [-1,0,1,0][a]
        self.x = int(np.clip(self.x + dx, 0, self.W-1))
        self.y = int(np.clip(self.y + dy, 0, self.H-1))
        self.t += 1
        r = np.zeros(2, dtype=np.float32)
        # time penalty
        r[1] = -1.0
        # treasure on bottom row
        if self.y == self.H-1:
            # treasure value increases with x
            r[0] = float((self.x+1) * 10)
            self.done = True
        if self.t >= self.max_steps:
            self.done = True
        return self._obs(), r, self.done, {}

class ResourceGathering:
    """Two resources on two sides; objectives: (+gathered_A, +gathered_B).
    Episode length 50. 11x11 grid with two resource tiles replenishing slowly.
    """
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.W = self.H = 11
        self.max_steps = 50
        self.reset()

    def reset(self):
        self.x, self.y = 5, 5
        self.t = 0
        self.done = False
        self.A = 0
        self.B = 0
        return self._obs()

    def _obs(self):
        return np.array([self.x, self.y, self.A, self.B, self.t], dtype=np.float32)

    def step(self, a):
        if self.done:
            raise RuntimeError("step after done")
        dx = [0,1,0,-1][a]
        dy = [-1,0,1,0][a]
        self.x = int(np.clip(self.x + dx, 0, self.W-1))
        self.y = int(np.clip(self.y + dy, 0, self.H-1))
        self.t += 1
        r = np.zeros(2, dtype=np.float32)
        # resource A at (2,2), B at (8,8)
        if self.x == 2 and self.y == 2:
            self.A += 1; r[0] += 1.0
        if self.x == 8 and self.y == 8:
            self.B += 1; r[1] += 1.0
        if self.t >= self.max_steps:
            self.done = True
        return self._obs(), r, self.done, {}
