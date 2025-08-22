
# tessera/pp/distributions.py
import random
import math

class Normal:
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def sample(self):
        return random.gauss(self.loc, self.scale)

    def log_prob(self, x):
        return -0.5 * (((x - self.loc)/self.scale)**2 + math.log(2*math.pi*self.scale**2))
