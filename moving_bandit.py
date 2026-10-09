import math
import random

from bandit import Bandit

class MovingBandit(Bandit):
    def __init__(self) -> None:
        super().__init__()
        self.min_avg = random.uniform(-2.0, 0.0)
        self.max_avg = random.uniform(0.0, 2.0)
        self.period = random.randint(100, 500)
        self.phase = random.uniform(0.0, 2.0 * math.pi)
        self.step = 0
        self.avg = self._mean_at_current_phase()

    def _mean_at_current_phase(self) -> float:
        midpoint = (self.min_avg + self.max_avg) / 2
        amplitude = (self.max_avg - self.min_avg) / 2
        return midpoint + amplitude * math.sin(self.phase)

    def play(self) -> float:
        self.step += 1
        self.phase = (self.phase + 2.0 * math.pi / self.period) % (2.0 * math.pi)
        self.avg = self._mean_at_current_phase()
        return random.gauss(self.avg, 1.0)
