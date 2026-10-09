import random

class Bandit:
    def __init__(self):
        self.avg = random.gauss(0.0, 1.0)

    def play(self):
        return random.gauss(self.avg, 1.0)