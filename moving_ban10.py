from ban10 import Ban10
from moving_bandit import MovingBandit

class MovingBan10(Ban10):
    def __init__(self) -> None:
        self.bandits = [MovingBandit() for _ in range(10)]

    @property
    def best_bandit_number(self) -> int:
        return max(
            range(len(self.bandits)), key=lambda index: self.bandits[index].avg
        )
