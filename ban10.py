from bandit import Bandit

class Ban10:
    def __init__(self):
        self.bandits = [Bandit() for _ in range(10)]
        self.best_bandit_number = max(
            range(len(self.bandits)), key=lambda index: self.bandits[index].avg
        )

    def play(self, arm_number):
        if not 0 <= arm_number < len(self.bandits):
            raise ValueError("arm_number doit être compris entre 0 et 9.")
        return self.bandits[arm_number].play()