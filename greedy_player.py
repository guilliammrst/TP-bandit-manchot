import random

class GreedyPlayer:
    def __init__(self, n: int, eps: float) -> None:
        self.n = n
        self.eps = eps
        self.action_values = [0.0] * n
        self.eval_count = [0] * n

    def get_action(self) -> int:
        explore = random.random() < self.eps
        if explore:
            action = self._random_action()
        else:
            action = self._greedy_action()
        return action

    def _greedy_action(self) -> int:
        best_value = max(self.action_values)
        best_actions = [
            action
            for action, value in enumerate(self.action_values)
            if value == best_value
        ]
        return random.choice(best_actions)

    def _random_action(self) -> int:
        return random.randrange(self.n)

    def reward(self, action: int, reward_value: float) -> None:
        if not 0 <= action < self.n:
            raise ValueError(f"action doit être comprise entre 0 et {self.n - 1}.")

        self.eval_count[action] += 1
        count = self.eval_count[action]
        self.action_values[action] += (
            reward_value - self.action_values[action]
        ) / count
