import math
import random

from greedy_player import GreedyPlayer


class UCBGreedyPlayer(GreedyPlayer):
    def __init__(self, n: int, eps: float, c: float) -> None:
        super().__init__(n, eps)
        self.c = c

    def _greedy_action(self) -> int:
        unevaluated_actions = [
            action
            for action, count in enumerate(self.eval_count)
            if count == 0
        ]
        if unevaluated_actions:
            return random.choice(unevaluated_actions)

        total_evaluations = sum(self.eval_count)
        ucb_values = [
            value + self.c * math.sqrt(math.log(total_evaluations) / count)
            for value, count in zip(self.action_values, self.eval_count)
        ]
        best_value = max(ucb_values)
        best_actions = [
            action
            for action, value in enumerate(ucb_values)
            if value == best_value
        ]
        return random.choice(best_actions)
