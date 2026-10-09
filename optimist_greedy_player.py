from greedy_player import GreedyPlayer

class OptimistGreedyPlayer(GreedyPlayer):
    def __init__(self, n: int, eps: float) -> None:
        super().__init__(n, eps)
        self.action_values = [5.0] * n
