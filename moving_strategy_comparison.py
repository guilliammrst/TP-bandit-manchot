from copy import deepcopy

import matplotlib.pyplot as plt

from greedy_player import GreedyPlayer
from moving_ban10 import MovingBan10
from optimist_greedy_player import OptimistGreedyPlayer
from ucb_greedy_player import UCBGreedyPlayer

NUM_PLAYERS = 2000
NUM_STEPS = 1000
EPSILON = 0.1
UCB_C = 2.0


def main() -> None:
    strategy_names = ("epsilon-greedy", "optimistic", "ucb")
    initial_bandits = [MovingBan10() for _ in range(NUM_PLAYERS)]
    bandits = {
        "epsilon-greedy": initial_bandits,
        "optimistic": [deepcopy(bandit) for bandit in initial_bandits],
        "ucb": [deepcopy(bandit) for bandit in initial_bandits],
    }
    players = {
        "epsilon-greedy": [
            GreedyPlayer(n=10, eps=EPSILON) for _ in range(NUM_PLAYERS)
        ],
        "optimistic": [
            OptimistGreedyPlayer(n=10, eps=0.0) for _ in range(NUM_PLAYERS)
        ],
        "ucb": [
            UCBGreedyPlayer(n=10, eps=0.0, c=UCB_C)
            for _ in range(NUM_PLAYERS)
        ],
    }
    average_rewards = {name: [] for name in strategy_names}
    optimal_choice_percentages = {name: [] for name in strategy_names}

    for _ in range(NUM_STEPS):
        reward_totals = {name: 0.0 for name in strategy_names}
        optimal_choice_counts = {name: 0 for name in strategy_names}

        for name in strategy_names:
            for player, bandit in zip(players[name], bandits[name]):
                action = player.get_action()
                optimal_choice_counts[name] += (
                    action == bandit.best_bandit_number
                )
                reward = bandit.play(action)
                player.reward(action, reward)
                reward_totals[name] += reward

        for name in strategy_names:
            average_rewards[name].append(
                reward_totals[name] / NUM_PLAYERS
            )
            optimal_choice_percentages[name].append(
                100 * optimal_choice_counts[name] / NUM_PLAYERS
            )

    steps = range(1, NUM_STEPS + 1)
    labels = {
        "epsilon-greedy": "GreedyPlayer (ε = 0,1)",
        "optimistic": "OptimistGreedyPlayer (Q initial = 5)",
        "ucb": f"UCBGreedyPlayer (c = {UCB_C:g})",
    }

    for name in strategy_names:
        plt.plot(steps, average_rewards[name], label=labels[name])
    plt.xlabel("Itération")
    plt.ylabel("Récompense moyenne")
    plt.title("Bandits mobiles — récompense moyenne")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("moving_strategy_reward_comparison.png", dpi=150)

    plt.figure()
    for name in strategy_names:
        plt.plot(steps, optimal_choice_percentages[name], label=labels[name])
    plt.xlabel("Itération")
    plt.ylabel("Joueurs ayant choisi le bras optimal (%)")
    plt.title("Bandits mobiles — choix du bras optimal")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("moving_strategy_optimality_comparison.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
