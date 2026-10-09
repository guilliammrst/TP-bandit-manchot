import matplotlib.pyplot as plt

from ban10 import Ban10
from greedy_player import GreedyPlayer
from optimist_greedy_player import OptimistGreedyPlayer

NUM_PLAYERS = 2000
NUM_STEPS = 1000
EPSILON = 0.1

def main() -> None:
    bandits = [Ban10() for _ in range(NUM_PLAYERS)]
    epsilon_players = [GreedyPlayer(n=10, eps=EPSILON) for _ in range(NUM_PLAYERS)]
    optimistic_players = [
        OptimistGreedyPlayer(n=10, eps=0.0) for _ in range(NUM_PLAYERS)
    ]

    average_rewards = {"epsilon-greedy": [], "optimistic": []}
    optimal_choice_percentages = {"epsilon-greedy": [], "optimistic": []}

    for _ in range(NUM_STEPS):
        reward_totals = {"epsilon-greedy": 0.0, "optimistic": 0.0}
        optimal_choice_counts = {"epsilon-greedy": 0, "optimistic": 0}

        for bandit, epsilon_player, optimistic_player in zip(
            bandits, epsilon_players, optimistic_players
        ):
            for name, player in (
                ("epsilon-greedy", epsilon_player),
                ("optimistic", optimistic_player),
            ):
                action = player.get_action()
                reward = bandit.play(action)
                player.reward(action, reward)
                reward_totals[name] += reward
                optimal_choice_counts[name] += (
                    action == bandit.best_bandit_number
                )

        for name in average_rewards:
            average_rewards[name].append(reward_totals[name] / NUM_PLAYERS)
            optimal_choice_percentages[name].append(
                100 * optimal_choice_counts[name] / NUM_PLAYERS
            )

    steps = range(1, NUM_STEPS + 1)
    plt.plot(steps, average_rewards["epsilon-greedy"], label="GreedyPlayer (ε = 0,1)")
    plt.plot(
        steps,
        average_rewards["optimistic"],
        label="OptimistGreedyPlayer (Q initial = 5, ε = 0)",
    )
    plt.xlabel("Itération")
    plt.ylabel("Récompense moyenne")
    plt.title(f"Récompenses moyennes ({NUM_PLAYERS} bandits communs)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("optimistic_reward_comparison.png", dpi=150)

    plt.figure()
    plt.plot(
        steps,
        optimal_choice_percentages["epsilon-greedy"],
        label="GreedyPlayer (ε = 0,1)",
    )
    plt.plot(
        steps,
        optimal_choice_percentages["optimistic"],
        label="OptimistGreedyPlayer (Q initial = 5, ε = 0)",
    )
    plt.xlabel("Itération")
    plt.ylabel("Joueurs ayant choisi le bras optimal (%)")
    plt.title(f"Choix optimaux ({NUM_PLAYERS} bandits communs)")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("optimistic_optimality_comparison.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
