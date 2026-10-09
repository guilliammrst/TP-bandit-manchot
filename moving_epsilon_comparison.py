import matplotlib.pyplot as plt

from greedy_player import GreedyPlayer
from moving_ban10 import MovingBan10

NUM_PLAYERS = 2000
NUM_STEPS = 1000
EPSILON_VALUES = (0.0, 0.01, 0.1)


def simulate(epsilon: float) -> tuple[list[float], list[float]]:
    bandits = [MovingBan10() for _ in range(NUM_PLAYERS)]
    players = [GreedyPlayer(n=10, eps=epsilon) for _ in range(NUM_PLAYERS)]
    average_rewards = []
    optimal_choice_percentages = []

    for _ in range(NUM_STEPS):
        total_reward = 0.0
        optimal_choice_count = 0

        for player, bandit in zip(players, bandits):
            action = player.get_action()
            optimal_choice_count += action == bandit.best_bandit_number
            reward = bandit.play(action)
            player.reward(action, reward)
            total_reward += reward

        average_rewards.append(total_reward / NUM_PLAYERS)
        optimal_choice_percentages.append(
            100 * optimal_choice_count / NUM_PLAYERS
        )

    return average_rewards, optimal_choice_percentages


def main() -> None:
    results = {
        epsilon: simulate(epsilon)
        for epsilon in EPSILON_VALUES
    }
    steps = range(1, NUM_STEPS + 1)

    for epsilon, (average_rewards, _) in results.items():
        plt.plot(steps, average_rewards, label=f"ε = {epsilon:g}")
    plt.xlabel("Itération")
    plt.ylabel("Récompense moyenne")
    plt.title(f"Bandits non stationnaires — récompense moyenne ({NUM_PLAYERS} joueurs)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("moving_reward_evolution.png", dpi=150)

    plt.figure()
    for epsilon, (_, optimal_choice_percentages) in results.items():
        plt.plot(steps, optimal_choice_percentages, label=f"ε = {epsilon:g}")
    plt.xlabel("Itération")
    plt.ylabel("Joueurs ayant choisi le bras optimal (%)")
    plt.title(f"Bandits non stationnaires — optimalité ({NUM_PLAYERS} joueurs)")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("moving_optimality_evolution.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
