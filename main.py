import matplotlib.pyplot as plt

from ban10 import Ban10
from greedy_player import GreedyPlayer

NUM_PLAYERS = 2000
NUM_STEPS = 1000
EPSILON_VALUES = (0.0, 0.01, 0.1)

def simulate(epsilon: float) -> tuple[list[float], list[float]]:
    bandits = [Ban10() for _ in range(NUM_PLAYERS)]
    players = [GreedyPlayer(n=10, eps=epsilon) for _ in range(NUM_PLAYERS)]
    average_rewards = []
    optimal_choice_percentages = []

    for step in range(NUM_STEPS):
        total_reward = 0.0
        optimal_choice_count = 0

        for player_index, (player, bandit) in enumerate(zip(players, bandits)):
            action = player.get_action()
            reward = bandit.play(action)
            player.reward(action, reward)
            optimal = action == bandit.best_bandit_number

            total_reward += reward
            optimal_choice_count += optimal

            if player_index == 0:
                print(
                    f"epsilon={epsilon:g}, itération {step + 1} — bras choisi : {action}, "
                    f"récompense obtenue : {reward:.3f}, "
                    f"choix optimal : {'oui' if optimal else 'non'}"
                )

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
    plt.title(f"Récompense moyenne ({NUM_PLAYERS} joueurs par valeur de ε)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reward_evolution.png", dpi=150)

    plt.figure()
    for epsilon, (_, optimal_choice_percentages) in results.items():
        plt.plot(steps, optimal_choice_percentages, label=f"ε = {epsilon:g}")
    plt.xlabel("Itération")
    plt.ylabel("Joueurs ayant choisi le bras optimal (%)")
    plt.title(f"Choix optimaux ({NUM_PLAYERS} joueurs par valeur de ε)")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("optimality_evolution.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
