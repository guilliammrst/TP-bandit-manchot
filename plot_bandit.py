import math
import random
import statistics

import matplotlib.pyplot as plt

from bandit import Bandit


def main():
    random.seed(42)
    bandit = Bandit()
    rewards = [bandit.play() for _ in range(1000)]

    plt.hist(rewards, bins=30, density=True, alpha=0.7, edgecolor="black", label="Tirages")

    x_min, x_max = plt.xlim()
    x_values = [x_min + (x_max - x_min) * index / 200 for index in range(201)]
    density = [
        math.exp(-0.5 * (value - bandit.avg) ** 2) / math.sqrt(2 * math.pi)
        for value in x_values
    ]
    plt.plot(x_values, density, color="red", linewidth=2, label="Normale théorique")
    plt.xlabel("Récompense")
    plt.ylabel("Densité")
    plt.title("Distribution de 1000 récompenses du bandit")
    plt.legend()
    plt.tight_layout()
    plt.savefig("bandit_histogram.png", dpi=150)
    plt.show()

    print(f"Moyenne théorique : {bandit.avg:.3f}")
    print(f"Moyenne observée : {statistics.mean(rewards):.3f}")
    print(f"Écart-type observé : {statistics.stdev(rewards):.3f}")


if __name__ == "__main__":
    main()