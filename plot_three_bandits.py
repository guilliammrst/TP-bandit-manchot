import math
import random
import statistics

import matplotlib.pyplot as plt

from bandit import Bandit


def main():
    random.seed(2)
    bandits = [Bandit() for _ in range(3)]
    rewards_by_bandit = [[bandit.play() for _ in range(1000)] for bandit in bandits]
    expectations = [bandit.avg for bandit in bandits]
    assert len(set(expectations)) == 3, "Les trois espérances doivent être différentes."

    all_rewards = [reward for rewards in rewards_by_bandit for reward in rewards]
    x_min = min(all_rewards) - 0.5
    x_max = max(all_rewards) + 0.5
    x_values = [x_min + (x_max - x_min) * index / 300 for index in range(301)]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
    for index, (axis, bandit, rewards) in enumerate(zip(axes, bandits, rewards_by_bandit), start=1):
        axis.hist(
            rewards,
            bins=30,
            range=(x_min, x_max),
            density=True,
            alpha=0.7,
            edgecolor="black",
            label="Tirages",
        )
        density = [
            math.exp(-0.5 * (value - bandit.avg) ** 2) / math.sqrt(2 * math.pi)
            for value in x_values
        ]
        axis.plot(x_values, density, color="red", linewidth=2, label="Normale théorique")
        axis.axvline(bandit.avg, color="black", linestyle="--", label=f"Espérance = {bandit.avg:.2f}")
        axis.set_title(f"Bandit {index}")
        axis.set_xlabel("Récompense")
        axis.legend()

    axes[0].set_ylabel("Densité")
    figure.suptitle("Distributions de 1000 récompenses pour trois bandits")
    figure.tight_layout()
    figure.savefig("three_bandits_histogram.png", dpi=150)
    plt.show()

    for index, (bandit, rewards) in enumerate(zip(bandits, rewards_by_bandit), start=1):
        print(
            f"Bandit {index} : espérance = {bandit.avg:.3f}, "
            f"moyenne observée = {statistics.mean(rewards):.3f}, "
            f"écart-type observé = {statistics.stdev(rewards):.3f}"
        )


if __name__ == "__main__":
    main()