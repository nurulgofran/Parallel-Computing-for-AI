import numpy as np
import matplotlib.pyplot as plt


def speedup_formula(n, p):
    T_seq = n
    T_par = (n // p) + (p - 1)
    Sp = T_seq / T_par
    return Sp


def efficiency_formula(n, p):
    Sp = speedup_formula(n, p)
    Ep = Sp / p
    return Ep


def create_generalization_plots():
    n_values = [100, 200, 400, 800, 1600]
    p_values = [2, 4, 8, 16, 32]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    for n in n_values:
        speedups = [speedup_formula(n, p) for p in p_values]
        ax1.plot(p_values, speedups, "o-", label=f"n={n}")
    ax1.set_xlabel("Number of Processors (p)")
    ax1.set_ylabel("Speedup (Sp)")
    ax1.set_title("Speedup vs Processors")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for n in n_values:
        efficiencies = [efficiency_formula(n, p) * 100 for p in p_values]
        ax2.plot(p_values, efficiencies, "s-", label=f"n={n}")
    ax2.set_xlabel("Number of Processors (p)")
    ax2.set_ylabel("Efficiency (%)")
    ax2.set_title("Efficiency vs Processors")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    p_fixed = 8
    n_range = np.arange(100, 2001, 100)
    speedups = [speedup_formula(n, p_fixed) for n in n_range]
    ax3.plot(n_range, speedups, "o-")
    ax3.set_xlabel("Vector Size (n)")
    ax3.set_ylabel("Speedup (Sp)")
    ax3.set_title(f"Speedup vs Vector Size (p={p_fixed})")
    ax3.grid(True, alpha=0.3)

    efficiencies = [efficiency_formula(n, p_fixed) * 100 for n in n_range]
    ax4.plot(n_range, efficiencies, "s-")
    ax4.set_xlabel("Vector Size (n)")
    ax4.set_ylabel("Efficiency (%)")
    ax4.set_title(f"Efficiency vs Vector Size (p={p_fixed})")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("../../plots/task2_5_generalization.png", dpi=150, bbox_inches="tight")
    print("✓ Saved generalization plots: plots/task2_5_generalization.png")
    plt.close()


def main():
    print("=" * 80)
    print("TASK 2.5: GENERALIZATION OF SPEEDUP AND EFFICIENCY")
    print("=" * 80)
    print()

    print("FORMULAS")
    print("-" * 80)
    print("Sequential Time:     T_seq(n) = n")
    print("Parallel Time:       T_par(n,p) = n/p + (p-1)")
    print("  Phase 1 (parallel):   n/p steps")
    print("  Phase 2 (reduction):  p-1 steps")
    print()
    print("Speedup:             Sp(n,p) = T_seq / T_par = n / (n/p + p - 1)")
    print("Efficiency:          Ep(n,p) = Sp / p = n / (p * (n/p + p - 1))")
    print()

    print("EXAMPLE CALCULATIONS")
    print("-" * 80)
    examples = [(160, 8), (160, 4), (320, 8), (1000, 16)]

    for n, p in examples:
        Sp = speedup_formula(n, p)
        Ep = efficiency_formula(n, p)
        T_seq = n
        T_par = (n // p) + (p - 1)
        print(
            f"n={n:4d}, p={p:2d}:  Sp={Sp:6.3f},  Ep={Ep:6.3f} ({Ep * 100:5.1f}%),  T_seq={T_seq:4d},  T_par={T_par:4d}"
        )

    print()
    print("KEY OBSERVATIONS")
    print("-" * 80)
    print("1. Speedup is limited by the reduction phase (p-1 steps)")
    print("2. For fixed n, efficiency decreases as p increases")
    print("3. For fixed p, efficiency increases as n increases")
    print("4. Optimal p depends on n: larger n allows more processors efficiently")
    print()

    print("Generating generalization plots...")
    create_generalization_plots()
    print()

    print("=" * 80)
    print("Task 2.5 completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
