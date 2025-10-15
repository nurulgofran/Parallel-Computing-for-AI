import numpy as np
import matplotlib.pyplot as plt
from task2_1_scalar_product_sequential import sequential_scalar_product
from task2_2_scalar_product_parallel import parallel_scalar_product


def create_comparison_visualization(n, p, seq_steps, par_steps):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    methods = ["Sequential", "Parallel"]
    steps = [seq_steps, par_steps]

    ax1.bar(methods, steps)
    ax1.set_title("Time Steps Comparison")
    ax1.set_ylabel("Time Steps")
    for i, v in enumerate(steps):
        ax1.text(i, v, str(v), ha="center", va="bottom")

    speedup = seq_steps / par_steps
    efficiency = speedup / p * 100

    ax2.bar(["Speedup", "Efficiency (%)"], [speedup, efficiency])
    ax2.set_title("Performance Metrics")
    ax2.set_ylabel("Value")
    for i, v in enumerate([speedup, efficiency]):
        ax2.text(i, v, f"{v:.2f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig("../../plots/task2_3_comparison.png", dpi=150, bbox_inches="tight")
    print("✓ Saved comparison visualization: plots/task2_3_comparison.png")
    plt.close()


def main():
    print("=" * 80)
    print("TASK 2.3: TIME STEPS ANALYSIS")
    print("=" * 80)
    print()

    n = 160
    p = 8

    np.random.seed(42)
    A = np.random.rand(n)
    B = np.random.rand(n)

    print(f"Problem: Scalar product of two vectors")
    print(f"Vector size: n = {n}")
    print(f"Processors: p = {p}")
    print()

    print("Running sequential algorithm...")
    seq_result, seq_steps, seq_ops = sequential_scalar_product(A, B)

    print("Running parallel algorithm...")
    par_result, par_steps, par_ops, _, _, _ = parallel_scalar_product(A, B, p)

    print()
    print("=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Metric':<25} {'Sequential':<15} {'Parallel':<15}")
    print("-" * 80)
    print(f"{'Result':<25} {seq_result:<15.6f} {par_result:<15.6f}")
    print(f"{'Time Steps':<25} {seq_steps:<15} {par_steps:<15}")
    print(f"{'Operations':<25} {seq_ops:<15} {par_ops:<15}")
    print(f"{'CPUs':<25} {'1':<15} {p:<15}")
    print()

    speedup = seq_steps / par_steps
    efficiency = speedup / p

    print("-" * 80)
    print("TASK 2.4: SPEEDUP AND EFFICIENCY")
    print("-" * 80)
    print(f"Speedup (Sp):      {speedup:.4f}")
    print(f"  Formula:         Sp = T_sequential / T_parallel")
    print(f"  Calculation:     Sp = {seq_steps} / {par_steps} = {speedup:.4f}")
    print()
    print(f"Efficiency (Ep):   {efficiency:.4f} = {efficiency * 100:.2f}%")
    print(f"  Formula:         Ep = Sp / p")
    print(f"  Calculation:     Ep = {speedup:.4f} / {p} = {efficiency:.4f}")
    print()

    print("Generating visualizations...")
    create_comparison_visualization(n, p, seq_steps, par_steps)
    print()

    print("=" * 80)
    print("Task 2.3 completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
