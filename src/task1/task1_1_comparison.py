import sys
import os

from task1_1_prefix_sum_sequential import sequential_prefix_sum
from task1_1_prefix_sum_parallel import (
    parallel_prefix_sum_simulation,
    calculate_parallel_metrics,
)

import matplotlib.pyplot as plt
import numpy as np
import math


def create_comparison_visualization(x):
    seq_result, seq_ops = sequential_prefix_sum(x)
    par_result, par_steps, par_ops, par_details = parallel_prefix_sum_simulation(x)

    par_result_inclusive = par_result[1:] + [sum(x)]

    n = len(x)
    seq_steps = n - 1
    par_metrics = calculate_parallel_metrics(n)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    ax1.bar(range(len(x)), x)
    ax1.set_title("Input")
    ax1.set_xlabel("Index")
    for i, v in enumerate(x):
        ax1.text(i, v + 0.3, str(v), ha="center")

    ax2.bar(range(len(seq_result)), seq_result)
    ax2.set_title("Sequential")
    ax2.set_xlabel("Index")
    for i, v in enumerate(seq_result):
        ax2.text(i, v + 0.5, str(v), ha="center")

    ax3.bar(range(len(par_result_inclusive)), par_result_inclusive)
    ax3.set_title("Parallel")
    ax3.set_xlabel("Index")
    for i, v in enumerate(par_result_inclusive):
        ax3.text(i, v + 0.5, str(v), ha="center")

    plt.tight_layout()
    plt.savefig("../../plots/task1_1_comparison.png", dpi=150, bbox_inches="tight")
    print("✓ Saved comparison visualization: plots/task1_1_comparison.png")
    plt.close()

    return seq_result, par_result_inclusive, seq_steps, par_steps, seq_ops, par_ops


def main():
    print("\n" + "=" * 80)
    print(" " * 20 + "TASK 1.1: PREFIX SUM ALGORITHMS")
    print(" " * 15 + "Sequential and Parallel Implementation")
    print("=" * 80)
    print()

    x = [2, 4, 6, 8, 1, 3, 5, 7]
    print(f"Given Array: {x}")
    print(f"Array Size: n = {len(x)}")
    print()

    print("Running algorithms...")
    print()

    seq_result, par_result, seq_steps, par_steps, seq_ops, par_ops = (
        create_comparison_visualization(x)
    )

    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()
    print(f"Sequential Result: {seq_result}")
    print(f"Parallel Result:   {par_result}")
    print(f"Results Match:     {'✓ YES' if seq_result == par_result else '✗ NO'}")
    print()

    print("-" * 80)
    print("PERFORMANCE METRICS")
    print("-" * 80)
    print()
    print(f"{'Metric':<25} {'Sequential':<15} {'Parallel':<15} {'Speedup':<15}")
    print("-" * 80)
    print(
        f"{'Time Steps':<25} {seq_steps:<15} {par_steps:<15} {seq_steps / par_steps:.2f}x"
    )
    print(f"{'Operations':<25} {seq_ops:<15} {par_ops:<15} {par_ops / seq_ops:.2f}x")
    print(f"{'CPUs Required':<25} {'1':<15} {'4 (max)':<15} {'─':<15}")
    print(f"{'Time Complexity':<25} {'O(n)':<15} {'O(log n)':<15} {'Better':<15}")
    print()

    print("-" * 80)
    print("KEY INSIGHTS")
    print("-" * 80)
    print()
    print("1. Sequential Algorithm:")
    print("   - Simple and intuitive implementation")
    print("   - O(n) time complexity - each element depends on previous")
    print("   - Requires only 1 CPU")
    print("   - Best for small arrays or single-core systems")
    print()
    print("2. Parallel Algorithm (Blelloch):")
    print("   - Work-efficient parallel algorithm")
    print("   - O(log n) time complexity with parallel execution")
    print("   - Requires O(n/log n) average parallelism")
    print("   - Best for large arrays on multi-core/GPU systems")
    print()
    print("3. Trade-offs:")
    print(f"   - Speedup: {seq_steps / par_steps:.2f}x faster time steps")
    print(f"   - Resource cost: {par_ops / seq_ops:.2f}x more total operations")
    print("   - Parallel requires 4 CPUs vs 1 CPU for sequential")
    print()

    print("=" * 80)
    print("Task 1.1 completed! Check the plots/ directory for visualizations.")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
