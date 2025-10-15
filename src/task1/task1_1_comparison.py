"""
Task 1.1: Run both Sequential and Parallel Prefix Sum Algorithms
PARMAI Exercise Sheet 1 - Think Parallel (20 pt.)

This script runs both algorithms and creates a comparison.
"""

import sys
import os

# Import both implementations
from task1_1_prefix_sum_sequential import sequential_prefix_sum
from task1_1_prefix_sum_parallel import (
    parallel_prefix_sum_simulation,
    calculate_parallel_metrics,
)

import matplotlib.pyplot as plt
import numpy as np
import math


def create_comparison_visualization(x):
    """Create a comprehensive comparison of both approaches"""

    # Run both algorithms
    seq_result, seq_ops = sequential_prefix_sum(x)
    par_result, par_steps, par_ops, par_details = parallel_prefix_sum_simulation(x)

    # Convert parallel result to inclusive (to match sequential)
    par_result_inclusive = par_result[1:] + [sum(x)]

    # Calculate metrics
    n = len(x)
    seq_steps = n - 1
    par_metrics = calculate_parallel_metrics(n)

    # Create comprehensive comparison figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # Title
    fig.suptitle(
        "Prefix Sum: Sequential vs Parallel Comparison",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Input Array
    ax1 = fig.add_subplot(gs[0, :])
    ax1.bar(range(len(x)), x, color="skyblue", edgecolor="navy", linewidth=2)
    ax1.set_xlabel("Index", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Value", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Input Array: x = [2, 4, 6, 8, 1, 3, 5, 7]", fontsize=13, fontweight="bold"
    )
    ax1.set_xticks(range(len(x)))
    for i, v in enumerate(x):
        ax1.text(i, v + 0.3, str(v), ha="center", fontweight="bold", fontsize=11)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Sequential Result
    ax2 = fig.add_subplot(gs[1, 0])
    bars1 = ax2.bar(
        range(len(seq_result)),
        seq_result,
        color="lightcoral",
        edgecolor="darkred",
        linewidth=2,
    )
    ax2.set_xlabel("Index", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Value", fontsize=11, fontweight="bold")
    ax2.set_title(
        "Sequential Prefix Sum", fontsize=12, fontweight="bold", color="darkred"
    )
    ax2.set_xticks(range(len(seq_result)))
    for i, v in enumerate(seq_result):
        ax2.text(i, v + 0.5, str(v), ha="center", fontweight="bold", fontsize=10)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    # Parallel Result
    ax3 = fig.add_subplot(gs[1, 1])
    bars2 = ax3.bar(
        range(len(par_result_inclusive)),
        par_result_inclusive,
        color="lightgreen",
        edgecolor="darkgreen",
        linewidth=2,
    )
    ax3.set_xlabel("Index", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Value", fontsize=11, fontweight="bold")
    ax3.set_title(
        "Parallel Prefix Sum (Blelloch)",
        fontsize=12,
        fontweight="bold",
        color="darkgreen",
    )
    ax3.set_xticks(range(len(par_result_inclusive)))
    for i, v in enumerate(par_result_inclusive):
        ax3.text(i, v + 0.5, str(v), ha="center", fontweight="bold", fontsize=10)
    ax3.grid(axis="y", alpha=0.3, linestyle="--")

    # Metrics Comparison Table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("off")

    # Create comparison table
    table_data = [
        ["Metric", "Sequential", "Parallel (Blelloch)", "Advantage"],
        ["─" * 20, "─" * 15, "─" * 20, "─" * 15],
        [
            "Time Steps",
            f"{seq_steps}",
            f"{par_steps}",
            f"Parallel: {seq_steps / par_steps:.1f}x faster",
        ],
        ["Total Operations", f"{seq_ops}", f"{par_ops}", "Similar work"],
        [
            "CPUs Required",
            "1",
            f"{par_metrics['max_cpus']}",
            "Parallel: more resources",
        ],
        ["Time Complexity", "O(n)", "O(log n)", "Parallel: better"],
        ["Work Complexity", "O(n)", "O(n)", "Same"],
        ["Space Complexity", "O(n)", "O(n)", "Same"],
        ["Best Use Case", "Small arrays", "Large arrays", "─"],
        ["", "Single CPU", "Multi-core/GPU", "─"],
    ]

    # Format table
    col_widths = [0.25, 0.20, 0.25, 0.30]
    y_start = 0.85
    y_step = 0.08

    for i, row in enumerate(table_data):
        y_pos = y_start - i * y_step
        x_pos = 0.05

        for j, cell in enumerate(row):
            if i == 0:  # Header
                weight = "bold"
                size = 11
            elif i == 1:  # Separator
                weight = "normal"
                size = 10
            else:
                weight = "normal"
                size = 10

            ax4.text(
                x_pos,
                y_pos,
                cell,
                fontsize=size,
                fontweight=weight,
                family="monospace",
                transform=ax4.transAxes,
                verticalalignment="center",
            )
            x_pos += col_widths[j]

    # Summary box
    summary_y = y_start - len(table_data) * y_step - 0.08
    summary_text = (
        "SUMMARY: The parallel algorithm achieves better time complexity O(log n) vs O(n), "
        f"resulting in {seq_steps / par_steps:.1f}x speedup,\n"
        "but requires more computational resources (CPUs). For this array of 8 elements, "
        "parallel execution\n"
        "completes in 6 steps vs 7 steps sequentially, using up to 4 parallel processors."
    )

    ax4.text(
        0.05,
        summary_y,
        summary_text,
        fontsize=10,
        style="italic",
        transform=ax4.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.savefig("../../plots/task1_1_comparison.png", dpi=300, bbox_inches="tight")
    print("✓ Saved comparison visualization: plots/task1_1_comparison.png")
    plt.close()

    return seq_result, par_result_inclusive, seq_steps, par_steps, seq_ops, par_ops


def main():
    """Run complete Task 1.1 analysis"""
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

    # Create comparison
    seq_result, par_result, seq_steps, par_steps, seq_ops, par_ops = (
        create_comparison_visualization(x)
    )

    # Display results
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
