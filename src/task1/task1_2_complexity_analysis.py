"""
Task 1.2: Calculate Time Steps, Operations, and Required CPUs
PARMAI Exercise Sheet 1 - Think Parallel (20 pt.)

This module provides detailed analysis of both sequential and parallel
prefix sum algorithms, calculating:
1. Number of time steps
2. Number of operations
3. Number of required CPUs

Given array: x = [2, 4, 6, 8, 1, 3, 5, 7] (n = 8)
"""

import matplotlib.pyplot as plt
import numpy as np
import math


def analyze_sequential(n):
    """
    Analyze Sequential Prefix Sum Algorithm

    Algorithm: prefix[i] = prefix[i-1] + x[i] for i = 1 to n-1

    Args:
        n: Array size

    Returns:
        Dictionary with analysis metrics
    """
    time_steps = n - 1  # Each step depends on previous
    operations = n - 1  # One addition per step
    cpus = 1  # Sequential execution

    return {
        "time_steps": time_steps,
        "operations": operations,
        "cpus": cpus,
        "time_complexity": "O(n)",
        "work_complexity": "O(n)",
        "space_complexity": "O(n)",
    }


def analyze_parallel(n):
    """
    Analyze Parallel Prefix Sum Algorithm (Blelloch)

    Two phases:
    1. Up-sweep (Reduce): Build binary tree of partial sums - log(n) steps
    2. Down-sweep: Distribute values down tree - log(n) steps

    Args:
        n: Array size (must be power of 2)

    Returns:
        Dictionary with analysis metrics
    """
    if n & (n - 1) != 0:
        raise ValueError("n must be power of 2 for Blelloch algorithm")

    log_n = int(math.log2(n))

    # Up-sweep phase
    up_sweep_steps = log_n
    up_sweep_ops = n - 1  # Sum of n/2 + n/4 + ... + 1 = n-1

    # Down-sweep phase
    down_sweep_steps = log_n
    down_sweep_ops = n - 1  # Same as up-sweep

    # Total
    time_steps = up_sweep_steps + down_sweep_steps  # 2 * log(n)
    operations = up_sweep_ops + down_sweep_ops  # 2(n-1)
    max_cpus = n // 2  # Maximum parallelism at first level

    return {
        "time_steps": time_steps,
        "up_sweep_steps": up_sweep_steps,
        "down_sweep_steps": down_sweep_steps,
        "operations": operations,
        "up_sweep_ops": up_sweep_ops,
        "down_sweep_ops": down_sweep_ops,
        "cpus": max_cpus,
        "time_complexity": "O(log n)",
        "work_complexity": "O(n)",
        "space_complexity": "O(n)",
        "parallelism": n / log_n if log_n > 0 else n,
    }


def create_complexity_table():
    """Create detailed complexity comparison table"""
    n = 8
    seq = analyze_sequential(n)
    par = analyze_parallel(n)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis("off")

    # Title
    title = "Task 1.2: Complexity Analysis - Sequential vs Parallel Prefix Sum"
    ax.text(
        0.5,
        0.95,
        title,
        ha="center",
        fontsize=16,
        fontweight="bold",
        transform=ax.transAxes,
    )

    # Given information
    ax.text(
        0.5,
        0.90,
        f"Given Array: x = [2, 4, 6, 8, 1, 3, 5, 7]   |   n = {n}",
        ha="center",
        fontsize=12,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    )

    # Main comparison table
    y_start = 0.82

    table_data = [
        ["Metric", "Sequential", "Parallel (Blelloch)", "Ratio/Notes"],
        ["═" * 25, "═" * 20, "═" * 25, "═" * 30],
        ["", "", "", ""],
        ["1. TIME STEPS", "", "", ""],
        ["─" * 25, "─" * 20, "─" * 25, "─" * 30],
        [
            "Total Time Steps",
            f"{seq['time_steps']}",
            f"{par['time_steps']}",
            f"Sequential / Parallel = {seq['time_steps'] / par['time_steps']:.2f}x",
        ],
        [
            "  Up-sweep Phase",
            "─",
            f"{par['up_sweep_steps']}",
            f"log₂({n}) = {par['up_sweep_steps']}",
        ],
        [
            "  Down-sweep Phase",
            "─",
            f"{par['down_sweep_steps']}",
            f"log₂({n}) = {par['down_sweep_steps']}",
        ],
        [
            "Time Complexity",
            seq["time_complexity"],
            par["time_complexity"],
            "Parallel is asymptotically better",
        ],
        ["", "", "", ""],
        ["2. OPERATIONS", "", "", ""],
        ["─" * 25, "─" * 20, "─" * 25, "─" * 30],
        [
            "Total Operations",
            f"{seq['operations']}",
            f"{par['operations']}",
            f"Parallel / Sequential = {par['operations'] / seq['operations']:.2f}x",
        ],
        [
            "  Up-sweep",
            "─",
            f"{par['up_sweep_ops']}",
            f"{n} - 1 = {par['up_sweep_ops']}",
        ],
        [
            "  Down-sweep",
            "─",
            f"{par['down_sweep_ops']}",
            f"{n} - 1 = {par['down_sweep_ops']}",
        ],
        [
            "Work Complexity",
            seq["work_complexity"],
            par["work_complexity"],
            "Same amount of total work",
        ],
        ["", "", "", ""],
        ["3. CPU RESOURCES", "", "", ""],
        ["─" * 25, "─" * 20, "─" * 25, "─" * 30],
        [
            "CPUs Required",
            f"{seq['cpus']}",
            f"{par['cpus']} (max)",
            f"{par['cpus']}x more processors needed",
        ],
        [
            "Average Parallelism",
            "1",
            f"{par['parallelism']:.2f}",
            "n / log(n) average processors",
        ],
        [
            "Utilization",
            "100%",
            f"{par['parallelism'] / par['cpus'] * 100:.1f}% avg",
            "Not all CPUs busy all the time",
        ],
        ["", "", "", ""],
        ["4. COMPLEXITY CLASSES", "", "", ""],
        ["─" * 25, "─" * 20, "─" * 25, "─" * 30],
        ["Time Complexity", "O(n)", "O(log n)", "Parallel is faster"],
        ["Work Complexity", "O(n)", "O(n)", "Same total work"],
        [
            "Space Complexity",
            seq["space_complexity"],
            par["space_complexity"],
            "Both use O(n) space",
        ],
    ]

    col_widths = [0.27, 0.18, 0.25, 0.30]
    y_pos = y_start
    line_height = 0.022

    for i, row in enumerate(table_data):
        x_pos = 0.02

        # Determine styling
        if i == 0:  # Header
            weight = "bold"
            size = 10
            color = "white"
            bg_color = "navy"
        elif "═" in row[0] or "─" in row[0]:  # Separators
            weight = "normal"
            size = 8
            color = "gray"
            bg_color = None
        elif row[0].startswith(("1.", "2.", "3.", "4.")):  # Section headers
            weight = "bold"
            size = 10
            color = "darkblue"
            bg_color = "lightgray"
        elif row[0] == "":  # Empty rows
            y_pos -= line_height * 0.5
            continue
        else:
            weight = "normal"
            size = 9
            color = "black"
            bg_color = None

        # Draw background for special rows
        if bg_color:
            rect = plt.Rectangle(
                (x_pos - 0.01, y_pos - line_height * 0.3),
                0.98,
                line_height * 1.2,
                facecolor=bg_color,
                transform=ax.transAxes,
                zorder=0,
                alpha=0.3,
            )
            ax.add_patch(rect)

        # Draw cells
        for j, cell in enumerate(row):
            ax.text(
                x_pos,
                y_pos,
                str(cell),
                fontsize=size,
                fontweight=weight,
                family="monospace",
                transform=ax.transAxes,
                color=color,
                verticalalignment="center",
            )
            x_pos += col_widths[j]

        y_pos -= line_height

    # Key formulas and insights
    y_pos -= line_height * 2

    formulas_title = "KEY FORMULAS AND INSIGHTS"
    ax.text(
        0.02,
        y_pos,
        formulas_title,
        fontsize=11,
        fontweight="bold",
        transform=ax.transAxes,
        color="darkgreen",
    )
    y_pos -= line_height * 1.5

    formulas = [
        f"• Sequential: Each step i depends on step i-1 → T_seq = n-1 = {seq['time_steps']} steps",
        f"• Parallel Up-sweep: log₂(n) levels, level d has n/2^(d+1) operations → T_up = log₂({n}) = {par['up_sweep_steps']} steps",
        f"• Parallel Down-sweep: Same as up-sweep → T_down = log₂({n}) = {par['down_sweep_steps']} steps",
        f"• Total Parallel Time: T_par = 2·log₂(n) = {par['time_steps']} steps",
        f"• Speedup: S = T_seq / T_par = {seq['time_steps']} / {par['time_steps']} = {seq['time_steps'] / par['time_steps']:.2f}x",
        f"• Efficiency: E = S / P = {seq['time_steps'] / par['time_steps']:.2f} / {par['cpus']} = {(seq['time_steps'] / par['time_steps']) / par['cpus']:.2f} = {(seq['time_steps'] / par['time_steps']) / par['cpus'] * 100:.1f}%",
        f"• Cost: C = P × T_par = {par['cpus']} × {par['time_steps']} = {par['cpus'] * par['time_steps']} processor-time units",
    ]

    for formula in formulas:
        ax.text(
            0.04, y_pos, formula, fontsize=9, family="monospace", transform=ax.transAxes
        )
        y_pos -= line_height * 1.3

    # Conclusions
    y_pos -= line_height
    conclusions_title = "CONCLUSIONS"
    ax.text(
        0.02,
        y_pos,
        conclusions_title,
        fontsize=11,
        fontweight="bold",
        transform=ax.transAxes,
        color="darkred",
    )
    y_pos -= line_height * 1.5

    conclusions = [
        f"1. For n={n}: Parallel is {seq['time_steps'] / par['time_steps']:.2f}x faster but uses {par['operations'] / seq['operations']:.0f}x more operations",
        f"2. Parallel efficiency is {(seq['time_steps'] / par['time_steps']) / par['cpus'] * 100:.1f}% (not all CPUs utilized fully)",
        f"3. For larger n, parallel advantage grows: O(n) vs O(log n)",
        "4. Trade-off: Speed vs Resources (1 CPU vs 4 CPUs for n=8)",
    ]

    for conclusion in conclusions:
        ax.text(0.04, y_pos, conclusion, fontsize=9, transform=ax.transAxes)
        y_pos -= line_height * 1.3

    plt.tight_layout()
    plt.savefig(
        "../../plots/task1_2_complexity_analysis.png", dpi=300, bbox_inches="tight"
    )
    print("✓ Saved: plots/task1_2_complexity_analysis.png")
    plt.close()


def create_scaling_charts():
    """Create charts showing how algorithms scale with array size"""

    sizes = [2**i for i in range(1, 11)]  # 2, 4, 8, 16, ..., 1024

    seq_times = []
    par_times = []
    seq_ops = []
    par_ops = []
    speedups = []

    for n in sizes:
        seq = analyze_sequential(n)
        par = analyze_parallel(n)

        seq_times.append(seq["time_steps"])
        par_times.append(par["time_steps"])
        seq_ops.append(seq["operations"])
        par_ops.append(par["operations"])
        speedups.append(seq["time_steps"] / par["time_steps"])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Scaling Analysis: Sequential vs Parallel Prefix Sum",
        fontsize=14,
        fontweight="bold",
    )

    # Time Steps Comparison
    ax1.plot(
        sizes,
        seq_times,
        "o-",
        linewidth=2,
        markersize=8,
        label="Sequential O(n)",
        color="red",
    )
    ax1.plot(
        sizes,
        par_times,
        "s-",
        linewidth=2,
        markersize=8,
        label="Parallel O(log n)",
        color="green",
    )
    ax1.set_xlabel("Array Size (n)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Time Steps", fontsize=11, fontweight="bold")
    ax1.set_title("Time Steps vs Array Size", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log", base=2)

    # Mark n=8
    idx_8 = sizes.index(8)
    ax1.plot(
        8,
        seq_times[idx_8],
        "ro",
        markersize=12,
        markerfacecolor="none",
        markeredgewidth=2,
        label="n=8",
    )
    ax1.plot(
        8,
        par_times[idx_8],
        "gs",
        markersize=12,
        markerfacecolor="none",
        markeredgewidth=2,
    )

    # Operations Comparison
    ax2.plot(
        sizes, seq_ops, "o-", linewidth=2, markersize=8, label="Sequential", color="red"
    )
    ax2.plot(
        sizes, par_ops, "s-", linewidth=2, markersize=8, label="Parallel", color="green"
    )
    ax2.set_xlabel("Array Size (n)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Total Operations", fontsize=11, fontweight="bold")
    ax2.set_title("Operations vs Array Size", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log", base=2)

    # Mark n=8
    ax2.plot(
        8,
        seq_ops[idx_8],
        "ro",
        markersize=12,
        markerfacecolor="none",
        markeredgewidth=2,
    )
    ax2.plot(
        8,
        par_ops[idx_8],
        "gs",
        markersize=12,
        markerfacecolor="none",
        markeredgewidth=2,
    )

    # Speedup
    ax3.plot(sizes, speedups, "o-", linewidth=2, markersize=8, color="blue")
    ax3.axhline(y=1, color="red", linestyle="--", linewidth=1, label="No speedup")
    ax3.set_xlabel("Array Size (n)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Speedup (T_seq / T_par)", fontsize=11, fontweight="bold")
    ax3.set_title("Speedup Factor vs Array Size", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale("log", base=2)

    # Mark n=8
    ax3.plot(
        8,
        speedups[idx_8],
        "bo",
        markersize=12,
        markerfacecolor="none",
        markeredgewidth=2,
        label=f"n=8: {speedups[idx_8]:.2f}x",
    )
    ax3.legend(fontsize=10)

    # CPUs Required (Parallel)
    cpus_required = [n // 2 for n in sizes]
    ax4.plot(sizes, cpus_required, "s-", linewidth=2, markersize=8, color="purple")
    ax4.set_xlabel("Array Size (n)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Max CPUs Required", fontsize=11, fontweight="bold")
    ax4.set_title("Parallel: Maximum CPUs Required", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale("log", base=2)
    ax4.set_yscale("log", base=2)

    # Mark n=8
    ax4.plot(
        8,
        cpus_required[idx_8],
        "ms",
        markersize=12,
        markerfacecolor="none",
        markeredgewidth=2,
        label=f"n=8: {cpus_required[idx_8]} CPUs",
    )
    ax4.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(
        "../../plots/task1_2_scaling_analysis.png", dpi=300, bbox_inches="tight"
    )
    print("✓ Saved: plots/task1_2_scaling_analysis.png")
    plt.close()


def print_detailed_analysis():
    """Print detailed textual analysis"""
    n = 8
    seq = analyze_sequential(n)
    par = analyze_parallel(n)

    print("=" * 80)
    print(" " * 20 + "TASK 1.2: DETAILED COMPLEXITY ANALYSIS")
    print("=" * 80)
    print()
    print(f"Given Array: x = [2, 4, 6, 8, 1, 3, 5, 7]")
    print(f"Array Size: n = {n}")
    print()

    print("-" * 80)
    print("1. SEQUENTIAL ALGORITHM ANALYSIS")
    print("-" * 80)
    print()
    print("Algorithm Description:")
    print("  for i = 1 to n-1:")
    print("    prefix[i] = prefix[i-1] + array[i]")
    print()
    print(f"Time Steps:          {seq['time_steps']} steps")
    print(f"  Reasoning:         Each iteration depends on previous result")
    print(f"                     Step i cannot start until step i-1 completes")
    print()
    print(f"Operations:          {seq['operations']} additions")
    print(f"  Reasoning:         One addition per iteration (i=1 to n-1)")
    print()
    print(f"CPUs Required:       {seq['cpus']} CPU")
    print(f"  Reasoning:         Sequential execution, no parallelism possible")
    print()
    print(f"Time Complexity:     {seq['time_complexity']}")
    print(f"Work Complexity:     {seq['work_complexity']}")
    print(f"Space Complexity:    {seq['space_complexity']}")
    print()

    print("-" * 80)
    print("2. PARALLEL ALGORITHM ANALYSIS (Blelloch)")
    print("-" * 80)
    print()
    print("Algorithm Description:")
    print("  Phase 1 - Up-sweep (Reduce):")
    print(f"    Build binary tree of partial sums in {par['up_sweep_steps']} levels")
    print("  Phase 2 - Down-sweep (Distribution):")
    print(f"    Distribute values down tree in {par['down_sweep_steps']} levels")
    print()
    print(f"Time Steps:          {par['time_steps']} steps total")
    print(f"  Up-sweep:          {par['up_sweep_steps']} steps (log₂({n}))")
    print(f"  Down-sweep:        {par['down_sweep_steps']} steps (log₂({n}))")
    print(f"  Reasoning:         Each level processes independently in parallel")
    print()
    print(f"Operations:          {par['operations']} operations total")
    print(f"  Up-sweep:          {par['up_sweep_ops']} operations")
    print("                     Level 0: n/2 = 4 ops")
    print("                     Level 1: n/4 = 2 ops")
    print("                     Level 2: n/8 = 1 op")
    print(f"                     Total: 4+2+1 = {par['up_sweep_ops']}")
    print(f"  Down-sweep:        {par['down_sweep_ops']} operations")
    print("                     Mirror of up-sweep")
    print()
    print(f"CPUs Required:       {par['cpus']} CPUs (maximum)")
    print(f"  Reasoning:         Maximum parallelism at level 0 (n/2)")
    print(f"  Average:           {par['parallelism']:.2f} CPUs active on average")
    print()
    print(f"Time Complexity:     {par['time_complexity']}")
    print(f"Work Complexity:     {par['work_complexity']}")
    print(f"Space Complexity:    {par['space_complexity']}")
    print(f"Parallelism:         {par['parallelism']:.2f} (average)")
    print()

    print("-" * 80)
    print("3. COMPARISON AND PERFORMANCE METRICS")
    print("-" * 80)
    print()
    speedup = seq["time_steps"] / par["time_steps"]
    efficiency = speedup / par["cpus"]
    cost_seq = seq["cpus"] * seq["time_steps"]
    cost_par = par["cpus"] * par["time_steps"]

    print(f"Speedup (S):         {speedup:.2f}x")
    print(f"  Formula:           S = T_sequential / T_parallel")
    print(
        f"  Calculation:       S = {seq['time_steps']} / {par['time_steps']} = {speedup:.2f}"
    )
    print()
    print(f"Efficiency (E):      {efficiency:.2f} = {efficiency * 100:.1f}%")
    print(f"  Formula:           E = Speedup / Number_of_CPUs")
    print(f"  Calculation:       E = {speedup:.2f} / {par['cpus']} = {efficiency:.2f}")
    print(
        f"  Interpretation:    {efficiency * 100:.1f}% of parallel resources effectively used"
    )
    print()
    print(f"Cost (Sequential):   {cost_seq} processor-time units")
    print(f"  Formula:           Cost = CPUs × Time")
    print(f"  Calculation:       {seq['cpus']} × {seq['time_steps']} = {cost_seq}")
    print()
    print(f"Cost (Parallel):     {cost_par} processor-time units")
    print(f"  Calculation:       {par['cpus']} × {par['time_steps']} = {cost_par}")
    print(f"  Overhead:          {cost_par / cost_seq:.2f}x more expensive")
    print()

    print("-" * 80)
    print("4. SUMMARY AND CONCLUSIONS")
    print("-" * 80)
    print()
    print(f"For n = {n}:")
    print(f"  ✓ Parallel is {speedup:.2f}x faster in time")
    print(
        f"  ✗ Parallel uses {par['operations'] / seq['operations']:.1f}x more operations"
    )
    print(f"  ✗ Parallel requires {par['cpus']}x more processors")
    print(f"  • Efficiency is {efficiency * 100:.1f}% (moderate)")
    print()
    print("When to use Sequential:")
    print("  • Small arrays (n < 100)")
    print("  • Single-core systems")
    print("  • When CPU resources are limited")
    print("  • Simple implementation preferred")
    print()
    print("When to use Parallel:")
    print("  • Large arrays (n > 1000)")
    print("  • Multi-core CPUs or GPUs available")
    print("  • Time-critical applications")
    print("  • Scales better: O(log n) vs O(n)")
    print()
    print("=" * 80)


def main():
    """Main function for Task 1.2"""
    print_detailed_analysis()

    print("\nGenerating visualizations...")
    create_complexity_table()
    create_scaling_charts()

    print("\n" + "=" * 80)
    print("Task 1.2 completed! All analyses and visualizations generated.")
    print("=" * 80)


if __name__ == "__main__":
    main()
