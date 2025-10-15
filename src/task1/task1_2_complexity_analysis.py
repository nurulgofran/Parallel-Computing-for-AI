import matplotlib.pyplot as plt
import numpy as np
import math


def analyze_sequential(n):
    time_steps = n - 1
    operations = n - 1
    cpus = 1

    return {
        "time_steps": time_steps,
        "operations": operations,
        "cpus": cpus,
        "time_complexity": "O(n)",
        "work_complexity": "O(n)",
        "space_complexity": "O(n)",
    }


def analyze_parallel(n):
    if n & (n - 1) != 0:
        raise ValueError("n must be power of 2 for Blelloch algorithm")

    log_n = int(math.log2(n))

    up_sweep_steps = log_n
    up_sweep_ops = n - 1

    down_sweep_steps = log_n
    down_sweep_ops = n - 1

    time_steps = up_sweep_steps + down_sweep_steps
    operations = up_sweep_ops + down_sweep_ops
    max_cpus = n // 2

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


def create_complexity_chart():
    n = 8
    seq = analyze_sequential(n)
    par = analyze_parallel(n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    metrics = ["Time Steps", "Operations", "CPUs"]
    seq_vals = [seq["time_steps"], seq["operations"], seq["cpus"]]
    par_vals = [par["time_steps"], par["operations"], par["cpus"]]

    x = np.arange(len(metrics))
    width = 0.35

    ax1.bar(x - width / 2, seq_vals, width, label="Sequential")
    ax1.bar(x + width / 2, par_vals, width, label="Parallel")
    ax1.set_ylabel("Value")
    ax1.set_title("Task 1.2: Complexity Metrics")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()

    for i, v in enumerate(seq_vals):
        ax1.text(i - width / 2, v + 0.2, str(v), ha="center")
    for i, v in enumerate(par_vals):
        ax1.text(i + width / 2, v + 0.2, str(v), ha="center")

    speedup = seq["time_steps"] / par["time_steps"]
    efficiency = speedup / par["cpus"] * 100

    ax2.bar(["Speedup", "Efficiency (%)"], [speedup, efficiency])
    ax2.set_title("Performance Metrics")
    ax2.set_ylabel("Value")
    for i, v in enumerate([speedup, efficiency]):
        ax2.text(i, v + 0.05, f"{v:.2f}", ha="center")

    plt.tight_layout()
    plt.savefig(
        "../../plots/task1_2_complexity_analysis.png", dpi=150, bbox_inches="tight"
    )
    print("✓ Saved: plots/task1_2_complexity_analysis.png")
    plt.close()


def create_scaling_charts():
    sizes = [2**i for i in range(1, 11)]

    seq_times = []
    par_times = []
    speedups = []

    for n in sizes:
        seq = analyze_sequential(n)
        par = analyze_parallel(n)
        seq_times.append(seq["time_steps"])
        par_times.append(par["time_steps"])
        speedups.append(seq["time_steps"] / par["time_steps"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(sizes, seq_times, "o-", label="Sequential")
    ax1.plot(sizes, par_times, "s-", label="Parallel")
    ax1.set_xlabel("Array Size (n)")
    ax1.set_ylabel("Time Steps")
    ax1.set_title("Time Steps vs Array Size")
    ax1.legend()
    ax1.set_xscale("log", base=2)

    ax2.plot(sizes, speedups, "o-")
    ax2.set_xlabel("Array Size (n)")
    ax2.set_ylabel("Speedup")
    ax2.set_title("Speedup vs Array Size")
    ax2.set_xscale("log", base=2)

    plt.tight_layout()
    plt.savefig(
        "../../plots/task1_2_scaling_analysis.png", dpi=150, bbox_inches="tight"
    )
    print("✓ Saved: plots/task1_2_scaling_analysis.png")
    plt.close()


def print_detailed_analysis():
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
    print_detailed_analysis()

    print("\nGenerating visualizations...")
    create_complexity_chart()
    create_scaling_charts()

    print("\n" + "=" * 80)
    print("Task 1.2 completed! All analyses and visualizations generated.")
    print("=" * 80)


if __name__ == "__main__":
    main()
