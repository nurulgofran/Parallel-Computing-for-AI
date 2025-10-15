"""
Task 1.1: Parallel Prefix Sum Algorithm (Blelloch's Algorithm)
PARMAI Exercise Sheet 1 - Think Parallel (20 pt.)

Given array: x = [2, 4, 6, 8, 1, 3, 5, 7]

Parallel Algorithm (Work-Efficient):
- Uses the "up-sweep" (reduce) and "down-sweep" phases
- Time Complexity: O(log n)
- Work Complexity: O(n)
- Highly efficient for parallel execution

Reference: Guy E. Blelloch, Prefix Sums and Their Applications,
School of Computer Science Carnegie Mellon University Pittsburgh.
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import math


def parallel_prefix_sum_simulation(arr):
    """
    Parallel Prefix Sum Algorithm (Blelloch's Work-Efficient Algorithm)

    This simulates parallel execution with step-by-step tracking.

    Algorithm:
    1. Up-sweep (Reduce) Phase: Build a binary tree of partial sums
    2. Down-sweep Phase: Traverse down the tree to compute prefix sums

    Time Complexity: O(log n) with O(n) processors
    Work Complexity: O(n)

    Args:
        arr: Input array (length must be power of 2)

    Returns:
        Prefix sum array, time steps, operations count, and step details
    """
    n = len(arr)

    # Ensure n is a power of 2 for this algorithm
    if n & (n - 1) != 0:
        raise ValueError("Array length must be a power of 2")

    # Working array (will be modified in place)
    result = arr.copy()

    # Track operations and steps
    time_steps = 0
    total_operations = 0
    step_details = []

    # UP-SWEEP PHASE (Reduce phase)
    print("\n" + "=" * 70)
    print("UP-SWEEP PHASE (Building partial sums tree)")
    print("=" * 70)

    d_levels = int(math.log2(n))

    for d in range(d_levels):
        stride = 2 ** (d + 1)
        operations_this_step = 0
        modified_indices = []

        # Parallel operations at this level
        for k in range(0, n, stride):
            left_idx = k + 2**d - 1
            right_idx = k + 2 ** (d + 1) - 1

            result[right_idx] += result[left_idx]
            operations_this_step += 1
            modified_indices.append((left_idx, right_idx, result[right_idx]))

        time_steps += 1
        total_operations += operations_this_step

        step_info = {
            "phase": "up-sweep",
            "level": d,
            "stride": stride,
            "operations": operations_this_step,
            "array_state": result.copy(),
            "modified": modified_indices,
        }
        step_details.append(step_info)

        print(f"Level {d}: stride={stride}, operations={operations_this_step}")
        print(f"  Array: {result}")
        for left, right, val in modified_indices:
            print(f"  result[{right}] = result[{right}] + result[{left}] = {val}")

    # Set last element to 0 (identity for addition)
    result[n - 1] = 0

    # DOWN-SWEEP PHASE
    print("\n" + "=" * 70)
    print("DOWN-SWEEP PHASE (Computing prefix sums)")
    print("=" * 70)

    for d in range(d_levels - 1, -1, -1):
        stride = 2 ** (d + 1)
        operations_this_step = 0
        modified_indices = []

        # Parallel operations at this level
        for k in range(0, n, stride):
            left_idx = k + 2**d - 1
            right_idx = k + 2 ** (d + 1) - 1

            temp = result[left_idx]
            result[left_idx] = result[right_idx]
            result[right_idx] = result[right_idx] + temp
            operations_this_step += 2  # One assignment + one addition
            modified_indices.append((left_idx, right_idx))

        time_steps += 1
        total_operations += operations_this_step

        step_info = {
            "phase": "down-sweep",
            "level": d,
            "stride": stride,
            "operations": operations_this_step,
            "array_state": result.copy(),
            "modified": modified_indices,
        }
        step_details.append(step_info)

        print(f"Level {d}: stride={stride}, operations={operations_this_step}")
        print(f"  Array: {result}")

    return result, time_steps, total_operations, step_details


def calculate_parallel_metrics(n):
    """Calculate theoretical metrics for parallel algorithm"""
    log_n = int(math.log2(n))

    # Up-sweep operations
    up_sweep_ops = n - 1

    # Down-sweep operations
    down_sweep_ops = n - 1

    total_ops = up_sweep_ops + down_sweep_ops
    time_steps = 2 * log_n  # log(n) for up-sweep + log(n) for down-sweep

    # CPUs needed at each level
    max_cpus = n // 2

    return {
        "time_steps": time_steps,
        "total_operations": total_ops,
        "max_cpus": max_cpus,
        "up_sweep_ops": up_sweep_ops,
        "down_sweep_ops": down_sweep_ops,
    }


def visualize_parallel(arr, prefix_sum, metrics, step_details):
    """Visualize the parallel prefix sum computation"""
    fig = plt.figure(figsize=(14, 12))

    # Create grid for subplots
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

    # Original Array
    ax1 = fig.add_subplot(gs[0, :])
    ax1.bar(range(len(arr)), arr, color="lightblue", edgecolor="black", linewidth=2)
    ax1.set_xlabel("Index", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Value", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Original Array x = [2, 4, 6, 8, 1, 3, 5, 7]", fontsize=14, fontweight="bold"
    )
    ax1.set_xticks(range(len(arr)))
    for i, v in enumerate(arr):
        ax1.text(i, v + 0.5, str(v), ha="center", fontweight="bold", fontsize=11)
    ax1.grid(axis="y", alpha=0.3)

    # Prefix Sum Result
    ax2 = fig.add_subplot(gs[1, :])
    ax2.bar(
        range(len(prefix_sum)),
        prefix_sum,
        color="lightgreen",
        edgecolor="black",
        linewidth=2,
    )
    ax2.set_xlabel("Index", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Value", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Prefix Sum (Parallel - Blelloch Algorithm)", fontsize=14, fontweight="bold"
    )
    ax2.set_xticks(range(len(prefix_sum)))
    for i, v in enumerate(prefix_sum):
        ax2.text(i, v + 0.5, str(v), ha="center", fontweight="bold", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    # Algorithm Phases Diagram
    ax3 = fig.add_subplot(gs[2, :])
    ax3.text(
        0.5,
        0.95,
        "Parallel Algorithm Phases (Blelloch)",
        ha="center",
        fontsize=13,
        fontweight="bold",
        transform=ax3.transAxes,
    )

    phase_text = []
    phase_text.append("UP-SWEEP PHASE (Reduce - Build partial sums):")

    up_steps = [s for s in step_details if s["phase"] == "up-sweep"]
    for step in up_steps:
        phase_text.append(
            f"  Step {step['level']}: stride={step['stride']:2d}, "
            f"ops={step['operations']:2d}, array={step['array_state']}"
        )

    phase_text.append("")
    phase_text.append("DOWN-SWEEP PHASE (Distribution - Compute prefix sums):")

    down_steps = [s for s in step_details if s["phase"] == "down-sweep"]
    for i, step in enumerate(down_steps):
        phase_text.append(
            f"  Step {i}: stride={step['stride']:2d}, "
            f"ops={step['operations']:2d}, array={step['array_state']}"
        )

    y_pos = 0.85
    for text in phase_text:
        ax3.text(
            0.05, y_pos, text, fontsize=9, family="monospace", transform=ax3.transAxes
        )
        y_pos -= 0.055

    ax3.axis("off")

    # Complexity Analysis
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.text(
        0.5,
        0.95,
        "Complexity Analysis",
        ha="center",
        fontsize=12,
        fontweight="bold",
        transform=ax4.transAxes,
    )

    complexity_info = [
        f"Time Steps: {metrics['time_steps']}",
        f"Total Operations: {metrics['total_operations']}",
        f"Up-sweep ops: {metrics['up_sweep_ops']}",
        f"Down-sweep ops: {metrics['down_sweep_ops']}",
        f"Max CPUs needed: {metrics['max_cpus']}",
        f"Time Complexity: O(log n)",
        f"Work Complexity: O(n)",
        f"Parallelism: O(n/log n)",
    ]

    y_pos = 0.80
    for info in complexity_info:
        ax4.text(
            0.1, y_pos, info, fontsize=10, family="monospace", transform=ax4.transAxes
        )
        y_pos -= 0.10

    ax4.axis("off")

    # Comparison with Sequential
    ax5 = fig.add_subplot(gs[3, 1])
    ax5.text(
        0.5,
        0.95,
        "Sequential vs Parallel",
        ha="center",
        fontsize=12,
        fontweight="bold",
        transform=ax5.transAxes,
    )

    n = len(arr)
    comparison = [
        "Metric           Sequential  Parallel",
        "─" * 40,
        f"Time Steps       {n - 1:4d}       {metrics['time_steps']:4d}",
        f"Total Ops        {n - 1:4d}       {metrics['total_operations']:4d}",
        f"CPUs             {1:4d}       {metrics['max_cpus']:4d}",
        f"Complexity       O(n)        O(log n)",
        "",
        f"Speedup: {(n - 1) / metrics['time_steps']:.2f}x faster",
    ]

    y_pos = 0.80
    for info in comparison:
        ax5.text(
            0.1, y_pos, info, fontsize=9, family="monospace", transform=ax5.transAxes
        )
        y_pos -= 0.10

    ax5.axis("off")

    plt.savefig("../../plots/task1_1_parallel.png", dpi=300, bbox_inches="tight")
    print("✓ Saved visualization: plots/task1_1_parallel.png")
    plt.close()


def main():
    """Main function to run parallel prefix sum"""
    print("=" * 70)
    print("TASK 1.1: PARALLEL PREFIX SUM ALGORITHM (BLELLOCH)")
    print("=" * 70)
    print()

    # Given array
    x = [2, 4, 6, 8, 1, 3, 5, 7]
    print(f"Input Array x: {x}")
    print(f"Array Length n: {len(x)} (power of 2: ✓)")
    print()

    # Run parallel algorithm
    start_time = time.perf_counter()
    prefix_sum, time_steps, operations, step_details = parallel_prefix_sum_simulation(x)
    end_time = time.perf_counter()

    execution_time = (end_time - start_time) * 1000  # Convert to milliseconds

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Prefix Sum Result: {prefix_sum}")
    print()

    # Calculate metrics
    metrics = calculate_parallel_metrics(len(x))

    print("-" * 70)
    print("PARALLEL ALGORITHM ANALYSIS (BLELLOCH)")
    print("-" * 70)
    print(f"Time Steps:        {time_steps} steps (parallel execution)")
    print(f"  Up-sweep:        {int(math.log2(len(x)))} steps")
    print(f"  Down-sweep:      {int(math.log2(len(x)))} steps")
    print()
    print(f"Total Operations:  {operations} operations")
    print(f"  Up-sweep:        {metrics['up_sweep_ops']} operations")
    print(f"  Down-sweep:      {metrics['down_sweep_ops']} operations")
    print()
    print(f"CPUs Required:     {metrics['max_cpus']} CPUs (maximum at any step)")
    print(
        f"Time Complexity:   O(log n) = O(log {len(x)}) = O({int(math.log2(len(x)))})"
    )
    print(f"Work Complexity:   O(n) = O({len(x)})")
    print(f"Execution Time:    {execution_time:.4f} ms (simulation)")
    print()

    # Verify correctness
    expected = [0, 2, 6, 12, 20, 21, 24, 29]  # Exclusive prefix sum
    # Note: Blelloch's algorithm produces exclusive prefix sum by default
    # For inclusive prefix sum (which matches sequential), we need to adjust

    # Convert to inclusive prefix sum
    inclusive_prefix = prefix_sum[1:] + [sum(x)]
    expected_inclusive = [2, 6, 12, 20, 21, 24, 29, 36]

    print(f"Exclusive Prefix Sum: {prefix_sum}")
    print(f"Inclusive Prefix Sum: {inclusive_prefix}")
    print()

    verification = (
        "✓ CORRECT" if inclusive_prefix == expected_inclusive else "✗ INCORRECT"
    )
    print(f"Verification:      {verification}")
    print(f"Expected (incl.):  {expected_inclusive}")
    print()

    # Create visualization
    print("Generating visualization...")
    visualize_parallel(x, inclusive_prefix, metrics, step_details)
    print()

    print("=" * 70)
    print("Parallel algorithm completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
