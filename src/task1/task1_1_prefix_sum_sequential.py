"""
Task 1.1: Sequential Prefix Sum Algorithm
PARMAI Exercise Sheet 1 - Think Parallel (20 pt.)

Given array: x = [2, 4, 6, 8, 1, 3, 5, 7]

Sequential Algorithm:
- Compute prefix sum sequentially from left to right
- Each element is the sum of all elements up to that position

Reference: Guy E. Blelloch, Prefix Sums and Their Applications,
School of Computer Science Carnegie Mellon University Pittsburgh.
"""

import time
import matplotlib.pyplot as plt
import numpy as np


def sequential_prefix_sum(arr):
    """
    Sequential Prefix Sum Algorithm

    For an array [x0, x1, x2, ..., xn-1], compute:
    [x0, x0+x1, x0+x1+x2, ..., x0+x1+...+xn-1]

    Time Complexity: O(n)
    Space Complexity: O(n)

    Args:
        arr: Input array

    Returns:
        Prefix sum array and operation count
    """
    n = len(arr)
    prefix_sum = [0] * n
    operations = 0

    # First element is same as input
    prefix_sum[0] = arr[0]

    # Each subsequent element is sum of previous prefix and current element
    for i in range(1, n):
        prefix_sum[i] = prefix_sum[i - 1] + arr[i]
        operations += 1  # One addition operation

    return prefix_sum, operations


def visualize_sequential(arr, prefix_sum, operations):
    """Visualize the sequential prefix sum computation"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # Original Array
    ax1.bar(range(len(arr)), arr, color="lightblue", edgecolor="black")
    ax1.set_xlabel("Index", fontsize=12)
    ax1.set_ylabel("Value", fontsize=12)
    ax1.set_title(
        "Original Array x = [2, 4, 6, 8, 1, 3, 5, 7]", fontsize=14, fontweight="bold"
    )
    ax1.set_xticks(range(len(arr)))
    for i, v in enumerate(arr):
        ax1.text(i, v + 0.5, str(v), ha="center", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # Prefix Sum Result
    ax2.bar(range(len(prefix_sum)), prefix_sum, color="lightgreen", edgecolor="black")
    ax2.set_xlabel("Index", fontsize=12)
    ax2.set_ylabel("Value", fontsize=12)
    ax2.set_title("Prefix Sum (Sequential)", fontsize=14, fontweight="bold")
    ax2.set_xticks(range(len(prefix_sum)))
    for i, v in enumerate(prefix_sum):
        ax2.text(i, v + 0.5, str(v), ha="center", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    # Sequential Process Diagram
    ax3.text(
        0.5,
        0.9,
        "Sequential Computation Process",
        ha="center",
        fontsize=14,
        fontweight="bold",
        transform=ax3.transAxes,
    )

    process_text = []
    process_text.append("Step 0: prefix[0] = x[0] = 2")
    for i in range(1, len(arr)):
        process_text.append(
            f"Step {i}: prefix[{i}] = prefix[{i - 1}] + x[{i}] = {prefix_sum[i - 1]} + {arr[i]} = {prefix_sum[i]}"
        )

    y_pos = 0.75
    for text in process_text:
        ax3.text(
            0.1, y_pos, text, fontsize=10, family="monospace", transform=ax3.transAxes
        )
        y_pos -= 0.09

    ax3.text(
        0.1,
        0.05,
        f"Total Operations: {operations}",
        fontsize=12,
        fontweight="bold",
        transform=ax3.transAxes,
    )
    ax3.axis("off")

    plt.tight_layout()
    plt.savefig("../../plots/task1_1_sequential.png", dpi=300, bbox_inches="tight")
    print("✓ Saved visualization: plots/task1_1_sequential.png")
    plt.close()


def main():
    """Main function to run sequential prefix sum"""
    print("=" * 70)
    print("TASK 1.1: SEQUENTIAL PREFIX SUM ALGORITHM")
    print("=" * 70)
    print()

    # Given array
    x = [2, 4, 6, 8, 1, 3, 5, 7]
    print(f"Input Array x: {x}")
    print(f"Array Length n: {len(x)}")
    print()

    # Measure execution time
    start_time = time.perf_counter()
    prefix_sum, operations = sequential_prefix_sum(x)
    end_time = time.perf_counter()

    execution_time = (end_time - start_time) * 1000  # Convert to milliseconds

    # Display results
    print(f"Prefix Sum Result: {prefix_sum}")
    print()

    print("-" * 70)
    print("SEQUENTIAL ALGORITHM ANALYSIS")
    print("-" * 70)
    print(f"Time Steps:        {len(x) - 1} steps (each addition is sequential)")
    print(f"Total Operations:  {operations} additions")
    print(f"CPUs Required:     1 CPU (sequential execution)")
    print(f"Time Complexity:   O(n) = O({len(x)})")
    print(f"Execution Time:    {execution_time:.4f} ms")
    print()

    # Verify correctness
    expected = [2, 6, 12, 20, 21, 24, 29, 36]
    verification = "✓ CORRECT" if prefix_sum == expected else "✗ INCORRECT"
    print(f"Verification:      {verification}")
    print(f"Expected Result:   {expected}")
    print()

    # Create visualization
    print("Generating visualization...")
    visualize_sequential(x, prefix_sum, operations)
    print()

    print("=" * 70)
    print("Sequential algorithm completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
