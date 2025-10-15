import time
import matplotlib.pyplot as plt
import numpy as np


def sequential_prefix_sum(arr):
    n = len(arr)
    prefix_sum = [0] * n
    operations = 0

    prefix_sum[0] = arr[0]

    for i in range(1, n):
        prefix_sum[i] = prefix_sum[i - 1] + arr[i]
        operations += 1

    return prefix_sum, operations


def visualize_sequential(arr, prefix_sum, operations):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar(range(len(arr)), arr)
    ax1.set_title("Input Array")
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Value")
    for i, v in enumerate(arr):
        ax1.text(i, v + 0.3, str(v), ha="center")

    ax2.bar(range(len(prefix_sum)), prefix_sum)
    ax2.set_title("Sequential Prefix Sum")
    ax2.set_xlabel("Index")
    ax2.set_ylabel("Value")
    for i, v in enumerate(prefix_sum):
        ax2.text(i, v + 0.5, str(v), ha="center")

    plt.tight_layout()
    plt.savefig("../../plots/task1_1_sequential.png", dpi=150, bbox_inches="tight")
    print("✓ Saved visualization: plots/task1_1_sequential.png")
    plt.close()


def main():
    print("=" * 70)
    print("TASK 1.1: SEQUENTIAL PREFIX SUM ALGORITHM")
    print("=" * 70)
    print()

    x = [2, 4, 6, 8, 1, 3, 5, 7]
    print(f"Input Array x: {x}")
    print(f"Array Length n: {len(x)}")
    print()

    start_time = time.perf_counter()
    prefix_sum, operations = sequential_prefix_sum(x)
    end_time = time.perf_counter()

    execution_time = (end_time - start_time) * 1000

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

    expected = [2, 6, 12, 20, 21, 24, 29, 36]
    verification = "✓ CORRECT" if prefix_sum == expected else "✗ INCORRECT"
    print(f"Verification:      {verification}")
    print(f"Expected Result:   {expected}")
    print()

    print("Generating visualization...")
    visualize_sequential(x, prefix_sum, operations)
    print()

    print("=" * 70)
    print("Sequential algorithm completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
