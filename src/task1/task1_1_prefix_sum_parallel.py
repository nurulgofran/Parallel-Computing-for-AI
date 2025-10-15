import time
import matplotlib.pyplot as plt
import numpy as np
import math


def parallel_prefix_sum_simulation(arr):
    n = len(arr)

    if n & (n - 1) != 0:
        raise ValueError("Array length must be a power of 2")

    result = arr.copy()

    time_steps = 0
    total_operations = 0
    step_details = []

    print("\n" + "=" * 70)
    print("UP-SWEEP PHASE (Building partial sums tree)")
    print("=" * 70)

    d_levels = int(math.log2(n))

    for d in range(d_levels):
        stride = 2 ** (d + 1)
        operations_this_step = 0
        modified_indices = []

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

    result[n - 1] = 0

    print("\n" + "=" * 70)
    print("DOWN-SWEEP PHASE (Computing prefix sums)")
    print("=" * 70)

    for d in range(d_levels - 1, -1, -1):
        stride = 2 ** (d + 1)
        operations_this_step = 0
        modified_indices = []

        for k in range(0, n, stride):
            left_idx = k + 2**d - 1
            right_idx = k + 2 ** (d + 1) - 1

            temp = result[left_idx]
            result[left_idx] = result[right_idx]
            result[right_idx] = result[right_idx] + temp
            operations_this_step += 2
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
    log_n = int(math.log2(n))

    up_sweep_ops = n - 1

    down_sweep_ops = n - 1

    total_ops = up_sweep_ops + down_sweep_ops
    time_steps = 2 * log_n

    max_cpus = n // 2

    return {
        "time_steps": time_steps,
        "total_operations": total_ops,
        "max_cpus": max_cpus,
        "up_sweep_ops": up_sweep_ops,
        "down_sweep_ops": down_sweep_ops,
    }


def visualize_parallel(arr, prefix_sum, metrics, step_details):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar(range(len(arr)), arr)
    ax1.set_title("Input Array")
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Value")
    for i, v in enumerate(arr):
        ax1.text(i, v + 0.3, str(v), ha="center")

    ax2.bar(range(len(prefix_sum)), prefix_sum)
    ax2.set_title("Parallel Prefix Sum")
    ax2.set_xlabel("Index")
    ax2.set_ylabel("Value")
    for i, v in enumerate(prefix_sum):
        ax2.text(i, v + 0.5, str(v), ha="center")

    plt.tight_layout()
    plt.savefig("../../plots/task1_1_parallel.png", dpi=150, bbox_inches="tight")
    print("✓ Saved visualization: plots/task1_1_parallel.png")
    plt.close()


def main():
    print("=" * 70)
    print("TASK 1.1: PARALLEL PREFIX SUM ALGORITHM (BLELLOCH)")
    print("=" * 70)
    print()

    x = [2, 4, 6, 8, 1, 3, 5, 7]
    print(f"Input Array x: {x}")
    print(f"Array Length n: {len(x)} (power of 2: ✓)")
    print()

    start_time = time.perf_counter()
    prefix_sum, time_steps, operations, step_details = parallel_prefix_sum_simulation(x)
    end_time = time.perf_counter()

    execution_time = (end_time - start_time) * 1000

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Prefix Sum Result: {prefix_sum}")
    print()

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

    expected = [0, 2, 6, 12, 20, 21, 24, 29]

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

    print("Generating visualization...")
    visualize_parallel(x, inclusive_prefix, metrics, step_details)
    print()

    print("=" * 70)
    print("Parallel algorithm completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
