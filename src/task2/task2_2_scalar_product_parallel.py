import numpy as np
import matplotlib.pyplot as plt
import math


def parallel_scalar_product(A, B, p):
    n = len(A)
    chunk_size = n // p

    partial_sums = []
    operations_phase1 = 0

    print(f"\nPHASE 1: Parallel Computation (p={p} processors)")
    print("=" * 70)
    for i in range(p):
        start = i * chunk_size
        end = start + chunk_size
        partial_sum = sum(A[j] * B[j] for j in range(start, end))
        partial_sums.append(partial_sum)
        operations_phase1 += chunk_size * 2
        print(
            f"Processor {i}: elements [{start}:{end}] → partial sum = {partial_sum:.6f}"
        )

    time_steps_phase1 = chunk_size

    print(f"\nPHASE 2: Sequential Reduction")
    print("=" * 70)
    result = sum(partial_sums)
    operations_phase2 = p - 1
    time_steps_phase2 = p - 1

    for i, ps in enumerate(partial_sums):
        print(f"Partial sum {i}: {ps:.6f}")
    print(f"Final result: {result:.6f}")

    total_time_steps = time_steps_phase1 + time_steps_phase2
    total_operations = operations_phase1 + operations_phase2

    return (
        result,
        total_time_steps,
        total_operations,
        partial_sums,
        time_steps_phase1,
        time_steps_phase2,
    )


def visualize_parallel(A, B, result, partial_sums, p, n):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar(range(p), partial_sums)
    ax1.set_title(f"Partial Sums (p={p} processors)")
    ax1.set_xlabel("Processor")
    ax1.set_ylabel("Partial Sum")
    for i, v in enumerate(partial_sums):
        ax1.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    ax2.text(
        0.5,
        0.6,
        f"Scalar Product Result",
        ha="center",
        fontsize=14,
        transform=ax2.transAxes,
    )
    ax2.text(
        0.5, 0.4, f"{result:.2f}", ha="center", fontsize=20, transform=ax2.transAxes
    )
    ax2.text(
        0.5, 0.2, f"n = {n}, p = {p}", ha="center", fontsize=12, transform=ax2.transAxes
    )
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig("../../plots/task2_2_parallel.png", dpi=150, bbox_inches="tight")
    print("✓ Saved visualization: plots/task2_2_parallel.png")
    plt.close()


def main():
    print("=" * 70)
    print("TASK 2.2: PARALLEL SCALAR PRODUCT")
    print("=" * 70)
    print()

    n = 160
    p = 8

    np.random.seed(42)
    A = np.random.rand(n)
    B = np.random.rand(n)

    print(f"Vector A: {n} elements")
    print(f"Vector B: {n} elements")
    print(f"Processors: {p}")
    print(f"Elements per processor: {n // p}")

    result, time_steps, operations, partial_sums, t1, t2 = parallel_scalar_product(
        A, B, p
    )

    print()
    print("-" * 70)
    print("PARALLEL ALGORITHM ANALYSIS")
    print("-" * 70)
    print(f"Time Steps:        {time_steps} steps total")
    print(f"  Phase 1 (parallel):  {t1} steps")
    print(f"  Phase 2 (reduction): {t2} steps")
    print(f"Total Operations:  {operations} operations")
    print(f"CPUs Required:     {p} CPUs")
    print(f"Time Complexity:   O(n/p + p) = O({n}/{p} + {p}) = O({n // p + p})")
    print()

    verification = np.dot(A, B)
    print(f"Verification (numpy.dot): {verification:.6f}")
    print(
        f"Match: {'✓ CORRECT' if abs(result - verification) < 1e-10 else '✗ INCORRECT'}"
    )
    print()

    print("Generating visualization...")
    visualize_parallel(A, B, result, partial_sums, p, n)
    print()

    print("=" * 70)
    print("Parallel scalar product completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
