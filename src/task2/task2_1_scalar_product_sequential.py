import numpy as np
import matplotlib.pyplot as plt


def sequential_scalar_product(A, B):
    n = len(A)
    result = 0
    operations = 0

    for i in range(n):
        result += A[i] * B[i]
        operations += 2

    time_steps = n

    return result, time_steps, operations


def visualize_sequential(A, B, result, n):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    indices = np.arange(min(20, n))
    ax1.plot(indices, A[: len(indices)], "o-", label="Vector A")
    ax1.plot(indices, B[: len(indices)], "s-", label="Vector B")
    ax1.set_title(f"Input Vectors (showing first {len(indices)} elements)")
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Value")
    ax1.legend()

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
        0.5, 0.2, f"n = {n} elements", ha="center", fontsize=12, transform=ax2.transAxes
    )
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig("../../plots/task2_1_sequential.png", dpi=150, bbox_inches="tight")
    print("✓ Saved visualization: plots/task2_1_sequential.png")
    plt.close()


def main():
    print("=" * 70)
    print("TASK 2.1: SEQUENTIAL SCALAR PRODUCT")
    print("=" * 70)
    print()

    n = 160
    np.random.seed(42)
    A = np.random.rand(n)
    B = np.random.rand(n)

    print(f"Vector A: {n} elements")
    print(f"Vector B: {n} elements")
    print(f"First 10 elements of A: {A[:10]}")
    print(f"First 10 elements of B: {B[:10]}")
    print()

    result, time_steps, operations = sequential_scalar_product(A, B)

    print(f"Scalar Product Result: {result:.6f}")
    print()

    print("-" * 70)
    print("SEQUENTIAL ALGORITHM ANALYSIS")
    print("-" * 70)
    print(f"Time Steps:        {time_steps} steps (one multiply-add per step)")
    print(
        f"Total Operations:  {operations} operations ({n} multiplications + {n} additions)"
    )
    print(f"CPUs Required:     1 CPU (sequential execution)")
    print(f"Time Complexity:   O(n) = O({n})")
    print()

    verification = np.dot(A, B)
    print(f"Verification (numpy.dot): {verification:.6f}")
    print(
        f"Match: {'✓ CORRECT' if abs(result - verification) < 1e-10 else '✗ INCORRECT'}"
    )
    print()

    print("Generating visualization...")
    visualize_sequential(A, B, result, n)
    print()

    print("=" * 70)
    print("Sequential scalar product completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
