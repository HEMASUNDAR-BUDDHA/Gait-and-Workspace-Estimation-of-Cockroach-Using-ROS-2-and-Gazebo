import numpy as np

from analytic_jacobian import calculate_analytic_jacobian


legs = [
    "left_fore",
    "right_fore",
    "left_middle",
    "right_middle",
    "left_hind",
    "right_hind"
]

print("\nCOCKROACH - SIX LEG JACOBIAN ANALYSIS")
print("=" * 55)

for leg in legs:

    J = calculate_analytic_jacobian(leg)

    singular_values = np.linalg.svd(
        J,
        compute_uv=False
    )

    rank = np.linalg.matrix_rank(J)

    condition = (
        singular_values[0] / singular_values[-1]
        if singular_values[-1] > 1e-12
        else float("inf")
    )

    print(f"\nLEG: {leg.upper()}")

    print("\nJacobian (mm/rad):")
    print(np.round(J * 1000, 4))

    print("\nRank:", rank)

    print("Singular values (mm/rad):")
    print(np.round(singular_values * 1000, 4))

    print("Condition number:", round(condition, 4))

    print("-" * 55)

print("\nSIX LEG ANALYSIS COMPLETED")
