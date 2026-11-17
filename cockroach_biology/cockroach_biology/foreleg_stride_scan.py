import numpy as np
from scipy.optimize import least_squares

from forward_kinematics import calculate_leg_fk
from inverse_kinematics import get_joint_names, JOINT_LIMITS

LEG = "left_fore"
JOINTS = get_joint_names(LEG)
SEGMENTS = ["coxa", "trochanter", "femur", "tibia", "tarsus"]

initial = calculate_leg_fk(LEG)

lower = np.array([JOINT_LIMITS[s][0] for s in SEGMENTS])
upper = np.array([JOINT_LIMITS[s][1] for s in SEGMENTS])

# Start from the best solution found by the global search.
q = np.array([
    -0.42650890,
     0.00000005,
     0.09919030,
    -0.65000002,
     0.90000004
])

print("\nFORELEG FORWARD STRIDE REACHABILITY SCAN")
print("=" * 55)
print("Tolerance: 0.001 mm")
print("All existing joint limits retained.\n")

for stride_mm in [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0]:

    target = initial + np.array([stride_mm / 1000.0, 0.0, 0.0])

    def objective(angles):
        position = calculate_leg_fk(
            LEG, dict(zip(JOINTS, angles))
        )
        return (target - position) * 1000.0

    result = least_squares(
        objective,
        np.clip(q, lower + 1e-9, upper - 1e-9),
        bounds=(lower, upper),
        max_nfev=10000,
        ftol=1e-13,
        xtol=1e-13,
        gtol=1e-13
    )

    q = result.x
    error = np.linalg.norm(objective(q))
    status = "PASS" if error < 0.001 else "NOT REACHED"

    print(
        f"Forward displacement: {stride_mm:4.1f} mm | "
        f"Error: {error:.9f} mm | {status}"
    )

    if status == "PASS":
        print("Joint angles:", np.round(q, 6))

