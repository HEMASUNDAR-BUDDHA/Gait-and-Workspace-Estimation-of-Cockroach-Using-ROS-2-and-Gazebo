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

q = np.zeros(5)

print("\nFORELEG BACKWARD REACHABILITY SCAN")
print("=" * 55)

for displacement_mm in [-5.0, -4.5, -4.0, -3.5, -3.0,
                        -2.5, -2.0, -1.5, -1.0]:

    target = initial + np.array([
        displacement_mm / 1000.0, 0.0, 0.0
    ])

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

    print(
        f"Backward displacement: {displacement_mm:5.1f} mm | "
        f"Error: {error:.9f} mm | "
        f"{'PASS' if error < 0.001 else 'NOT REACHED'}"
    )
