import numpy as np

from forward_kinematics import calculate_leg_fk
from jacobian_analysis import calculate_jacobian


SEGMENTS = [
    "coxa",
    "trochanter",
    "femur",
    "tibia",
    "tarsus"
]

LEGS = [
    "left_fore",
    "right_fore",
    "left_middle",
    "right_middle",
    "left_hind",
    "right_hind"
]

# Limits copied from urdf/cockroach.urdf.xacro
JOINT_LIMITS = {
    "coxa": (-1.2, 1.2),
    "trochanter": (-1.0, 1.0),
    "femur": (-1.5, 1.5),
    "tibia": (-1.8, 1.8),
    "tarsus": (-1.2, 1.2)
}

MAX_ITERATIONS = 1000
TOLERANCE = 1e-6       # metres (0.001 mm)
DAMPING = 1e-4
MAX_STEP = 0.05        # radians per joint per iteration


def get_joint_names(leg):
    return [
        f"{leg}_{segment}_joint"
        for segment in SEGMENTS
    ]


def enforce_joint_limits(leg, angles):
    limited_angles = {}

    for segment in SEGMENTS:
        joint = f"{leg}_{segment}_joint"
        lower, upper = JOINT_LIMITS[segment]

        limited_angles[joint] = float(
            np.clip(angles[joint], lower, upper)
        )

    return limited_angles


def inverse_kinematics(leg, target_position, initial_angles=None):

    if leg not in LEGS:
        raise ValueError(f"Unknown leg: {leg}")

    target = np.asarray(target_position, dtype=float)

    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("Target must contain exactly 3 finite values.")

    joint_names = get_joint_names(leg)

    if initial_angles is None:
        angles = {joint: 0.0 for joint in joint_names}
    else:
        angles = {
            joint: float(initial_angles.get(joint, 0.0))
            for joint in joint_names
        }

    angles = enforce_joint_limits(leg, angles)
    damping = DAMPING

    for iteration in range(MAX_ITERATIONS):

        current_position = calculate_leg_fk(leg, angles)
        error = target - current_position
        error_norm = np.linalg.norm(error)

        if error_norm < TOLERANCE:
            return angles, True, iteration, error_norm

        J = calculate_jacobian(leg, joint_names, angles)

        accepted = False

        # Retry with stronger damping if an update does not improve.
        for retry in range(8):

            system = J @ J.T + damping**2 * np.eye(3)

            delta = J.T @ np.linalg.solve(system, error)

            largest_step = np.max(np.abs(delta))
            if largest_step > MAX_STEP:
                delta *= MAX_STEP / largest_step

            # Backtracking line search.
            for backtrack in range(8):

                scale = 0.5 ** backtrack

                trial = {
                    joint: angles[joint] + scale * delta[i]
                    for i, joint in enumerate(joint_names)
                }

                trial = enforce_joint_limits(leg, trial)

                trial_position = calculate_leg_fk(leg, trial)
                trial_error = np.linalg.norm(target - trial_position)

                if trial_error < error_norm:
                    angles = trial
                    damping = max(damping * 0.5, 1e-6)
                    accepted = True
                    break

            if accepted:
                break

            damping = min(damping * 10.0, 1e2)

        # No improving feasible update was found.
        if not accepted:
            break

    final_position = calculate_leg_fk(leg, angles)
    final_error = np.linalg.norm(target - final_position)

    return (
        angles,
        final_error < TOLERANCE,
        iteration + 1,
        final_error
    )


if __name__ == "__main__":

    print("\nSIX-LEG LIMIT-AWARE INVERSE KINEMATICS TEST")
    print("=" * 55)

    for leg in LEGS:

        initial_position = calculate_leg_fk(leg)

        # Test a 1 mm displacement in X and Z.
        target = initial_position + np.array([
            0.001,
            0.0,
            0.001
        ])

        angles, success, iterations, error = inverse_kinematics(
            leg,
            target
        )

        calculated_position = calculate_leg_fk(leg, angles)

        print(f"\nLEG: {leg.upper()}")
        print("-" * 45)

        print("Initial position (mm):",
              initial_position * 1000)

        print("Target position (mm):",
              target * 1000)

        print("Calculated position (mm):",
              calculated_position * 1000)

        print("Success:", success)
        print("Iterations:", iterations)
        print("Position error (mm):", error * 1000)

        print("\nJoint angles and limit verification:")

        for segment in SEGMENTS:
            joint = f"{leg}_{segment}_joint"
            lower, upper = JOINT_LIMITS[segment]
            value = angles[joint]

            valid = lower <= value <= upper

            print(
                f"{segment:12s}: {value:+.6f} rad | "
                f"limits [{lower:+.2f}, {upper:+.2f}] | "
                f"{'PASS' if valid else 'FAIL'}"
            )
