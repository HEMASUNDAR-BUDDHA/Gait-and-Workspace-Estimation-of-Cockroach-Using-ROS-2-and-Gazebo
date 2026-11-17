import numpy as np
import matplotlib.pyplot as plt
import csv
import os

from forward_kinematics import calculate_leg_fk
from inverse_kinematics import inverse_kinematics, LEGS, SEGMENTS


# Gait parameters
STRIDE_LENGTH = 0.009
STEP_HEIGHT = 0.003
GAIT_FREQUENCY = 1.0
STANCE_RATIO = 0.65

TRIPOD_A = [
    "left_fore",
    "right_middle",
    "left_hind"
]

TRIPOD_B = [
    "right_fore",
    "left_middle",
    "right_hind"
]


def calculate_foot_trajectory(t, phase_offset):

    phase = (
        GAIT_FREQUENCY * t
        + phase_offset / (2 * np.pi)
    ) % 1.0

    if phase < STANCE_RATIO:

        x = (
            STRIDE_LENGTH / 2
            - STRIDE_LENGTH * phase / STANCE_RATIO
        )

        z = 0.0

    else:

        swing_phase = (
            phase - STANCE_RATIO
        ) / (1.0 - STANCE_RATIO)

        x = (
            -STRIDE_LENGTH / 2
            + STRIDE_LENGTH * swing_phase
        )

        z = STEP_HEIGHT * np.sin(np.pi * swing_phase)

    return x, z


def get_phase_offset(leg):

    if leg in TRIPOD_A:
        return 0.0

    return np.pi


if __name__ == "__main__":

    time_values = np.linspace(0, 2.0, 101)

    results = {}
    joint_angle_history = {leg: [] for leg in LEGS}

    print("\nTRIPOD GAIT + INVERSE KINEMATICS")
    print("=" * 50)

    for leg in LEGS:

        phase_offset = get_phase_offset(leg)

        initial_position = calculate_leg_fk(leg)

        joint_angles = None

        x_values = []
        z_values = []
        error_values = []

        success_count = 0
        failed_points = []

        for t in time_values:

            dx, dz = calculate_foot_trajectory(
                t,
                phase_offset
            )

            target = initial_position + np.array([
                dx,
                0.0,
                dz
            ])

            angles, success, iterations, error = (
                inverse_kinematics(
                    leg,
                    target,
                    joint_angles
                )
            )

            joint_angles = angles
            joint_angle_history[leg].append([
                angles[f'{leg}_{segment}_joint']
                for segment in SEGMENTS
            ])

            calculated_position = calculate_leg_fk(
                leg,
                joint_angles
            )

            x_values.append(
                calculated_position[0] * 1000
            )

            z_values.append(
                calculated_position[2] * 1000
            )

            error_values.append(error * 1000)

            if success:

                success_count += 1

            else:

                failed_points.append({
                    "time": t,
                    "target": target.copy(),
                    "calculated": calculated_position.copy(),
                    "iterations": iterations,
                    "error_mm": error * 1000,
                    "angles": angles.copy()
                })

        results[leg] = {
            "x": x_values,
            "z": z_values,
            "error": error_values,
            "success": success_count
        }

        print(f"\n{leg.upper()}")
        print("-" * 40)

        print(
            "Successful points:",
            success_count,
            "/",
            len(time_values)
        )

        print(
            "Maximum IK error (mm):",
            max(error_values)
        )

        if failed_points:

            print("\nFAILED TRAJECTORY POINTS:")

            for point in failed_points:

                print(
                    "\nTime:",
                    round(point["time"], 4),
                    "s"
                )

                print(
                    "Iterations:",
                    point["iterations"]
                )

                print(
                    "Error:",
                    point["error_mm"],
                    "mm"
                )

                print(
                    "Target (mm):",
                    point["target"] * 1000
                )

                print(
                    "Calculated (mm):",
                    point["calculated"] * 1000
                )

                print("Joint angles (rad):")

                for joint, angle in point["angles"].items():
                    print(f"  {joint}: {angle:.6f}")


    # Export 30-joint trajectory CSV
    os.makedirs("results", exist_ok=True)

    joint_names = [
        f"{leg}_{segment}_joint"
        for leg in LEGS
        for segment in SEGMENTS
    ]

    csv_path = "results/tripod_gait_joint_trajectories.csv"

    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time_s"] + joint_names)

        for i, t in enumerate(time_values):
            row = [float(t)]

            for leg in LEGS:
                row.extend(joint_angle_history[leg][i])

            writer.writerow(row)

    print("\\nJOINT TRAJECTORY CSV EXPORTED")
    print("File:", csv_path)
    print("Time samples:", len(time_values))
    print("Joint columns:", len(joint_names))

    # Plot X trajectories
    plt.figure(figsize=(11, 6))

    for leg in LEGS:
        plt.plot(
            time_values,
            results[leg]["x"],
            label=leg
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Foot X position (mm)")
    plt.title("Six-Leg Tripod Gait: Foot X Position")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("gait_ik_x.png", dpi=200)

    # Plot Z trajectories
    plt.figure(figsize=(11, 6))

    for leg in LEGS:
        plt.plot(
            time_values,
            results[leg]["z"],
            label=leg
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Foot Z position (mm)")
    plt.title("Six-Leg Tripod Gait: Foot Z Position")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("gait_ik_z.png", dpi=200)

    # Plot IK error
    plt.figure(figsize=(11, 6))

    for leg in LEGS:
        plt.plot(
            time_values,
            results[leg]["error"],
            label=leg
        )

    plt.xlabel("Time (s)")
    plt.ylabel("IK position error (mm)")
    plt.title("Inverse Kinematics Error During Tripod Gait")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("gait_ik_error.png", dpi=200)

    plt.show()
