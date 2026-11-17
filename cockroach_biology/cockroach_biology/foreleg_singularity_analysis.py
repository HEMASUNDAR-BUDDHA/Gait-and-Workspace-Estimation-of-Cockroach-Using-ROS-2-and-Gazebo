import numpy as np

from forward_kinematics import calculate_leg_fk
from jacobian_analysis import calculate_jacobian
from inverse_kinematics import get_joint_names


leg = "left_fore"

joint_names = get_joint_names(leg)

zero_angles = {
    joint: 0.0 for joint in joint_names
}

failed_angles = {
    "left_fore_coxa_joint": -0.426509,
    "left_fore_trochanter_joint": 0.0,
    "left_fore_femur_joint": 0.099190,
    "left_fore_tibia_joint": -0.650000,
    "left_fore_tarsus_joint": 0.900000
}


def analyze_configuration(label, angles):

    print("\n" + "=" * 55)
    print(label)
    print("=" * 55)

    position = calculate_leg_fk(leg, angles)

    J = calculate_jacobian(
        leg,
        joint_names,
        angles
    )

    # Singular Value Decomposition
    U, singular_values, Vt = np.linalg.svd(
        J,
        full_matrices=False
    )

    rank = np.linalg.matrix_rank(J)

    smallest = singular_values[-1]

    if smallest > 1e-12:
        condition = singular_values[0] / smallest
    else:
        condition = float("inf")

    print("\nFoot position (mm):")
    print(position * 1000)

    print("\nJacobian (mm/rad):")
    print(J * 1000)

    print("\nSingular values (mm/rad):")
    print(singular_values * 1000)

    print("\nJacobian numerical rank:", rank)

    print("Condition number:", condition)

    print("\nNear-null joint movement direction:")

    null_direction = Vt[-1]

    for joint, value in zip(joint_names, null_direction):
        print(f"{joint}: {value:.8f}")

    print("\nJoint sensitivity (mm/rad):")

    for i, joint in enumerate(joint_names):

        sensitivity = np.linalg.norm(J[:, i]) * 1000

        print(f"{joint}: {sensitivity:.6f}")


analyze_configuration(
    "ZERO-ANGLE CONFIGURATION",
    zero_angles
)

analyze_configuration(
    "FAILED IK CONFIGURATION",
    failed_angles
)
