import numpy as np

from forward_kinematics import (
    joints,
    translation,
    rotation_rpy,
    rotation_axis,
    get_tarsus_endpoint
)


def calculate_analytic_jacobian(leg_name, angles=None):

    if angles is None:
        angles = {}

    segments = [
        "coxa",
        "trochanter",
        "femur",
        "tibia",
        "tarsus"
    ]

    T = np.eye(4)

    joint_origins = []
    joint_axes = []

    # Calculate each joint's position and axis
    for segment in segments:

        joint_name = f"{leg_name}_{segment}_joint"
        joint = joints[joint_name]

        T_origin = translation(*joint["xyz"])
        T_orientation = rotation_rpy(*joint["rpy"])

        T_joint = T @ T_origin @ T_orientation

        # Joint origin in base_link coordinates
        origin_world = T_joint[:3, 3]

        # Joint axis in base_link coordinates
        axis_world = (
            T_joint[:3, :3] @ joint["axis"]
        )

        joint_origins.append(origin_world)
        joint_axes.append(axis_world)

        angle = angles.get(joint_name, 0.0)

        T = T_joint @ rotation_axis(
            joint["axis"],
            angle
        )

    # Calculate foot endpoint
    tarsus_link = f"{leg_name}_tarsus"
    endpoint = get_tarsus_endpoint(tarsus_link)

    local_point = np.array([
        endpoint[0],
        endpoint[1],
        endpoint[2],
        1.0
    ])

    foot = (T @ local_point)[:3]

    # Construct analytical Jacobian
    J = np.zeros((3, 5))

    for i in range(5):

        J[:, i] = np.cross(
            joint_axes[i],
            foot - joint_origins[i]
        )

    return J


if __name__ == "__main__":

    leg = "left_fore"

    J = calculate_analytic_jacobian(leg)

    print("\nANALYTICAL JACOBIAN - LEFT FORELEG")
    print("---------------------------------------")

    print("Units: mm/rad")

    print(np.round(J * 1000, 4))

    print("\nJacobian rank:")
    print(np.linalg.matrix_rank(J))

    print("\nSingular values (mm/rad):")

    print(
        np.round(
            np.linalg.svd(J, compute_uv=False) * 1000,
            4
        )
    )

    print("---------------------------------------")
