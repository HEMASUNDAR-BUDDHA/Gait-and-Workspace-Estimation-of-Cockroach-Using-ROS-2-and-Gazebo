import numpy as np
from forward_kinematics import calculate_leg_fk

LEG = "left_fore"

JOINTS = [
    "left_fore_coxa_joint",
    "left_fore_trochanter_joint",
    "left_fore_femur_joint",
    "left_fore_tibia_joint",
    "left_fore_tarsus_joint"
]


def calculate_jacobian(leg, joint_names, angles=None):

    if angles is None:
        angles = {}

    J = np.zeros((3, len(joint_names)))

    h = 1e-5

    for i, joint in enumerate(joint_names):

        q_plus = angles.copy()
        q_minus = angles.copy()

        q_plus[joint] = q_plus.get(joint, 0.0) + h
        q_minus[joint] = q_minus.get(joint, 0.0) - h

        p_plus = calculate_leg_fk(leg, q_plus)
        p_minus = calculate_leg_fk(leg, q_minus)

        J[:, i] = (p_plus - p_minus) / (2 * h)

    return J


if __name__ == "__main__":

    J = calculate_jacobian(LEG, JOINTS)

    print("\nLEFT FORELEG JACOBIAN MATRIX")
    print("Units: mm/rad")
    print("---------------------------------------")

    print(np.round(J * 1000, 4))

    print("\nJOINT ORDER:")
    for i, joint in enumerate(JOINTS, 1):
        print(f"{i}. {joint}")

    print("\nSINGULAR VALUES:")
    print(np.round(np.linalg.svd(J, compute_uv=False), 6))

    print("\nJACOBIAN RANK:")
    print(np.linalg.matrix_rank(J))

    print("---------------------------------------")
