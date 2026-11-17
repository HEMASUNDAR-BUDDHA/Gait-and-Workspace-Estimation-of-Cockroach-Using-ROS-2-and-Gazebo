import numpy as np
import matplotlib.pyplot as plt

from forward_kinematics import calculate_leg_fk


joint_names = [
    "left_fore_coxa_joint",
    "left_fore_trochanter_joint",
    "left_fore_femur_joint",
    "left_fore_tibia_joint",
    "left_fore_tarsus_joint"
]

angles = np.linspace(-0.5, 0.5, 51)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

for joint in joint_names:

    positions = []

    for angle in angles:

        point = calculate_leg_fk(
            "left_fore",
            {joint: angle}
        )

        positions.append(point * 1000)

    positions = np.array(positions)

    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        label=joint.replace("_joint", "")
    )

ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")

ax.set_title("Cockroach Left Foreleg - FK Angle Analysis")

ax.legend()
plt.tight_layout()

plt.savefig("fk_angle_analysis.png", dpi=200)
plt.show()
