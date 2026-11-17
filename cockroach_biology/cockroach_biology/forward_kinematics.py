import xml.etree.ElementTree as ET
import numpy as np
import os


URDF_FILE = os.path.expanduser(
    "~/ros2_ws/src/cockroach_biology/urdf/cockroach.urdf"
)


def translation(x, y, z):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


def rotation_rpy(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([
        [1, 0, 0, 0],
        [0, cr, -sr, 0],
        [0, sr, cr, 0],
        [0, 0, 0, 1]
    ])

    Ry = np.array([
        [cp, 0, sp, 0],
        [0, 1, 0, 0],
        [-sp, 0, cp, 0],
        [0, 0, 0, 1]
    ])

    Rz = np.array([
        [cy, -sy, 0, 0],
        [sy, cy, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    return Rz @ Ry @ Rx


def rotation_axis(axis, angle):
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c

    T = np.eye(4)

    T[:3, :3] = [
        [x*x*C+c,   x*y*C-z*s, x*z*C+y*s],
        [y*x*C+z*s, y*y*C+c,   y*z*C-x*s],
        [z*x*C-y*s, z*y*C+x*s, z*z*C+c]
    ]

    return T


def get_vector(element, default):
    if element is None:
        return np.array(default, dtype=float)

    return np.array(
        [float(v) for v in element.split()],
        dtype=float
    )


tree = ET.parse(URDF_FILE)
robot = tree.getroot()

joints = {}

for joint in robot.findall("joint"):
    if joint.get("type") != "revolute":
        continue

    name = joint.get("name")

    origin = joint.find("origin")
    axis = joint.find("axis")

    xyz = get_vector(
        origin.get("xyz") if origin is not None else None,
        [0, 0, 0]
    )

    rpy = get_vector(
        origin.get("rpy") if origin is not None else None,
        [0, 0, 0]
    )

    axis_vector = get_vector(
        axis.get("xyz") if axis is not None else None,
        [1, 0, 0]
    )

    joints[name] = {
        "xyz": xyz,
        "rpy": rpy,
        "axis": axis_vector,
        "parent": joint.find("parent").get("link"),
        "child": joint.find("child").get("link")
    }


def get_tarsus_endpoint(link_name):
    link = robot.find(f"link[@name='{link_name}']")

    visual = link.find("visual")
    origin = visual.find("origin")
    geometry = visual.find("geometry")
    box = geometry.find("box")

    size = get_vector(box.get("size"), [0, 0, 0])

    visual_xyz = get_vector(
        origin.get("xyz") if origin is not None else None,
        [0, 0, 0]
    )

    # Endpoint of the box along its local X axis
    endpoint = visual_xyz + np.array([size[0] / 2, 0, 0])

    return endpoint


def calculate_leg_fk(leg_name, angles=None):

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

    for segment in segments:

        joint_name = f"{leg_name}_{segment}_joint"
        joint = joints[joint_name]

        T_origin = translation(*joint["xyz"])
        T_orientation = rotation_rpy(*joint["rpy"])

        angle = angles.get(joint_name, 0.0)

        T_rotation = rotation_axis(
            joint["axis"],
            angle
        )

        T = T @ T_origin @ T_orientation @ T_rotation

    tarsus_link = f"{leg_name}_tarsus"
    endpoint = get_tarsus_endpoint(tarsus_link)

    point = np.array([
        endpoint[0],
        endpoint[1],
        endpoint[2],
        1.0
    ])

    foot_position = T @ point

    return foot_position[:3]


if __name__ == "__main__":

    legs = [
        "left_fore",
        "right_fore",
        "left_middle",
        "right_middle",
        "left_hind",
        "right_hind"
    ]

    print("\nCOCKROACH FORWARD KINEMATICS")
    print("All joint angles are initially zero.")
    print("\nFoot positions relative to base_link:\n")

    for leg in legs:

        position = calculate_leg_fk(leg)

        print(f"{leg.upper()} LEG")
        print(f"X = {position[0] * 1000:.3f} mm")
        print(f"Y = {position[1] * 1000:.3f} mm")
        print(f"Z = {position[2] * 1000:.3f} mm")
        print("-" * 35)
