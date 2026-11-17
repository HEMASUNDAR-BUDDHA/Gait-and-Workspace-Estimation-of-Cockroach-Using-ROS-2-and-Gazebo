import numpy as np
from scipy.optimize import differential_evolution, least_squares

from forward_kinematics import calculate_leg_fk
from inverse_kinematics import get_joint_names, JOINT_LIMITS

LEG = "left_fore"
JOINTS = get_joint_names(LEG)
SEGMENTS = ["coxa", "trochanter", "femur", "tibia", "tarsus"]

initial_position = calculate_leg_fk(LEG)

target = initial_position + np.array([0.005, 0.0, 0.0])

bounds = [JOINT_LIMITS[s] for s in SEGMENTS]

def objective(q):
    angles = dict(zip(JOINTS, q))
    position = calculate_leg_fk(LEG, angles)
    return (target - position) * 1000.0

def cost(q):
    error = objective(q)
    return float(error @ error)

print("\nGLOBAL FORELEG REACHABILITY SEARCH")
print("=" * 55)
print("Target (mm):", target * 1000)
print("Searching complete joint-limit ranges...")

global_result = differential_evolution(
    cost,
    bounds,
    maxiter=500,
    popsize=15,
    tol=1e-10,
    atol=1e-12,
    seed=42,
    polish=True,
    updating="immediate"
)

refined = least_squares(
    objective,
    global_result.x,
    bounds=(
        np.array([b[0] for b in bounds]),
        np.array([b[1] for b in bounds])
    ),
    max_nfev=20000,
    ftol=1e-14,
    xtol=1e-14,
    gtol=1e-14
)

error = np.linalg.norm(objective(refined.x))

print("\nGLOBAL SEARCH RESULT")
print("Optimizer success:", global_result.success)
print("Refined position error (mm):", error)

print("\nJoint angles:")
for joint, angle in zip(JOINTS, refined.x):
    print(f"{joint}: {angle:+.8f} rad")

print("\nReachability at 0.001 mm tolerance:",
      "PASS" if error < 0.001 else "NOT CONFIRMED")
