import numpy as np
from scipy.optimize import least_squares

from forward_kinematics import calculate_leg_fk
from inverse_kinematics import get_joint_names, JOINT_LIMITS

LEG = "left_fore"
JOINTS = get_joint_names(LEG)
SEGMENTS = ["coxa", "trochanter", "femur", "tibia", "tarsus"]

initial_position = calculate_leg_fk(LEG)

target = initial_position + np.array([
    0.005,
    0.0,
    0.0
])

lower = np.array([JOINT_LIMITS[s][0] for s in SEGMENTS])
upper = np.array([JOINT_LIMITS[s][1] for s in SEGMENTS])

def objective(q):
    angles = dict(zip(JOINTS, q))
    position = calculate_leg_fk(LEG, angles)
    return (target - position) * 1000.0

initial_guesses = [
    [0, 0, 0, 0, 0],
    [0, 0, 0.5, -0.5, 0],
    [0, 0, -0.5, 0.5, 0],
    [0.3, 0.2, 0.3, -0.3, 0.1],
    [-0.3, -0.2, -0.3, 0.3, -0.1],
    [0.5, 0.5, 0.5, 0.5, 0.5],
    [-0.5, -0.5, -0.5, -0.5, -0.5],
    [0, 0, 1.0, -1.0, 0.5],
    [0, 0, -1.0, 1.0, -0.5]
]

print("\nBOUNDED FORELEG MULTI-START REACHABILITY TEST")
print("=" * 55)
print("Initial position (mm):", initial_position * 1000)
print("Target position (mm): ", target * 1000)
print("Joint limits enforced.")

best_error = float("inf")
best_solution = None

for index, guess in enumerate(initial_guesses):

    guess = np.clip(np.array(guess, dtype=float),
                    lower + 1e-9, upper - 1e-9)

    result = least_squares(
        objective,
        guess,
        bounds=(lower, upper),
        max_nfev=10000,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12
    )

    final_error = np.linalg.norm(objective(result.x))

    print(f"\nINITIAL GUESS {index + 1}")
    print("Final error (mm):", final_error)
    print("Joint angles (rad):", np.round(result.x, 6))
    print("Optimizer success:", result.success)

    if final_error < best_error:
        best_error = final_error
        best_solution = result.x.copy()

print("\n" + "=" * 55)
print("BEST BOUNDED SOLUTION")
print("Minimum position error (mm):", best_error)

for joint, angle in zip(JOINTS, best_solution):
    print(f"{joint}: {angle:+.6f} rad")

print("\nReachability at 0.001 mm tolerance:",
      "PASS" if best_error < 0.001 else "NOT CONFIRMED")
