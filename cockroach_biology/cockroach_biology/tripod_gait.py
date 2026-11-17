import numpy as np
import matplotlib.pyplot as plt


# Initial simulation assumptions
STRIDE_LENGTH = 0.009       # metres
STEP_HEIGHT = 0.003         # metres
GAIT_FREQUENCY = 1.0        # Hz
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
        GAIT_FREQUENCY * t +
        phase_offset / (2 * np.pi)
    ) % 1.0

    if phase < STANCE_RATIO:

        x = (
            STRIDE_LENGTH / 2
            - STRIDE_LENGTH * phase / STANCE_RATIO
        )

        z = 0.0

    else:

        swing_phase = (
            (phase - STANCE_RATIO) /
            (1.0 - STANCE_RATIO)
        )

        x = (
            -STRIDE_LENGTH / 2
            + STRIDE_LENGTH * swing_phase
        )

        z = STEP_HEIGHT * np.sin(
            np.pi * swing_phase
        )

    return x, z


if __name__ == "__main__":

    time = np.linspace(
        0,
        2 / GAIT_FREQUENCY,
        500
    )

    plt.figure()

    for leg in TRIPOD_A + TRIPOD_B:

        if leg in TRIPOD_A:
            offset = 0.0
        else:
            offset = np.pi

        x_values = []
        z_values = []

        for t in time:

            x, z = calculate_foot_trajectory(
                t,
                offset
            )

            x_values.append(x * 1000)
            z_values.append(z * 1000)

        plt.plot(
            time,
            x_values,
            label=f"{leg} X"
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Foot X displacement (mm)")
    plt.title("Alternating Tripod Gait - Foot X Trajectories")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "tripod_gait_x.png",
        dpi=200
    )

    plt.figure()

    for leg in TRIPOD_A + TRIPOD_B:

        offset = 0.0 if leg in TRIPOD_A else np.pi

        z_values = []

        for t in time:

            _, z = calculate_foot_trajectory(
                t,
                offset
            )

            z_values.append(z * 1000)

        plt.plot(
            time,
            z_values,
            label=leg
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Foot Z displacement (mm)")
    plt.title("Alternating Tripod Gait - Foot Lift")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "tripod_gait_z.png",
        dpi=200
    )

    plt.show()
