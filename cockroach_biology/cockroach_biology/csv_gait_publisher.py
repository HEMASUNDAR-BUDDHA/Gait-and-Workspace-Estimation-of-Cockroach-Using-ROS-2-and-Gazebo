import csv
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class CockroachGaitPublisher(Node):

    def __init__(self):
        super().__init__("cockroach_gait_publisher")

        csv_path = (
            Path(__file__).resolve().parent.parent
            / "results"
            / "tripod_gait_joint_trajectories.csv"
        )

        with open(csv_path, newline="") as file:
            reader = csv.reader(file)
            header = next(reader)
            self.joint_names = header[1:]
            self.trajectory = [
                [float(value) for value in row[1:]]
                for row in reader
            ]

        self.publisher = self.create_publisher(
            JointState,
            "/joint_states",
            10
        )

        self.index = 0
        self.timer = self.create_timer(0.02, self.publish_frame)

        self.get_logger().info(
            f"Loaded {len(self.trajectory)} frames "
            f"and {len(self.joint_names)} joints."
        )

        self.get_logger().info("Publishing tripod gait on /joint_states")

    def publish_frame(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.trajectory[self.index]

        self.publisher.publish(msg)

        self.index = (self.index + 1) % len(self.trajectory)


def main(args=None):
    rclpy.init(args=args)
    node = CockroachGaitPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
