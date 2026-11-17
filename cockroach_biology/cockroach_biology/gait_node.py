import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class GaitNode(Node):

    def __init__(self):
        super().__init__('gait_node')

        # All 30 revolute joints from the cockroach URDF
        self.joint_names = [
            'left_fore_coxa_joint',
            'left_fore_trochanter_joint',
            'left_fore_femur_joint',
            'left_fore_tibia_joint',
            'left_fore_tarsus_joint',

            'right_fore_coxa_joint',
            'right_fore_trochanter_joint',
            'right_fore_femur_joint',
            'right_fore_tibia_joint',
            'right_fore_tarsus_joint',

            'left_middle_coxa_joint',
            'left_middle_trochanter_joint',
            'left_middle_femur_joint',
            'left_middle_tibia_joint',
            'left_middle_tarsus_joint',

            'right_middle_coxa_joint',
            'right_middle_trochanter_joint',
            'right_middle_femur_joint',
            'right_middle_tibia_joint',
            'right_middle_tarsus_joint',

            'left_hind_coxa_joint',
            'left_hind_trochanter_joint',
            'left_hind_femur_joint',
            'left_hind_tibia_joint',
            'left_hind_tarsus_joint',

            'right_hind_coxa_joint',
            'right_hind_trochanter_joint',
            'right_hind_femur_joint',
            'right_hind_tibia_joint',
            'right_hind_tarsus_joint'
        ]

        # Initialize all joints to zero radians
        self.joint_positions = [0.0] * len(self.joint_names)

        # Publisher for RViz / robot_state_publisher
        self.publisher = self.create_publisher(
            JointState,
            '/joint_states',
            10
        )

        # Publish joint positions at 20 Hz
        self.timer = self.create_timer(0.05, self.publish_joint_states)

        self.get_logger().info('Cockroach Gait Node initialized')
        self.get_logger().info(
            f'Initialized {len(self.joint_names)} revolute joints'
        )

    def publish_joint_states(self):

        msg = JointState()

        msg.header.stamp = self.get_clock().now().to_msg()

        msg.name = self.joint_names
        msg.position = self.joint_positions

        self.publisher.publish(msg)


def main(args=None):

    rclpy.init(args=args)

    node = GaitNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
