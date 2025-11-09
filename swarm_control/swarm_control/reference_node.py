#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from swarm_control_msgs.msg import Info, RBroadcast
import numpy as np

class ReferenceNode(Node):
    def __init__(self):
        super().__init__('reference_node')

        # Params
        self.declare_parameter('num_bots', 3)
        self.declare_parameter('frequency', 2.0)

        self.num_bots = self.get_parameter('num_bots').get_parameter_value().integer_value
        self.freq = self.get_parameter('frequency').get_parameter_value().double_value
        self.leaders = set()

        # Publishers
        self.ref_pub = self.create_publisher(PointStamped, '/reference', 10)
        self.broadcast_pub = self.create_publisher(RBroadcast, '/r_broadcast', 10)

        # Subscribers
        for i in range(1, self.num_bots+1):
            self.create_subscription(Info, f'/bot{i}/info', self.info_callback, 10)

        # Timer
        self.create_timer(1.0 / self.freq, self.timer_callback)
        self.get_logger().info("Reference node started")

    def info_callback(self, msg: Info):
        if msg.role == 'leader' and msg.is_active:
            self.leaders.add(msg.id)
        else:
            self.leaders.discard(msg.id)

    def timer_callback(self):
        # Generate r(t)
        now = self.get_clock().now().to_msg()
        t = now.sec
        r_vec = np.array([10*np.cos(t), 10*np.sin(t)])

        # Publish /reference
        ref_msg = PointStamped()
        ref_msg.header.stamp = self.get_clock().now().to_msg()
        ref_msg.header.frame_id = 'world'
        ref_msg.point.x, ref_msg.point.y = r_vec
        self.ref_pub.publish(ref_msg)

        # If any leaders exist, broadcast reference
        for leader_id in self.leaders:
            bmsg = RBroadcast()
            bmsg.id = leader_id
            bmsg.stamp = self.get_clock().now().to_msg()
            bmsg.point.x = r_vec[0]
            bmsg.point.y = r_vec[1]
            self.broadcast_pub.publish(bmsg)

        self.get_logger().debug(f"Leaders={list(self.leaders)}, r={r_vec}")

def main(args=None):
    rclpy.init(args=args)
    node = ReferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()