#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
import numpy as np

class VisualizerNode(Node):
    def __init__(self):
        super().__init__('visualizer_node')
        
        # Parameters
        self.declare_parameter('num_bots', 3)
        self.num_bots = self.get_parameter('num_bots').get_parameter_value().integer_value
        
        # Store trajectory histories
        self.positions = {b: [] for b in range(1, self.num_bots+1)}
        self.centroid_traj = []

        # Subscribers
        for i in range(1, self.num_bots+1):
            self.create_subscription(PoseStamped, f'/bot{i}/pose', self.make_pose_callback(i), 10)

        # Timer for plotting (1 Hz)
        self.create_timer(0.1, self.timer_callback)
        self.get_logger().info(f"Visualizer node started for {self.num_bots} bots")

        # Initialize plot
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Swarm Trajectories")
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.axis("equal")
        self.ax.grid(True)

    def make_pose_callback(self, bot_id):
        def pose_callback(msg: PoseStamped):
            pose = np.array([msg.pose.position.x, msg.pose.position.y])
            self.positions[bot_id].append(pose)
        return pose_callback

    def timer_callback(self):
        # Skip if no data yet
        if any(len(pose) == 0 for pose in self.positions.values()):
            return

        # Compute centroid if all have at least one pose
        latest_positions = [pose[-1] for pose in self.positions.values()]
        centroid = np.mean(latest_positions, axis=0)
        self.centroid_traj.append(centroid)

        # Compute reference point on circular trajectory
        t = self.get_clock().now().to_msg().sec
        ref_x, ref_y = 10 * np.cos(t), 10 * np.sin(t)

        # Clear plot
        self.ax.clear()
        self.ax.set_title("Swarm Trajectories with Centroid & Reference")
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.axis("equal")
        self.ax.grid(True)

        # Plot reference circle (static) and current ref point
        theta = np.linspace(0, 2*np.pi, 200)
        self.ax.plot(10*np.cos(theta), 10*np.sin(theta), 'r--', label="Reference Circle")
        self.ax.plot(ref_x, ref_y, 'ro', label="Current Reference")

        # Plot each bot trajectory
        for bot_id, traj in self.positions.items():
            traj = np.array(traj)
            self.ax.plot(traj[:,0], traj[:,1], label=f'Bot {bot_id}')
            self.ax.plot(traj[-1,0], traj[-1,1], 'o')  # current position

        # Plot centroid trajectory
        c_traj = np.array(self.centroid_traj)
        self.ax.plot(c_traj[:,0], c_traj[:,1], 'k-', linewidth=2, label="Centroid Path")
        self.ax.plot(c_traj[-1,0], c_traj[-1,1], 'ks', label="Current Centroid")

        # Legend & update
        self.ax.legend(loc="upper right")
        plt.pause(0.001)


def main(args=None):
    rclpy.init(args=args)
    node = VisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        plt.show()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()