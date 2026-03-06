#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PointStamped, Twist, PoseStamped
from swarm_control_msgs.msg import RBroadcast
import numpy as np
import casadi as ca


class DMPCController(Node):
    def __init__(self):
        super().__init__('dmpc_controller')
        
        # ========== Parameters ==========
        self.declare_parameter('bot_id', 'bot1')
        self.declare_parameter('num_bots', 3)
        self.declare_parameter('mpc_horizon', 10)
        self.declare_parameter('max_linear_vel', 1.5)
        self.declare_parameter('max_angular_vel', 1.0)
        self.declare_parameter('max_linear_acceleration', 1.0)
        self.declare_parameter('max_angular_acceleration', 1.0)
        self.declare_parameter('control_frequency', 10.0)
        
        # Get parameters
        self.bot_id = self.get_parameter('bot_id').get_parameter_value().string_value
        self.num_bots = self.get_parameter('num_bots').get_parameter_value().integer_value
        self.N = self.get_parameter('mpc_horizon').get_parameter_value().integer_value
        self.max_linear_vel = self.get_parameter('max_linear_vel').get_parameter_value().double_value
        self.max_angular_vel = self.get_parameter('max_angular_vel').get_parameter_value().double_value
        self.max_linear_acc = self.get_parameter('max_linear_acceleration').get_parameter_value().double_value
        self.max_angular_acc = self.get_parameter('max_angular_acceleration').get_parameter_value().double_value
        self.control_freq = self.get_parameter('control_frequency').get_parameter_value().double_value

        # ========= DMPC State Variables =========
        self.E_i = 0.2                              # Max allowed deviation from previous plan (Eq 25)
        self.D_safe = 0.4                           # Minimum safe distance between robots
        self.own_prev_traj = np.zeros((2, self.N))  # ˜x_i
        self.neighbor_trajectories = {}             # ˜x_j

        # Determine neighbors
        self.agent_list = [f'bot{i+1}' for i in range(self.num_bots)]
        if self.bot_id in self.agent_list:
            self.agent_list.remove(self.bot_id)

        # ========= Internal State =========
        self.current_pose = None    # [x, y, yaw]
        self.current_delta = None   # [delta_x, delta_y]
        self.r_vec = None           # [r_x, r_y]

        # ========= ROS Setup =========
        self.cmd_pub = self.create_publisher(Twist, f'/{self.bot_id}/cmd_vel', 10)
        self.traj_pub = self.create_publisher(Float64MultiArray, f'/{self.bot_id}/planned_trajectory', 10)
        
        self.create_subscription(PoseStamped, f'/{self.bot_id}/pose', self.pose_callback, 10)
        self.create_subscription(PointStamped, f'/{self.bot_id}/delta', self.delta_callback, 10)
        self.create_subscription(RBroadcast, '/r_broadcast', self.reference_callback, 10)

        # Subscribe to neighbors' planned trajectories
        for bot in self.agent_list:
            self.create_subscription(
                Float64MultiArray, f'/{bot}/planned_trajectory', 
                self.make_traj_callback(bot), 10
            )
            self.neighbor_trajectories[bot] = np.full((2, self.N), 999.0) # Init far away

        self.timer = self.create_timer(1.0 / self.control_freq, self.control_loop)

        # ========= MPC Initialization =========
        self.opti = ca.Opti()
        self.setup_mpc()
        self.get_logger().info(f"{self.bot_id}: MPC Controller initialized")

    # =======================================
    # ROS Callbacks
    # =======================================
    def pose_callback(self, msg: PoseStamped):
        # Extract planar pose
        x, y = msg.pose.position.x, msg.pose.position.y
        qz, qw = msg.pose.orientation.z, msg.pose.orientation.w
        yaw = 2 * np.arctan2(qz, qw)
        self.current_pose = np.array([x, y, yaw])

    def delta_callback(self, msg: PointStamped):
        self.current_delta = np.array([msg.point.x, msg.point.y])

    def reference_callback(self, msg: RBroadcast):
        self.r_vec = np.array([msg.point.x, msg.point.y])

    def make_traj_callback(self, bot_name):
        def callback(msg):
            # Reshape flat array [x0,y0, x1,y1...] back to 2xN
            traj_array = np.array(msg.data).reshape((self.N, 2)).T
            self.neighbor_trajectories[bot_name] = traj_array
        return callback

    # =======================================
    # MPC Formulation
    # =======================================
    def setup_mpc(self):
        """Define MPC optimization problem using CasADi."""
        N = self.N
        dt = 1.0 / self.control_freq

        # State [x, y, theta]
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        states = ca.vertcat(x, y, theta)
        n_states = states.numel()

        # Control [v, omega]
        v = ca.SX.sym('v')
        omega = ca.SX.sym('omega')
        controls = ca.vertcat(v, omega)
        n_controls = controls.numel()

        # Differential drive kinematics
        rhs = ca.vertcat(v * ca.cos(theta),
                         v * ca.sin(theta),
                         omega)

        # Integrate over timestep
        f = ca.Function('f', [states, controls], [rhs])

        # Decision variables
        X = self.opti.variable(n_states, N + 1)
        U = self.opti.variable(n_controls, N)

        # Parameters
        X0 = self.opti.parameter(n_states)
        X_ref = self.opti.parameter(2)
        
        # DMPC Parameters
        P_own_prev = self.opti.parameter(2, N) # Own previous plan ˜x_i
        
        P_neighbors = []
        for _ in range(self.num_bots - 1):
            P_neighbors.append(self.opti.parameter(2, N)) # Neighbors' plans ˜x_j

        # Dynamics constraints
        self.opti.subject_to(X[:,0] == X0)
        for k in range(N):
            x_next = X[:, k] + dt * f(X[:, k], U[:, k])
            self.opti.subject_to(X[:, k + 1] == x_next)

            # ---------------------------------------------------------
            # DMPC CONSTRAINT 1: Consistency
            # ˆx_i(k+j) - ˜x_i(k+j) \in E_i 
            # Ensures we don't deviate from what we told neighbors
            # ---------------------------------------------------------
            dist_to_prev_plan = ca.sumsqr(X[0:2, k+1] - P_own_prev[:, k])
            self.opti.subject_to(dist_to_prev_plan <= self.E_i**2)

            # ---------------------------------------------------------
            # DMPC CONSTRAINT 2: Dynamic Collision Avoidance
            # Uses ˜x_j from neighbors to ensure safety
            # ---------------------------------------------------------
            for P_neigh in P_neighbors:
                dist_to_neighbor = ca.sumsqr(X[0:2, k+1] - P_neigh[:, k])
                self.opti.subject_to(dist_to_neighbor >= self.D_safe**2)

        # Objective: minimize tracking error in delta frame
        Q = ca.diag(ca.DM([10, 10, 1]))     # state cost
        R = ca.diag(ca.DM([0.1, 0.1]))      # control effort cost
        cost = 0
        for k in range(N):
            delta_pos = X[0:2, k] - X_ref
            cost += ca.mtimes([delta_pos.T, Q[0:2, 0:2], delta_pos])
            cost += ca.mtimes([U[:, k].T, R, U[:, k]])
        self.opti.minimize(cost)

        # Constraints on velocities
        self.opti.subject_to(self.opti.bounded(-self.max_linear_vel, U[0, :], self.max_linear_vel))
        self.opti.subject_to(self.opti.bounded(-self.max_angular_vel, U[1, :], self.max_angular_vel))

        # Solver options
        opts = {"ipopt.print_level": 0, "print_time": 0}
        self.opti.solver('ipopt', opts)

        # Store symbolic handles including DMPC ones
        self.X, self.U, self.X0, self.X_ref = X, U, X0, X_ref
        self.P_own_prev = P_own_prev
        self.P_neighbors = P_neighbors

    # =======================================
    # Control Loop
    # =======================================

    def control_loop(self):
        """Solve the MPC problem and publish velocity commands."""
        if self.current_pose is None or self.current_delta is None or self.r_vec is None:
            self.get_logger().info(f"{self.bot_id}: waiting for messages...")
            return
        try:
            x0 = np.array([self.current_pose[0], self.current_pose[1], self.current_pose[2]])
            ref = np.array([self.current_delta[0] + self.r_vec[0], self.current_delta[1] + self.r_vec[1]])

            if not hasattr(self, 'is_initialized'):
                # Initialize the previous trajectory to the current physical position
                self.own_prev_traj = np.tile(x0[0:2].reshape(-1, 1), (1, self.N))
                self.is_initialized = True

            self.opti.set_value(self.X0, x0)
            self.opti.set_value(self.X_ref, ref)

            # 1. Pass own shifted previous trajectory (˜x_i)
            self.opti.set_value(self.P_own_prev, self.own_prev_traj)

            # 2. Pass neighbors' trajectories (˜x_j)
            for i, bot_name in enumerate(self.agent_list):
                self.opti.set_value(self.P_neighbors[i], self.neighbor_trajectories[bot_name])

            self.opti.set_initial(self.U, np.zeros((2, self.N)))
            self.opti.set_initial(self.X, np.tile(x0.reshape(-1, 1), (1, self.N + 1)))

            sol = self.opti.solve()
            u_opt = sol.value(self.U)[:, 0]
            
            # Extract the optimal predicted path ˆx_i (excluding current state k=0)
            x_opt_path = sol.value(self.X)[0:2, 1:] 

            # 3. Publish cmd_vel
            cmd = Twist()
            cmd.linear.x = float(u_opt[0])
            cmd.angular.z = float(u_opt[1])
            self.cmd_pub.publish(cmd)

            # 4. Broadcast the predicted trajectory to neighbors (˜x_i for next step)
            traj_msg = Float64MultiArray()
            traj_msg.data = x_opt_path.T.flatten().tolist()
            self.traj_pub.publish(traj_msg)

            # 5. Shift trajectory for consistency constraint in the NEXT iteration
            # x(k+1) becomes the new x(k)
            self.own_prev_traj[:, 0:-1] = x_opt_path[:, 1:]
            self.own_prev_traj[:, -1] = x_opt_path[:, -1] # Duplicate last point to pad

        except Exception as e:
            self.get_logger().warning(f"{self.bot_id}: MPC solver failed: {e}")
            
            # If solver fails, relax the consistency constraint by resetting the prev plan
            self.own_prev_traj = np.tile(x0[0:2].reshape(-1, 1), (1, self.N))

def main(args=None):
    rclpy.init(args=args)
    node = DMPCController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
