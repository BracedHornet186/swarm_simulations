# swarm_control

A ROS 2 package for **decentralized swarm coordination and control** using a **finite-time kinematic model** and **dynamic leader election** based on graph connectivity.

This package simulates a multi-robot system (e.g., TurtleBot3s in Gazebo) where each agent interacts only with its neighbors within a vision radius.  
The system collectively tracks a reference trajectory in a **distributed** and **leader-adaptive** manner.


## Package Overview

```
swarm_control/
├── CMakeLists.txt
├── launch
│   ├── empty_world.launch.py
│   ├── swarm_launch.py
│   └── swarm_mpc_launch.py
├── models
├── package.xml
├── params
│   ├── minimal_bridge.yaml
│   └── swarm_config.yaml
├── rviz
│   └── swarm.rviz
├── swarm_control
│   ├── graph_observer.py
│   ├── kinematic_node.py
│   ├── mpc_controller.py
│   ├── pose_publisher_node.py
│   ├── reference_node.py
│   ├── visualizer_node.py
│   └── utils
│       └── namespace_utils.py
├── urdf
│   ├── common_properties.urdf
│   ├── minimal_urdf.urdf
│   └── turtlebot3_waffle.urdf
├── worlds
│   ├── empty_world.world
│   └── tb3_world.world
└── README.md
```

```
swarm_control_msgs/
├── CMakeLists.txt
├── package.xml
├── msg/
│   ├── Info.msg
│   └── RBroadcast.msg
└── README.md
```

## Parameters

| Parameter | Node | Description | Default |
|------------|------|--------------|----------|
| `num_bots` | all | Number of robots in the swarm | `3` |
| `bot_id` | kinematic_node | Robot namespace ID (e.g. `"bot1"`) | — |
| `role` | kinematic_node | `"leader"` or `"agent"` | `"agent"` |
| `delta_radius` | graph_observer | Vision/communication radius [m] | `1.5` |
| `sampling_freq` | reference_node | Sampling Frequency of reference [hz] | `2.0` |
| `control_freq` | mpc_node | Controller Frequency of MPC node [hz] | `10.0` |

## Running the Simulation

### 1. Build
```bash
cd ~/ros2_ws/src
git clone https://github.com/BracedHornet186/swarm_simulations.git
export GZ_SIM_RESOURCE_PATH=$(pwd)/swarm_control/models/:$GZ_SIM_RESOURCE_PATH
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

### 2. Launch Swarm Control Stack
```bash
ros2 launch swarm_control swarm_mpc_launch.py num_bots:=3 
```

This will:
- Spawn 3 robots in Gazebo  
- Launch one `kinematic_node.py` per bot
- Launch one `mpc_controller.py` per bot 
- Start `pose_publisher_node.py` 
- Start `graph_observer.py`  
- Start `reference.py`


## Topic Overview (per bot)

| Topic | Type | Publisher | Description |
|--------|------|------------|--------------|
| `/bot_i/pose` | `geometry_msgs/PoseStamped` | Gazebo | Ground-truth position |
| `/bot_i/delta` | `geometry_msgs/PointStamped` | kinematic_node | Local Δ(t) state |
| `/bot_i/info` | `swarm_control/Info` | graph_observer | Status + role |
| `/r_broadcast` | `swarm_control/RBroadcast` | reference | Leader’s reference broadcast |
| `/reference` | `geometry_msgs/PointStamped` | reference | True reference trajectory |
| `/bot_i/cmd_vel` | `geometry_msgs/Twist` | mpc_node | Control input (velocity) |


## Algorithmic Flow

1. **Graph Construction:** `graph_observer` builds a time-varying adjacency graph based on inter-robot distances.  
2. **Leader Election:** The robot with the **highest degree** in each connected component becomes leader.  
3. **Reference Broadcast:** Leaders publish the reference r(t) to `/r_broadcast`.  
4. **Decentralized Control:** Each agent computes Δᵢ = zᵢ − r(t) using neighbor states and performs the finite-time update.  
5. **Dynamic Reconfiguration:** When components merge/split, `graph_observer` redefines leaders and publishes updated roles.
6. **MPC Node:** Solves the control problem using differential drive model to publish `/bot_i/cmd_vel` input.



## Extending the Package

- Add a **Model Predictive Control (MPC)** layer for velocity control using `/bot_i/delta` as the input reference.  
- Include a **visualization node** publishing `visualization_msgs/Marker` lines for graph edges in RViz.  
- Implement health metrics in `Info.msg` for real-world diagnostics.  
- Integrate `rclpy.lifecycle` nodes for fault-tolerant reconfiguration.



## Dependencies

- **ROS 2 Jazzy**
- **Gazebo Harmonic**
- `geometry_msgs`
- `rclpy`
- `numpy`


## Author & Maintainers

Developed by **Yash Purswani** as part of a decentralized swarm control project for **ME5253: Network Dynamics and Controls** at **IIT Madras**.
