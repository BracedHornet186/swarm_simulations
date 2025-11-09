#!/usr/bin/env python3

import os
import yaml
import random

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    IncludeLaunchDescription,
    DeclareLaunchArgument,
    OpaqueFunction
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

from swarm_control.utils.namespace_utils import load_sdf_with_namespace, create_namespaced_bridge_yaml

def launch_setup(context, *args, **kwargs):
    random.seed(42)
    # Paths
    swarm_dir = get_package_share_directory('swarm_control')
    ros_gz_sim_dir = get_package_share_directory('ros_gz_sim')

    # Simulation config
    world_path = os.path.join(swarm_dir, 'worlds', 'empty_world.world')
    
    actions = []
    # Launch Gazebo server and client
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim_dir, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': f'-r -s -v2 {world_path}', 'on_exit_shutdown': 'true'}.items()
    )
    actions.append(gzserver_cmd)

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim_dir, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': '-g -v2', 'on_exit_shutdown': 'true'}.items()
    )
    actions.append(gzclient_cmd)

    # Add GZ model path to env
    environment = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(swarm_dir, 'models')
    )
    actions.append(environment)

    # Read evaluated values
    use_sim_time = LaunchConfiguration('use_sim_time', default='true').perform(context)
    num_bots = int(LaunchConfiguration('num_bots').perform(context))
    delta_radius = float(LaunchConfiguration('delta_radius', default=3.0).perform(context))
    sampling_freq = float(LaunchConfiguration('sampling_freq', default=2.0).perform(context))
    control_freq = float(LaunchConfiguration('control_freq', default=10.0).perform(context))

    # Load model and URDF
    TURTLEBOT3_MODEL = 'waffle'
    model_dir = f'turtlebot3_{TURTLEBOT3_MODEL}'
    # sdf_file_name = 'model.sdf'
    sdf_file_name = 'minimal_model.sdf'
    sdf_path = os.path.join(swarm_dir, 'models', model_dir, sdf_file_name)

    remappings = [("/tf", "tf"), ("/tf_static", "tf_static")]
    frame_prefix = LaunchConfiguration('frame_prefix', default='')

    # urdf_file_name = 'turtlebot3_' + TURTLEBOT3_MODEL + '.urdf'
    urdf_file_name = 'minimal_urdf.urdf'
    urdf_path = os.path.join(swarm_dir, 'urdf', urdf_file_name)

    with open(urdf_path, 'r') as infp:
        robot_desc = infp.read()

    # Spawn each bot
    for i in range(num_bots):
        namespace = f'bot{i + 1}'
        spawn_radius = 3.0
        x_pose = round(random.uniform(-spawn_radius, spawn_radius), 2)
        y_pose = round(random.uniform(-spawn_radius, spawn_radius), 2)

        patched_sdf = load_sdf_with_namespace(sdf_path, namespace)

        # Robot state publisher
        robot_state_publisher = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            namespace=namespace,
            remappings=remappings,
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time == 'true',
                'robot_description': robot_desc,
                'frame_prefix': PythonExpression(["'", frame_prefix, "/'"])
            }])
        actions.append(robot_state_publisher)

        # Spawn robot
        spawner_node = Node(
            package='ros_gz_sim',
            executable='create',
            namespace=namespace,
            arguments=[
                '-name', f'{namespace}',
                '-string', patched_sdf,
                '-x', str(x_pose),
                '-y', str(y_pose),
                '-z', '0.01',
            ],
            output='screen',
        )
        actions.append(spawner_node)

        # bridge_template = os.path.join(swarm_dir, 'params', f'{TURTLEBOT3_MODEL}_bridge.yaml')
        bridge_template = os.path.join(swarm_dir, 'params', 'minimal_bridge.yaml')
        namespaced_bridge = create_namespaced_bridge_yaml(bridge_template, namespace)

        bridge_node = Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=['--ros-args', '-p', f'config_file:={namespaced_bridge}'],
            output='screen',
        )
        actions.append(bridge_node)

    # In a multi-robot setup using Gazebo Sim (Harmonic or later), each robot typically
    # requires a separate ROS-Gazebo bridge to relay topics such as sensor data, odometry,
    # and control commands between Gazebo and ROS 2.
    # However, some topics like `/clock` are *global* and should be published only once
    # to avoid conflicts or duplication. If multiple bridges publish `/clock`, it may lead
    # to inconsistent simulation time behavior across nodes or unnecessary topic traffic.
    # Therefore, the `/clock` topic is handled separately:
    # - It is excluded from the per-robot bridge configuration files (YAMLs).
    # - A dedicated, single bridge instance is launched to publish `/clock` from Gazebo to ROS 2.
    # This ensures consistent simulation time across the entire ROS 2 system while supporting
    # multiple robot instances with their own bridges.

    # Global clock bridge
    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='clock_bridge',
        output='screen',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        )
    actions.append(clock_bridge)

    # Pose Publisher Node
    pose_pub_node = Node(
        package='swarm_control',
        executable='pose_publisher_node.py',
        name='pose_publisher_node',
        output='screen',
        parameters=[{'num_bots': num_bots}]
    )
    actions.append(pose_pub_node)

    # Graph Observer Node
    graph_node = Node(
        package='swarm_control',
        executable='graph_observer.py',
        name='graph_observer',
        output='screen',
        parameters=[{
            'num_bots': num_bots, 
            'delta_radius': delta_radius,
            'frequency': sampling_freq,        
        }]
        )
    actions.append(graph_node)

    # Reference Node
    reference_node = Node(
        package='swarm_control',
        executable='reference_node.py',
        name='reference_node',
        output='screen',
        parameters=[{
            'num_bots': num_bots,
            'frequency': sampling_freq,
        }]
    )
    actions.append(reference_node)

    # One per roboot
    for i in range(num_bots):
        bot_id = f'bot{i + 1}'
        
        # Kinematic Node
        kinematic_node = Node(
            package='swarm_control',
            executable='kinematic_node.py',
            name=f'kinematic_node_{bot_id}',
            namespace=bot_id,
            output='screen',
            parameters=[{
                'bot_id': bot_id,
                'num_bots': num_bots,
                'sampling_freq': sampling_freq,
            }]
        )
        actions.append(kinematic_node)

        # MPC Node
        mpc_node = Node(
            package='swarm_control',
            executable='mpc_controller.py',
            name=f'mpc_node_{bot_id}',
            namespace=bot_id,
            output='screen',
            parameters=[{
                'bot_id': bot_id,
                'num_bots': num_bots,
                'control_frequency': control_freq,
            }]
        )
        actions.append(mpc_node)

    # Visualize Node
    visualizer_node = Node(
        package='swarm_control',
        executable='visualizer_node.py',
        name='visualizer_node',
        output='screen',
        parameters=[{
            'num_bots': num_bots,
        }]
    )
    actions.append(visualizer_node)

    return actions

def generate_launch_description():
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    declare_num_bots = DeclareLaunchArgument(
        'num_bots',
        default_value='3',
        description='Number of TurtleBot3 robots to spawn'
    )

    declare_sampling_freq = DeclareLaunchArgument(
        'sampling_freq',
        default_value='2.0',
        description='Sampling frequency of kinematic model'
    )

    declare_control_freq = DeclareLaunchArgument(
        'control_freq',
        default_value='10.0',
        description='Control frequency of the MPC controller'
    )

    declare_delta_radius = DeclareLaunchArgument(
        'delta_radius',
        default_value='3.0',
        description='Radius of communication'
    )
    
    return LaunchDescription([
        declare_use_sim_time,
        declare_num_bots,
        declare_sampling_freq,
        declare_control_freq,
        declare_delta_radius,
        OpaqueFunction(function=launch_setup)
    ])