from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess,
                            LogInfo, RegisterEventHandler, TimerAction)
from launch.event_handlers import (OnExecutionComplete, OnProcessExit,
                                OnProcessIO, OnProcessStart, OnShutdown)
from launch.substitutions import (EnvironmentVariable, FindExecutable,
                                LaunchConfiguration, LocalSubstitution,
                                PythonExpression)

def generate_launch_description():

    #Setup Launch Parameters
    start_position = LaunchConfiguration('start_position')
    end_position = LaunchConfiguration('end_position')
    robot_radius = LaunchConfiguration('robot_radius')
    clearance = LaunchConfiguration('clearance')
    delta_time = LaunchConfiguration('delta_time')
    goal_threshold = LaunchConfiguration('goal_threshold')
    wheel_radius = LaunchConfiguration('wheel_radius')
    wheel_distance = LaunchConfiguration('wheel_distance')
    rpms = LaunchConfiguration('rpms')

    start_position_launch_arg = DeclareLaunchArgument(
        'start_position',
        default_value='[0.30, 1.50, 0.0]'
    )
    end_position_launch_arg = DeclareLaunchArgument(
        'end_position',
        default_value='[5.00, 1.50, 0.0]'
    )
    robot_radius_launch_arg = DeclareLaunchArgument(
        'robot_radius',
        default_value='0.171'
    )
    clearance_launch_arg = DeclareLaunchArgument(
        'clearance',
        default_value='0.01'
    )
    delta_time_launch_arg = DeclareLaunchArgument(
        'delta_time',
        default_value='4.0'
    )
    goal_threshold_launch_arg = DeclareLaunchArgument(
        'goal_threshold',
        default_value='0.2'
    )
    wheel_radius_launch_arg = DeclareLaunchArgument(
        'wheel_radius',
        default_value='0.033'
    )
    wheel_distance_launch_arg = DeclareLaunchArgument(
        'wheel_distance',
        default_value='0.287'
    )
    rpms_launch_arg = DeclareLaunchArgument(
        'rpms',
        default_value='[25.0, 100.0]'
    )

    #Create Process to Launch Falcon
    launch_falcon_sim = ExecuteProcess(
        cmd=[
            './Falcon.sh',
            '/home/ubuntu/Scenarios/AMRPathPlanning/AMRPathPlanning.usda',
            EnvironmentVariable('FALCON_CLOUD_TOKEN')
        ],
        cwd="/home/ubuntu/duality/falconsim/",
    )
    
    #Create Node to run AStar and Control TurtleBot
    astar_control_node = Node(
        package='astar_falcon_planner',
        # namespace='astar_ns',
        executable='falcon_amr_controller',
        name='falcon_amr_controller',
        parameters=[{
                'start_position': start_position,
                'end_position': end_position,
                'robot_radius': robot_radius,
                'clearance': clearance,
                'delta_time': delta_time,
                'goal_threshold': goal_threshold,
                'wheel_radius': wheel_radius,
                'wheel_distance': wheel_distance,
                'rpms': rpms,
            }]
    )

    return LaunchDescription([
        start_position_launch_arg,
        end_position_launch_arg,
        robot_radius_launch_arg,
        clearance_launch_arg,
        delta_time_launch_arg,
        goal_threshold_launch_arg,
        wheel_radius_launch_arg,
        wheel_distance_launch_arg,
        rpms_launch_arg,
        astar_control_node,
        launch_falcon_sim,
    ])
