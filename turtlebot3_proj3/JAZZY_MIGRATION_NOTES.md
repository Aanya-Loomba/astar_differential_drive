# Jazzy Migration Notes

This package was updated from a ROS 2 Humble + Gazebo Classic layout to a ROS 2 Jazzy + Gazebo Harmonic layout.

## Main changes
- Replaced `gazebo_ros_pkgs` / Gazebo Classic usage with `ros_gz_sim`, `ros_gz_bridge`, and `ros_gz_image`.
- Updated launch files to use `gz_sim.launch.py` and `ros_gz_sim create`.
- Added `params/turtlebot3_waffle_bridge.yaml` for ROS <-> Gazebo topic bridges.
- Converted the TurtleBot3 Waffle model from Gazebo Classic plugins to Gazebo systems plugins:
  - IMU sensor plugin removed in favor of the world IMU system
  - lidar changed from `ray` to `gpu_lidar`
  - camera plugin removed and replaced by topic + bridge based transport
  - diff drive moved to `gz::sim::systems::DiffDrive`
  - joint state moved to `gz::sim::systems::JointStatePublisher`
- Added Gazebo Harmonic world systems (`Physics`, `UserCommands`, `SceneBroadcaster`, `Sensors`, `Imu`).
- Updated the competition world to use Fuel `Ground Plane` and `Sun` models.
- Made `robot_state_publisher.launch.py` robust when `TURTLEBOT3_MODEL` is not set by defaulting to `waffle`.

## Expected external packages on Jazzy
Install at least:
- `ros-jazzy-ros-gz-sim`
- `ros-jazzy-ros-gz-bridge`
- `ros-jazzy-ros-gz-image`
- standard ROS 2 Jazzy desktop dependencies

## Launch
```bash
export TURTLEBOT3_MODEL=waffle
ros2 launch turtlebot3_project3 competition_world.launch.py
```
