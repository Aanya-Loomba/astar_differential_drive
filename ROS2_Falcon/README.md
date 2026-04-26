# FalconSim A* TurtleBot3 Planner

This repository contains a FalconSim-compatible A* planner and a closed-loop ROS 2 controller for navigating a TurtleBot3-style AMR through the warehouse map used in the AMR path planning task.

The planner computes a collision-checked path using differential-drive motion primitives. The controller converts the planned path into dense waypoints and tracks them using the robot pose published through TF.

## Members
Aanya Loomba - 122298880 aloomba@umd.edu

Anvesh Som - 122298682 anvesh@umd.edu

Ryan Lowe - 1222583389 rllowe25@umd.edu

## Repository layout

Place the files in the ROS 2 package as follows:

```text
astar_falcon_planner/
├── astar_falcon_planner/
│   ├── falcon_amr_controller.py
│   └── submodules/
│       └── astar_planner.py
└── launch/
    └── ros_falcon_astar.launch.py
```

Use these generated files:

```text
astar_planner.py        -> astar_falcon_planner/submodules/astar_planner.py
falcon_amr_controller.py -> astar_falcon_planner/falcon_amr_controller.py
```

## What the code does

### `astar_planner.py`

The planner exposes one public function:

```python
plan_path(
    start,
    end,
    robot_radius,
    clearance,
    delta_time,
    goal_threshold,
    wheel_radius,
    wheel_distance,
    rpm1,
    rpm2,
    logger=None,
)
```

It returns a list of relative Falcon-local motion increments:

```python
[[dx_cm, dy_cm, dtheta_rad], ...]
```

Internally, the planner:

- converts Falcon-local coordinates into the planner's y-up frame;
- expands obstacles and boundaries by `robot_radius + clearance`;
- searches with A* over differential-drive RPM motion primitives;
- reconstructs the final path from parent pointers;
- converts the internal planner-frame path back into Falcon-local increments.

### `falcon_amr_controller.py`

The controller is a ROS 2 node that:

- reads launch parameters for start, goal, robot dimensions, wheel dimensions, and RPMs;
- calls `plan_path()` once at startup;
- converts planner increments into dense waypoints;
- reads the robot pose from TF;
- estimates the offset between Falcon-local coordinates and the ROS TF frame;
- uses a bounded proportional controller to drive toward each waypoint.

This controller is closed-loop with respect to position. It is not a physical differential-drive wheel controller, because the provided FalconSim scenario applies `cmd_vel.linear.x` and `cmd_vel.linear.y` directly as actor world offsets.

## Coordinate conventions

The system uses three coordinate conventions:

| Frame | Units | Notes |
|---|---:|---|
| Launch parameters | meters | Passed through `ros_falcon_astar.launch.py` |
| Planner/controller local frame | centimeters | Used by `plan_path()` and waypoint tracking |
| ROS TF frame | meters | Published by FalconSim |

The measured Falcon relation was:

```text
tf_x = offset_x + local_x_cm / 100
tf_y = offset_y - local_y_cm / 100
```

So local X has the same sign as TF X, while local Y is flipped relative to TF Y.

The planner stores obstacle geometry in a y-up internal frame. The helper functions in `astar_planner.py` mirror coordinates at the API boundary so the planner can keep a clean map convention while the controller remains compatible with FalconSim.

## Installation

From the FalconSim ROS 2 workspace:

```bash
cd ~/Scenarios/AMRPathPlanning/ROS2/falcon_turtlebot3_project_ws
```

Copy the files into the package:

```bash
cp /path/to/astar_planner_opensource.py \
  src/astar_falcon_planner/astar_falcon_planner/submodules/astar_planner.py

cp /path/to/falcon_amr_controller_opensource.py \
  src/astar_falcon_planner/astar_falcon_planner/falcon_amr_controller.py
```

Build and source the workspace:

```bash
colcon build --symlink-install
source install/setup.bash
```

If you are iterating quickly, `--symlink-install` is useful because source-file edits are reflected without repeatedly copying into the install tree. Rebuild after changing package entry points, dependencies, or launch files.

## Running

Launch FalconSim and the planner/controller node:

```bash
ros2 launch astar_falcon_planner ros_falcon_astar.launch.py
```

You can override the start and goal from the command line:

```bash
ros2 launch astar_falcon_planner ros_falcon_astar.launch.py \
  start_position:="[0.30, 1.50, 0.0]" \
  end_position:="[3.70, 1.50, 0.0]"
```

Parameter units:

```text
start_position: [x_m, y_m, theta_rad]
end_position:   [x_m, y_m, theta_rad]
robot_radius:   meters
clearance:      meters
goal_threshold: meters
wheel_radius:   meters
wheel_distance: meters
rpms:           [rpm1, rpm2]
```

Example with a smaller clearance:

```bash
ros2 launch astar_falcon_planner ros_falcon_astar.launch.py \
  start_position:="[0.30, 1.50, 0.0]" \
  end_position:="[3.70, 1.50, 0.0]" \
  clearance:="0.005"
```

## Controller tuning

The closed-loop controller exposes several parameters:

| Parameter | Default | Meaning |
|---|---:|---|
| `control_rate_hz` | `30.0` | Timer frequency for waypoint tracking |
| `kp_xy` | `1.25` | Proportional gain in local X/Y |
| `max_speed_cm_s` | `45.0` | Speed limit for published linear command |
| `waypoint_tolerance_cm` | `8.0` | Distance threshold to advance to the next waypoint |
| `final_tolerance_cm` | `6.0` | Distance threshold for goal completion |
| `max_waypoint_spacing_cm` | `5.0` | Maximum spacing used when densifying planner increments |

If the robot overshoots waypoints or swings around the goal, try lowering speed and gain:

```bash
ros2 launch astar_falcon_planner ros_falcon_astar.launch.py \
  max_speed_cm_s:="30.0" \
  kp_xy:="0.8"
```

If the robot stalls close to waypoints, increase the waypoint tolerance:

```bash
ros2 launch astar_falcon_planner ros_falcon_astar.launch.py \
  waypoint_tolerance_cm:="12.0"
```

## Map and scale tuning

The planner map is controlled by constants in `astar_planner.py`:

```python
MAP_SCALE = 1.0
MAP_W_CM = 400.0 * MAP_SCALE
MAP_H_CM = 200.0 * MAP_SCALE
```

`MAP_SCALE` affects the planner's imagined obstacle geometry. It does not change the units of the commands sent to FalconSim.

If the robot follows the path well but the path appears shifted or scaled relative to real walls, check the obstacle map scale and frame conversion before changing the controller. A good debugging workflow is:

1. Run a simple straight-line plan.
2. Compare the expected TF start/goal in the logs with the actual TF values.
3. Check whether the robot reaches the correct TF displacement.
4. If displacement is correct but walls are wrong, tune the obstacle map scale or obstacle coordinates.
5. If displacement is wrong, inspect the local-to-TF conversion in the controller.

## How to inspect TF

To watch the robot transform:

```bash
ros2 run tf2_ros tf2_echo map IMUSensor_BP_C_0
```

The controller calibrates the local-to-TF offset from the first transform it receives. The log should print values close to the measured offsets for the current FalconSim scene.

## Known limitations

- The planner uses an assumed obstacle map rather than querying FalconSim geometry at runtime.
- The controller tracks waypoints in Falcon-local X/Y, not true differential-drive body-frame velocity.
- The FalconSim scenario applies `cmd_vel.linear.x` and `cmd_vel.linear.y` as actor offsets, so the motion model differs from a physical TurtleBot.
- If the imagined map is scaled or shifted relative to the actual FalconSim warehouse, the robot can follow the planned path correctly while still colliding with real walls.
- The planner stops when it reaches `goal_threshold`; a final correction segment may still be needed to reach the exact requested goal.

## Challenges faced during development

### 1. Open-loop execution drift

The first implementation executed each planner increment for a fixed duration. This worked only if FalconSim executed every command exactly as assumed. In practice, timing delay and transform drift caused the robot to move through walls or miss turns. The controller was changed to a closed-loop waypoint follower that advances only when TF feedback shows the robot has reached the next waypoint.

### 2. FalconSim does not behave like a standard differential-drive simulator

The planner uses differential-drive kinematics while the FalconSim scenario script applies `cmd_vel.linear.x` and `cmd_vel.linear.y` directly as world offsets. This can make the robot appear to slide sideways if large increments are executed open loop. The controller therefore tracks dense X/Y waypoints and lets the scenario's visual correction align the robot mesh with the commanded direction.

### 3. Coordinate-frame mismatch

The planner's original map uses a y-up convention, while FalconSim TF showed that local positive Y corresponds to decreasing TF Y. This required an explicit Y flip between Falcon-local coordinates and the internal planner frame.

### 4. Map-scale uncertainty

The original Gazebo planner assumed a scaled map. FalconSim appeared to use a differently scaled warehouse in some tests. This made the planner's imaginary obstacles appear shifted relative to the real walls. `MAP_SCALE` was kept explicit so it can be tuned independently from command units.

### 5. Start and goal validation after obstacle inflation

A point can look valid visually but still be invalid for planning after inflating walls and obstacles by the robot radius plus clearance. This was especially visible near the left boundary opening, where the raw opening is wider than the safe region for the robot center.

### 6. ROS 2 logger severity issue

A helper that dynamically called different logger severities from the same source line triggered a ROS 2 logging error. The logging helper was updated so each severity is dispatched from a separate branch, with a stdout fallback.

## Development notes

Useful commands while iterating:

```bash
# Rebuild the package
colcon build --symlink-install
source install/setup.bash

# Run the launch file
ros2 launch astar_falcon_planner ros_falcon_astar.launch.py

# Inspect TF
ros2 run tf2_ros tf2_echo map IMUSensor_BP_C_0

# Stop any running velocity command manually
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}"
```

## Expected behavior

A successful run should show logs similar to:

```text
READY Closed-Loop AStar Planner
Planning A* path ...
A* path found with N Falcon moves ...
Closed-loop path planned: N planner moves, M tracking waypoints
TF/local calibration: offset_x=..., offset_y=...
Goal reached by TF feedback ...
```
