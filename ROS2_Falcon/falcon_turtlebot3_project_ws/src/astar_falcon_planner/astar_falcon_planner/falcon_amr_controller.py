"""Closed-loop ROS 2 controller for the FalconSim A* planner.

The planner returns relative motion increments in Falcon-local centimeters.  This
node converts those increments into dense waypoints, estimates the relationship
between Falcon-local coordinates and ROS TF from the initial pose, and tracks the
waypoints using feedback from the published transform.

This controller intentionally publishes world-frame ``linear.x`` and
``linear.y`` commands because the provided FalconSim scenario script applies
``cmd_vel`` directly as actor offsets.  It is therefore a waypoint tracker for
the FalconSim bridge, not a physical differential-drive controller.
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException

from .submodules.astar_planner import plan_path


class RobotAStarPlannerNode(Node):
    """ROS node that plans once and tracks the resulting path with TF feedback.

    The first available transform is used to calibrate local map coordinates to
    the ROS TF frame.  After calibration, each timer tick computes the current
    Falcon-local pose, selects the next waypoint, and publishes a bounded
    proportional velocity command toward that target.
    """

    def __init__(self):
        super().__init__('robot_a_star_planner')
        self.get_logger().info('READY Closed-Loop AStar Planner')

        start = self.declare_parameter(
            'start_position', [0.30, 1.50, 0.0]
        ).get_parameter_value().double_array_value
        end = self.declare_parameter(
            'end_position', [5.00, 1.50, 0.0]
        ).get_parameter_value().double_array_value
        robot_radius = self.declare_parameter(
            'robot_radius', 0.171
        ).get_parameter_value().double_value
        clearance = self.declare_parameter(
            'clearance', 0.01
        ).get_parameter_value().double_value
        delta_time = self.declare_parameter(
            'delta_time', 4.0
        ).get_parameter_value().double_value
        goal_threshold = self.declare_parameter(
            'goal_threshold', 0.2
        ).get_parameter_value().double_value
        wheel_radius = self.declare_parameter(
            'wheel_radius', 0.033
        ).get_parameter_value().double_value
        wheel_distance = self.declare_parameter(
            'wheel_distance', 0.287
        ).get_parameter_value().double_value
        rpms = self.declare_parameter(
            'rpms', [25.0, 100.0]
        ).get_parameter_value().double_array_value

        # Controller gains and tolerances are expressed in Falcon-local centimeters.
        self.control_rate_hz = self.declare_parameter(
            'control_rate_hz', 30.0
        ).get_parameter_value().double_value
        self.kp_xy = self.declare_parameter(
            'kp_xy', 1.25
        ).get_parameter_value().double_value
        self.max_speed_cm_s = self.declare_parameter(
            'max_speed_cm_s', 45.0
        ).get_parameter_value().double_value
        self.waypoint_tolerance_cm = self.declare_parameter(
            'waypoint_tolerance_cm', 8.0
        ).get_parameter_value().double_value
        self.final_tolerance_cm = self.declare_parameter(
            'final_tolerance_cm', 6.0
        ).get_parameter_value().double_value
        self.max_waypoint_spacing_cm = self.declare_parameter(
            'max_waypoint_spacing_cm', 5.0
        ).get_parameter_value().double_value

        # Launch parameters are in meters.  The planner API uses centimeters so
        # that obstacle geometry and robot dimensions share one unit system.
        int_start = [int(element * 100) for element in start]
        int_end = [int(element * 100) for element in end]
        robot_radius_cm = robot_radius * 100.0
        clearance_cm = clearance * 100.0
        goal_threshold_cm = goal_threshold * 100.0
        wheel_radius_cm = wheel_radius * 100.0
        wheel_distance_cm = wheel_distance * 100.0

        self.start_xy_cm = (float(int_start[0]), float(int_start[1]))
        self.goal_xy_cm = (float(int_end[0]), float(int_end[1]))

        path = plan_path(
            tuple(int_start),
            tuple(int_end),
            robot_radius_cm,
            clearance_cm,
            delta_time,
            goal_threshold_cm,
            wheel_radius_cm,
            wheel_distance_cm,
            float(rpms[0]),
            float(rpms[1]),
            self.get_logger(),
        )

        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.current_twist = Twist()

        self.raw_path = path
        self.waypoints = self._build_waypoints_from_path(path)
        self.wp_idx = 1 if len(self.waypoints) > 1 else 0
        self.done = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_offset_x_m = None
        self.tf_offset_y_m = None

        timer_period = 1.0 / max(self.control_rate_hz, 1.0)
        self.timer = self.create_timer(timer_period, self.on_timer)

        self.get_logger().info(
            f'Closed-loop path planned: {len(path)} planner moves, '
            f'{len(self.waypoints)} tracking waypoints. '
            f'start_cm={self.start_xy_cm}, goal_cm={self.goal_xy_cm}'
        )

    def _build_waypoints_from_path(self, path):
        """Turn [dx, dy, dtheta] increments into dense local-frame waypoints."""
        waypoints = [[self.start_xy_cm[0], self.start_xy_cm[1]]]
        x = self.start_xy_cm[0]
        y = self.start_xy_cm[1]

        for move in path:
            if len(move) < 2:
                continue
            nx = x + float(move[0])
            ny = y + float(move[1])
            self._append_dense_segment(waypoints, (x, y), (nx, ny))
            x, y = nx, ny

        # Always include the requested goal as the final waypoint, even when the
        # planner stopped inside its goal threshold.
        self._append_dense_segment(waypoints, (x, y), self.goal_xy_cm)
        return waypoints

    def _append_dense_segment(self, waypoints, p0, p1):
        dx = float(p1[0] - p0[0])
        dy = float(p1[1] - p0[1])
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return

        spacing = max(float(self.max_waypoint_spacing_cm), 1.0)
        n = max(1, int(math.ceil(dist / spacing)))
        for i in range(1, n + 1):
            a = i / n
            waypoints.append([p0[0] + a * dx, p0[1] + a * dy])

    def _lookup_robot_tf(self):
        from_frame_rel = 'IMUSensor_BP_C_0'
        to_frame_rel = 'map'
        return self.tf_buffer.lookup_transform(
            to_frame_rel,
            from_frame_rel,
            rclpy.time.Time(),
        )

    def _initialize_tf_offsets_if_needed(self, transform):
        if self.tf_offset_x_m is not None and self.tf_offset_y_m is not None:
            return

        tx = transform.transform.translation.x
        ty = transform.transform.translation.y

        # Calibration model:
        #   tf_x = offset_x + local_x_cm / 100
        #   tf_y = offset_y - local_y_cm / 100
        self.tf_offset_x_m = tx - self.start_xy_cm[0] / 100.0
        self.tf_offset_y_m = ty + self.start_xy_cm[1] / 100.0

        self.get_logger().info(
            'TF/local calibration: '
            f'offset_x={self.tf_offset_x_m:.4f} m, '
            f'offset_y={self.tf_offset_y_m:.4f} m. '
            'Offsets were estimated from the first TF sample and launch start pose.'
        )

    def _current_local_xy_cm(self, transform):
        self._initialize_tf_offsets_if_needed(transform)
        tx = transform.transform.translation.x
        ty = transform.transform.translation.y

        x_cm = 100.0 * (tx - self.tf_offset_x_m)
        y_cm = -100.0 * (ty - self.tf_offset_y_m)
        return x_cm, y_cm

    @staticmethod
    def _make_zero_twist():
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0
        return msg

    def _publish_stop(self):
        self.current_twist = self._make_zero_twist()
        self.publisher.publish(self.current_twist)

    def on_timer(self):
        if self.done:
            self._publish_stop()
            return

        try:
            transform = self._lookup_robot_tf()
        except TransformException as ex:
            self.get_logger().info(f'Could not get robot TF: {ex}')
            self._publish_stop()
            return

        current_x, current_y = self._current_local_xy_cm(transform)

        if len(self.waypoints) <= 1:
            self.get_logger().warning('No valid waypoints. Stopping.')
            self.done = True
            self._publish_stop()
            return

        # Advance only after feedback confirms the robot is near the waypoint.
        while self.wp_idx < len(self.waypoints) - 1:
            wx, wy = self.waypoints[self.wp_idx]
            if math.hypot(wx - current_x, wy - current_y) > self.waypoint_tolerance_cm:
                break
            self.wp_idx += 1

        target_x, target_y = self.waypoints[self.wp_idx]
        error_x = target_x - current_x
        error_y = target_y - current_y
        # Stop once the measured pose is within the final goal tolerance.
        final_x, final_y = self.waypoints[-1]
        final_error = math.hypot(final_x - current_x, final_y - current_y)
        if self.wp_idx >= len(self.waypoints) - 1 and final_error <= self.final_tolerance_cm:
            self.get_logger().info(
                f'Goal reached by TF feedback. current=({current_x:.1f}, {current_y:.1f}) cm, '
                f'goal=({final_x:.1f}, {final_y:.1f}) cm'
            )
            self.done = True
            self._publish_stop()
            return

        # Proportional control in Falcon-local X/Y.  The FalconSim scenario uses
        # cmd_vel.linear.x/y as actor offsets, so these commands are world-frame
        # tracking velocities rather than body-frame wheel commands.
        vx = self.kp_xy * error_x
        vy = self.kp_xy * error_y
        speed = math.hypot(vx, vy)
        max_speed = max(float(self.max_speed_cm_s), 1.0)
        if speed > max_speed:
            scale = max_speed / speed
            vx *= scale
            vy *= scale

        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.linear.z = 0.0

        # Leave angular velocity at zero.  The scenario script aligns the actor
        # with the XY velocity vector when there is no explicit yaw command.
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0

        self.current_twist = msg
        self.publisher.publish(msg)


def main():
    rclpy.init()
    node = RobotAStarPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
