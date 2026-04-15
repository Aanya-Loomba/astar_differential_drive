#!/usr/bin/env python3
#
# ROS2 Jazzy keyboard teleop for TurtleBot3 / cmd_vel robots
#
# Features:
# - Normal cmd_vel mode
# - RPM test mode for differential drive tuning
# - Live switching between modes from keyboard
#
# Keys:
#   arrows      : motion / steering
#   space       : stop
#   q           : quit
#   m           : toggle cmd_vel / rpm mode
#   z / x       : decrease / increase RPM step size
#   a / s       : decrease / increase max linear speed (cmd_vel mode)
#   d / f       : decrease / increase max angular speed (cmd_vel mode)
#   1,2,3       : decay modes carried over from original script
#
# In RPM mode:
#   up/down     : increase/decrease BOTH wheel RPMs
#   left/right  : bias left/right by changing wheel RPM difference
#   r           : reset wheel RPMs to zero
#
# In cmd_vel mode:
#   up/down     : change linear command
#   left/right  : change angular command
#
# The published /cmd_vel values in RPM mode are computed from:
#   v = (R/2) * (wl + wr)
#   w = (R/L) * (wr - wl)
# where wl, wr are wheel angular speeds in rad/s derived from RPM.
#

import curses
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class TextWindow:
    def __init__(self, stdscr, lines=18):
        self._screen = stdscr
        self._screen.nodelay(True)
        curses.curs_set(0)
        self._num_lines = lines

    def read_key(self):
        keycode = self._screen.getch()
        return keycode if keycode != -1 else None

    def clear(self):
        self._screen.clear()

    def write_line(self, line_no, message):
        if line_no < 0 or line_no >= self._num_lines:
            raise ValueError(BColors.FAIL + 'line_no is out of bounds' + BColors.ENDC)
        height, width = self._screen.getmaxyx()
        y = int((height / self._num_lines) * line_no)
        x = 2
        for text in message.split('\n'):
            text = text.ljust(max(1, width - x))
            try:
                self._screen.addstr(y, x, text[: max(1, width - x - 1)])
            except curses.error:
                pass
            y += 1

    def refresh(self):
        self._screen.refresh()

    def beep(self):
        curses.flash()


class TeleopTurtlebot(Node):
    MODE_CMD_VEL = 'cmd_vel'
    MODE_RPM = 'rpm'

    def __init__(self, interface: TextWindow):
        super().__init__('teleop_turtlebot_keyboard')

        self._interface = interface

        # ROS params
        self.declare_parameter('hz', 15.0)
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        # cmd_vel mode params
        self.declare_parameter('max_linear_vel', 0.26)
        self.declare_parameter('max_angular_vel', 1.82)
        self.declare_parameter('linear_step', 0.01)
        self.declare_parameter('angular_step', 0.08)

        # RPM mode params
        self.declare_parameter('wheel_radius_m', 0.033)
        self.declare_parameter('wheel_separation_m', 0.287)
        self.declare_parameter('rpm_step', 5.0)
        self.declare_parameter('rpm_limit', 100.0)
        self.declare_parameter('start_mode', 'cmd_vel')

        self._hz = float(self.get_parameter('hz').value)
        cmd_topic = str(self.get_parameter('cmd_vel_topic').value)

        self._max_linear = float(self.get_parameter('max_linear_vel').value)
        self._max_angular = float(self.get_parameter('max_angular_vel').value)
        self._linear_step = float(self.get_parameter('linear_step').value)
        self._angular_step = float(self.get_parameter('angular_step').value)

        self._wheel_radius = float(self.get_parameter('wheel_radius_m').value)
        self._wheel_separation = float(self.get_parameter('wheel_separation_m').value)
        self._rpm_step = float(self.get_parameter('rpm_step').value)
        self._rpm_limit = float(self.get_parameter('rpm_limit').value)

        start_mode = str(self.get_parameter('start_mode').value).strip().lower()
        self._control_mode = self.MODE_RPM if start_mode == self.MODE_RPM else self.MODE_CMD_VEL

        self._pub_cmd = self.create_publisher(Twist, cmd_topic, 10)

        self._mode_holder = 0

        self._linear = 0.0
        self._angular = 0.0

        self._left_rpm = 0.0
        self._right_rpm = 0.0

        self.get_logger().info(
            BColors.OKGREEN +
            f'teleop_turtlebot_keyboard started. Publishing to {cmd_topic}. ' +
            f'Initial mode: {self._control_mode}' +
            BColors.ENDC
        )

    @staticmethod
    def _clamp(value, vmin, vmax):
        return max(vmin, min(vmax, value))

    @staticmethod
    def _rpm_to_rad_per_sec(rpm):
        return rpm * 2.0 * math.pi / 60.0

    def _toggle_mode(self):
        self._control_mode = self.MODE_RPM if self._control_mode == self.MODE_CMD_VEL else self.MODE_CMD_VEL

    def _stop_all(self):
        self._linear = 0.0
        self._angular = 0.0
        self._left_rpm = 0.0
        self._right_rpm = 0.0

    def _handle_cmd_vel_key(self, keycode):
        if keycode == curses.KEY_UP:
            self._linear += self._linear_step
        elif keycode == curses.KEY_DOWN:
            self._linear -= self._linear_step
        elif keycode == curses.KEY_LEFT:
            self._angular += self._angular_step
        elif keycode == curses.KEY_RIGHT:
            self._angular -= self._angular_step
        else:
            return False

        self._linear = self._clamp(self._linear, -self._max_linear, self._max_linear)
        self._angular = self._clamp(self._angular, -self._max_angular, self._max_angular)
        return True

    def _handle_rpm_key(self, keycode):
        if keycode == curses.KEY_UP:
            self._left_rpm += self._rpm_step
            self._right_rpm += self._rpm_step
        elif keycode == curses.KEY_DOWN:
            self._left_rpm -= self._rpm_step
            self._right_rpm -= self._rpm_step
        elif keycode == curses.KEY_LEFT:
            self._left_rpm -= self._rpm_step
            self._right_rpm += self._rpm_step
        elif keycode == curses.KEY_RIGHT:
            self._left_rpm += self._rpm_step
            self._right_rpm -= self._rpm_step
        elif keycode == ord('r'):
            self._left_rpm = 0.0
            self._right_rpm = 0.0
        else:
            return False

        self._left_rpm = self._clamp(self._left_rpm, -self._rpm_limit, self._rpm_limit)
        self._right_rpm = self._clamp(self._right_rpm, -self._rpm_limit, self._rpm_limit)
        return True

    def _key_pressed(self, keycode):
        if keycode == ord('q'):
            self.get_logger().info(BColors.OKBLUE + 'Shutdown requested by user.' + BColors.ENDC)
            rclpy.shutdown()
            return True

        if keycode == ord(' '):
            self._stop_all()
            return True

        if keycode == ord('m'):
            self._toggle_mode()
            return True

        if keycode == ord('z'):
            self._rpm_step = max(1.0, self._rpm_step - 1.0)
            return True

        if keycode == ord('x'):
            self._rpm_step = min(50.0, self._rpm_step + 1.0)
            return True

        if keycode == ord('a'):
            self._max_linear = max(0.01, self._max_linear - 0.01)
            self._linear = self._clamp(self._linear, -self._max_linear, self._max_linear)
            return True

        if keycode == ord('s'):
            self._max_linear = min(2.0, self._max_linear + 0.01)
            self._linear = self._clamp(self._linear, -self._max_linear, self._max_linear)
            return True

        if keycode == ord('d'):
            self._max_angular = max(0.05, self._max_angular - 0.05)
            self._angular = self._clamp(self._angular, -self._max_angular, self._max_angular)
            return True

        if keycode == ord('f'):
            self._max_angular = min(6.0, self._max_angular + 0.05)
            self._angular = self._clamp(self._angular, -self._max_angular, self._max_angular)
            return True

        if keycode == ord('1'):
            self._mode_holder = 1
            return True
        if keycode == ord('2'):
            self._mode_holder = 2
            return True
        if keycode == ord('3'):
            self._mode_holder = 3
            return True

        if self._control_mode == self.MODE_CMD_VEL:
            return self._handle_cmd_vel_key(keycode)

        return self._handle_rpm_key(keycode)

    def _key_released_decay(self, mode):
        if self._control_mode == self.MODE_CMD_VEL:
            if mode == 1:
                if self._linear > 0.0:
                    self._linear = max(0.0, self._linear - self._linear_step)
                elif self._linear < 0.0:
                    self._linear = min(0.0, self._linear + self._linear_step)

            if mode in (1, 3):
                if self._angular > 0.0:
                    self._angular = max(0.0, self._angular - self._angular_step)
                elif self._angular < 0.0:
                    self._angular = min(0.0, self._angular + self._angular_step)
        else:
            if mode in (1, 3):
                if self._left_rpm > 0.0:
                    self._left_rpm = max(0.0, self._left_rpm - self._rpm_step)
                elif self._left_rpm < 0.0:
                    self._left_rpm = min(0.0, self._left_rpm + self._rpm_step)

                if self._right_rpm > 0.0:
                    self._right_rpm = max(0.0, self._right_rpm - self._rpm_step)
                elif self._right_rpm < 0.0:
                    self._right_rpm = min(0.0, self._right_rpm + self._rpm_step)

    def _compute_twist(self):
        msg = Twist()

        if self._control_mode == self.MODE_CMD_VEL:
            msg.linear.x = float(self._linear)
            msg.angular.z = float(self._angular)
            return msg, self._linear, self._angular, None

        wl = self._rpm_to_rad_per_sec(self._left_rpm)
        wr = self._rpm_to_rad_per_sec(self._right_rpm)

        linear_cmd = (self._wheel_radius / 2.0) * (wl + wr)
        angular_cmd = (self._wheel_radius / self._wheel_separation) * (wr - wl)

        msg.linear.x = float(linear_cmd)
        msg.angular.z = float(angular_cmd)
        return msg, linear_cmd, angular_cmd, (self._left_rpm, self._right_rpm)

    def _publish(self):
        self._interface.clear()

        twist, linear_cmd, angular_cmd, rpm_pair = self._compute_twist()

        self._interface.write_line(
            1,
            f'Mode: {self._control_mode}   Decay mode: {self._mode_holder}   Topic: {self._pub_cmd.topic}'
        )

        if self._control_mode == self.MODE_CMD_VEL:
            self._interface.write_line(
                3,
                f'cmd_vel state -> linear.x: {self._linear:+.3f} m/s   angular.z: {self._angular:+.3f} rad/s'
            )
            self._interface.write_line(
                4,
                f'Limits -> max_linear: {self._max_linear:.3f}   max_angular: {self._max_angular:.3f}'
            )
            self._interface.write_line(
                5,
                f'Steps  -> linear_step: {self._linear_step:.3f}   angular_step: {self._angular_step:.3f}'
            )
        else:
            left_rpm, right_rpm = rpm_pair
            self._interface.write_line(
                3,
                f'RPM state -> left: {left_rpm:+.1f} RPM   right: {right_rpm:+.1f} RPM   rpm_step: {self._rpm_step:.1f}'
            )
            self._interface.write_line(
                4,
                f'Robot kinematics -> wheel_radius: {self._wheel_radius:.4f} m   wheel_separation: {self._wheel_separation:.4f} m'
            )
            self._interface.write_line(
                5,
                f'Equivalent cmd_vel -> linear.x: {linear_cmd:+.3f} m/s   angular.z: {angular_cmd:+.3f} rad/s'
            )

        self._interface.write_line(
            7,
            'Keys: arrows=drive/turn, SPACE=stop, m=toggle mode, q=quit'
        )
        self._interface.write_line(
            8,
            'RPM mode keys: z/x change RPM step, r reset RPMs'
        )
        self._interface.write_line(
            9,
            'cmd_vel mode keys: a/s max linear -, +   d/f max angular -, +'
        )
        self._interface.write_line(
            10,
            'Decay modes: 1=auto-center, 2=hold, 3=turn auto-center'
        )
        self._interface.write_line(
            12,
            f'Published -> linear.x: {twist.linear.x:+.3f}   angular.z: {twist.angular.z:+.3f}'
        )
        self._interface.refresh()

        self._pub_cmd.publish(twist)

    def run(self):
        period = 1.0 / self._hz
        last_key_time = time.time()

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.0)

            keycode = self._interface.read_key()

            if keycode is not None:
                handled = self._key_pressed(keycode)
                if handled:
                    last_key_time = time.time()
                else:
                    self._interface.beep()
            else:
                if self._mode_holder in (1, 3):
                    if time.time() - last_key_time > 0.3:
                        self._key_released_decay(self._mode_holder)

            self._publish()
            time.sleep(period)


def main(stdscr):
    rclpy.init()
    node = TeleopTurtlebot(TextWindow(stdscr))
    try:
        node.run()
    finally:
        try:
            stop = Twist()
            node._pub_cmd.publish(stop)
            time.sleep(0.05)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        if rclpy.ok():
            rclpy.shutdown()