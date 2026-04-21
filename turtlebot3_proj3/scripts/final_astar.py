#!/usr/bin/env python3

import math
import time
import heapq
import numpy as np
import cv2

# ============================================================
# Gazebo-aligned 2x planner version
# ------------------------------------------------------------
# Original project map: 4000 x 2000 mm
# Gazebo world map:     8000 x 4000 mm (2x scaled)
#
# Gazebo frame placement given by user:
# - Gazebo origin is 0.5 m left of the UPPER-LEFT map corner
# - Gazebo origin is 2.0 m below the UPPER-LEFT map corner
# - Robot spawn is at x=0.5, y=0.0 in Gazebo
#
# With a 2x-scaled 8m x 4m map, this means:
# - upper-left map corner in Gazebo = (0.5,  2.0)
# - lower-left map corner in Gazebo = (0.5, -2.0)
# - upper-right map corner in Gazebo = (8.5,  2.0)
# - lower-right map corner in Gazebo = (8.5, -2.0)
#
# Internal planning is still done in millimeters, but now on the
# FULL 2x-scaled geometry so that open-loop execution matches Gazebo.
#
# Coordinate conversion used here:
#   x_gazebo_m = 0.5 + x_mm / 1000
#   y_gazebo_m = -2.0 + y_mm / 1000
#
# and inverse:
#   x_mm = (x_gazebo_m - 0.5) * 1000
#   y_mm = (y_gazebo_m + 2.0) * 1000
# ============================================================

# waffle measurements
WHEEL_RADIUS_MM = 33.0
WHEEL_DISTANCE_MM = 287.0
ROBOT_RADIUS_MM = 150.0

# =========================
# Gazebo-aligned 2x map constants
# =========================
MAP_SCALE = 2.0
MAP_W_MM = int(4000 * MAP_SCALE)   # 8000 mm = 8 m
MAP_H_MM = int(2000 * MAP_SCALE)   # 4000 mm = 4 m

# Gazebo placement of map corners (meters)
GAZEBO_UPPER_LEFT_X_M = 0.5
GAZEBO_UPPER_LEFT_Y_M = 2.0
GAZEBO_LOWER_LEFT_X_M = 0.5
GAZEBO_LOWER_LEFT_Y_M = -2.0

# Planning resolution
XY_RES_MM = 50
# THETA_RES_DEG = 30
THETA_RES_DEG = 15
DT = 0.05
ACTION_TIME = 1.0
GOAL_THRESH_MM = 100.0

# =========================
# Open-loop action timing calibration
# =========================
# MIN_ACTION_TIME = 0.35
MIN_ACTION_TIME = 0.65
MAX_ACTION_TIME = 4.00

STRAIGHT_DISTANCE_TARGET_M = 0.24
# STRAIGHT_DISTANCE_TARGET_M = 0.54
ARC_DISTANCE_TARGET_M = 0.18
ARC_ANGLE_TARGET_DEG = 18.0
PIVOT_ANGLE_TARGET_DEG = 24.0

# STRAIGHT_TIME_GAIN = 1.25
STRAIGHT_TIME_GAIN = 2.25
ARC_TIME_GAIN = 1.00
PIVOT_TIME_GAIN = 0.90

# Display scale for OpenCV visualization
DISPLAY_SCALE = 0.10   # 8000x4000 mm -> 800x400 px


# =========================
# Utility
# =========================
def wrap_angle_deg(theta_deg):
    return theta_deg % 360.0


def wrap_angle_rad(theta_rad):
    return theta_rad % (2.0 * math.pi)


def rpm_to_rad_per_sec(rpm):
    return rpm * 2.0 * math.pi / 60.0


def world_to_img(x_mm, y_mm):
    """Map world coordinates (origin bottom-left) to image coordinates."""
    px = int(round(x_mm * DISPLAY_SCALE))
    py = int(round((MAP_H_MM - y_mm) * DISPLAY_SCALE))
    return px, py


def gazebo_to_planner_mm(x_gz_m, y_gz_m):
    """
    Gazebo -> planner conversion for the 2x-scaled map.

    Planner frame:
      - origin at bottom-left of map
      - units in mm
      - +x right, +y up

    Gazebo frame:
      - map lower-left corner is at (0.5, -2.0)
      - units in meters
      - +x right, +y up
    """
    x_mm = (x_gz_m - GAZEBO_LOWER_LEFT_X_M) * 1000.0
    y_mm = (y_gz_m - GAZEBO_LOWER_LEFT_Y_M) * 1000.0
    return x_mm, y_mm


def planner_mm_to_gazebo(x_mm, y_mm):
    x_gz_m = GAZEBO_LOWER_LEFT_X_M + x_mm / 1000.0
    y_gz_m = GAZEBO_LOWER_LEFT_Y_M + y_mm / 1000.0
    return x_gz_m, y_gz_m


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def primitive_action_time(ul_rpm, ur_rpm):
    """
    Open-loop action duration derived from wheel kinematics.
    Shared by planner simulation and Gazebo execution.
    """
    R = 0.033
    L = 0.287

    wl = rpm_to_rad_per_sec(ul_rpm)
    wr = rpm_to_rad_per_sec(ur_rpm)

    v = (R / 2.0) * (wl + wr)
    w = (R / L) * (wr - wl)

    abs_v = abs(v)
    abs_w = abs(w)

    if abs_w < 1e-6:
        if abs_v < 1e-6:
            return MIN_ACTION_TIME
        t = STRAIGHT_DISTANCE_TARGET_M / abs_v
        t *= STRAIGHT_TIME_GAIN
        return clamp(t, MIN_ACTION_TIME, MAX_ACTION_TIME)

    if abs(ul_rpm) < 1e-6 or abs(ur_rpm) < 1e-6:
        target_angle_rad = math.radians(PIVOT_ANGLE_TARGET_DEG)
        t = target_angle_rad / abs_w
        t *= PIVOT_TIME_GAIN
        return clamp(t, MIN_ACTION_TIME, MAX_ACTION_TIME)

    target_dist_time = ARC_DISTANCE_TARGET_M / max(abs_v, 1e-6)
    target_angle_time = math.radians(ARC_ANGLE_TARGET_DEG) / max(abs_w, 1e-6)
    t = max(target_dist_time, target_angle_time)
    t *= ARC_TIME_GAIN
    return clamp(t, MIN_ACTION_TIME, MAX_ACTION_TIME)


# =========================
# Obstacle helpers
# =========================
def point_in_rect(x, y, xmin, xmax, ymin, ymax):
    return xmin <= x <= xmax and ymin <= y <= ymax


def point_in_rotated_rect(x, y, cx, cy, length, width, angle_deg):
    theta = math.radians(angle_deg)
    dx = x - cx
    dy = y - cy

    # Rotate point into rectangle frame
    xr = dx * math.cos(theta) + dy * math.sin(theta)
    yr = -dx * math.sin(theta) + dy * math.cos(theta)

    return abs(xr) <= length / 2.0 and abs(yr) <= width / 2.0


def build_obstacle_check(clearance_mm):
    """
    Returns a closure obstacle_free(x, y) for the 2x-scaled project map.

    Modification for Gazebo:
    - the LEFT OUTER WALL now has a centered opening of 1.0 m in Gazebo
      (= 1000 mm in this 2x planner map)
    - that opening is centered at the robot spawn height, y = 0.0 m in Gazebo,
      which corresponds to planner y = 2000 mm
    """
    inflate = ROBOT_RADIUS_MM + clearance_mm

    # Original project-map approximations, now scaled by 2x.
    square_side = 400.0 * MAP_SCALE

    squares = [
        (420.0 * MAP_SCALE,  450.0 * MAP_SCALE, square_side),
        (1335.0 * MAP_SCALE, 1550.0 * MAP_SCALE, square_side),
        (2200.0 * MAP_SCALE, 1740.0 * MAP_SCALE, square_side),
    ]

    bars = [
        (800.0 * MAP_SCALE,  1400.0 * MAP_SCALE, 1400.0 * MAP_SCALE, 50.0 * MAP_SCALE, -60.0),
        (1700.0 * MAP_SCALE,  500.0 * MAP_SCALE, 1200.0 * MAP_SCALE, 50.0 * MAP_SCALE,  60.0),
        (2925.0 * MAP_SCALE, 1275.0 * MAP_SCALE, 1450.0 * MAP_SCALE, 50.0 * MAP_SCALE,  90.0),
    ]

    # Left wall opening: 1.0 m in Gazebo = 1000 mm in planner.
    left_gap_height_mm = 1000.0
    left_gap_center_y_mm = 2000.0   # spawn y = 0.0 m in Gazebo
    left_gap_ymin = left_gap_center_y_mm - left_gap_height_mm / 2.0
    left_gap_ymax = left_gap_center_y_mm + left_gap_height_mm / 2.0

    def obstacle_free(x, y):
        # Right, bottom, and top boundaries are still closed and inflated inward.
        if x > (MAP_W_MM - inflate):
            return False
        if y < inflate or y > (MAP_H_MM - inflate):
            return False

        # Left boundary is open only inside the requested gap.
        # Outside the gap, it behaves like a normal inflated wall.
        if x < inflate:
            if y < left_gap_ymin + inflate or y > left_gap_ymax - inflate:
                return False

        for cx, cy, side in squares:
            half = side / 2.0 + inflate
            if point_in_rect(x, y, cx - half, cx + half, cy - half, cy + half):
                return False

        for cx, cy, length, width, angle_deg in bars:
            if point_in_rotated_rect(
                x, y,
                cx, cy,
                length + 2.0 * inflate,
                width + 2.0 * inflate,
                angle_deg,
            ):
                return False

        return True

    return obstacle_free, squares, bars, inflate, (left_gap_ymin, left_gap_ymax)


# =========================
# Differential drive motion primitive
# =========================
def move_with_rpms(node, ul_rpm, ur_rpm, obstacle_free):
    """
    node = (x_mm, y_mm, theta_deg)
    returns:
        (x_new, y_new, theta_new_deg), distance_cost, curve_points
    or None if collision occurs during integration
    """
    x, y, theta_deg = node
    theta = math.radians(theta_deg)

    ul = rpm_to_rad_per_sec(ul_rpm)
    ur = rpm_to_rad_per_sec(ur_rpm)

    t = 0.0
    D = 0.0
    curve_points = [(x, y)]
    action_time = primitive_action_time(ul_rpm, ur_rpm)

    while t < action_time:
        t += DT

        x_prev = x
        y_prev = y

        x += 0.5 * WHEEL_RADIUS_MM * (ul + ur) * math.cos(theta) * DT
        y += 0.5 * WHEEL_RADIUS_MM * (ul + ur) * math.sin(theta) * DT
        theta += (WHEEL_RADIUS_MM / WHEEL_DISTANCE_MM) * (ur - ul) * DT

        if not obstacle_free(x, y):
            return None

        D += np.hypot(x - x_prev, y - y_prev)
        curve_points.append((x, y))

    theta_deg_new = wrap_angle_deg(math.degrees(theta))
    return (x, y, theta_deg_new), D, curve_points


# =========================
# A* bookkeeping
# =========================
def get_index(node):
    x, y, theta_deg = node
    ix = int(round(x / XY_RES_MM))
    iy = int(round(y / XY_RES_MM))
    it = int(round(wrap_angle_deg(theta_deg) / THETA_RES_DEG)) % int(360 / THETA_RES_DEG)
    return ix, iy, it


def heuristic(node, goal):
    return np.hypot(node[0] - goal[0], node[1] - goal[1])


def reached_goal(node, goal):
    return np.hypot(node[0] - goal[0], node[1] - goal[1]) <= GOAL_THRESH_MM


# =========================
# Map drawing
# =========================
def make_base_map(squares, bars, inflate, left_gap=None):
    w = int(MAP_W_MM * DISPLAY_SCALE)
    h = int(MAP_H_MM * DISPLAY_SCALE)
    img = np.ones((h, w, 3), dtype=np.uint8) * 255

    # Border with optional left-wall gap
    if left_gap is None:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 0), 2)
    else:
        gap_ymin, gap_ymax = left_gap
        # top wall
        cv2.line(img, (0, 0), (w - 1, 0), (0, 0, 0), 2)
        # bottom wall
        cv2.line(img, (0, h - 1), (w - 1, h - 1), (0, 0, 0), 2)
        # right wall
        cv2.line(img, (w - 1, 0), (w - 1, h - 1), (0, 0, 0), 2)

        # left wall split into two segments around the opening
        p_top = world_to_img(0.0, gap_ymax)
        p_bot = world_to_img(0.0, gap_ymin)
        cv2.line(img, (0, 0), (0, p_top[1]), (0, 0, 0), 2)
        cv2.line(img, (0, p_bot[1]), (0, h - 1), (0, 0, 0), 2)

    # Draw inflated squares
    for cx, cy, side in squares:
        half = side / 2.0 + inflate
        p1 = world_to_img(cx - half, cy - half)
        p2 = world_to_img(cx + half, cy + half)
        cv2.rectangle(
            img,
            (min(p1[0], p2[0]), min(p1[1], p2[1])),
            (max(p1[0], p2[0]), max(p1[1], p2[1])),
            (0, 0, 0),
            -1,
        )

    # Draw inflated bars
    for cx, cy, length, width, angle_deg in bars:
        box = cv2.boxPoints((
            (cx * DISPLAY_SCALE, (MAP_H_MM - cy) * DISPLAY_SCALE),
            ((length + 2.0 * inflate) * DISPLAY_SCALE, (width + 2.0 * inflate) * DISPLAY_SCALE),
            -angle_deg,
        ))
        box = np.round(box).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(img, [box], (0, 0, 0))

    # End line at right map edge
    endline_x = int(MAP_W_MM * DISPLAY_SCALE)
    cv2.line(img, (endline_x - 3, 0), (endline_x - 3, h - 1), (255, 0, 0), 3)

    return img


# =========================
# A*
# =========================
def astar(start, goal, rpm1, rpm2, obstacle_free, squares, bars, inflate, left_gap=None):
    # actions = [
    #     (0, rpm1),
    #     (rpm1, 0),
    #     (rpm1, rpm1),
    #     (0, rpm2),
    #     (rpm2, 0),
    #     (rpm2, rpm2),
    #     (rpm1, rpm2),
    #     (rpm2, rpm1),
    # ]
    actions = [
        (0, rpm1),
        (rpm1, 0),
        (rpm1, rpm1),
        (0, rpm2),
        (rpm2, 0),
        (rpm2, rpm1),
        (rpm1, rpm2),
        (rpm2, rpm2),
    ]
    

    open_list = []
    parent = {}
    parent_action = {}
    cost_to_come = {}
    visited = set()

    start_idx = get_index(start)
    cost_to_come[start_idx] = 0.0
    heapq.heappush(open_list, (heuristic(start, goal), 0.0, start))

    img = make_base_map(squares, bars, inflate, left_gap)

    while open_list:
        f, g, current = heapq.heappop(open_list)
        cidx = get_index(current)

        if cidx in visited:
            continue
        visited.add(cidx)

        if reached_goal(current, goal):
            return parent, parent_action, current, img

        for ul, ur in actions:
            result = move_with_rpms(current, ul, ur, obstacle_free)
            if result is None:
                continue

            nxt, edge_cost, curve_points = result
            nidx = get_index(nxt)

            if nidx in visited:
                continue

            new_g = g + edge_cost

            if (nidx not in cost_to_come) or (new_g < cost_to_come[nidx]):
                cost_to_come[nidx] = new_g
                new_f = new_g + heuristic(nxt, goal)

                parent[nidx] = cidx
                parent_action[nidx] = (ul, ur, curve_points, nxt)
                heapq.heappush(open_list, (new_f, new_g, nxt))

                for i in range(len(curve_points) - 1):
                    p1 = world_to_img(curve_points[i][0], curve_points[i][1])
                    p2 = world_to_img(curve_points[i + 1][0], curve_points[i + 1][1])
                    cv2.line(img, p1, p2, (150, 150, 150), 1)

        cv2.imshow("A* Exploration", img)
        cv2.waitKey(1)

    return None, None, None, img


def reconstruct_path(start, final_node, parent, parent_action):
    path_actions = []
    current_idx = get_index(final_node)
    start_idx = get_index(start)

    while current_idx != start_idx:
        action = parent_action[current_idx]
        path_actions.append(action)
        current_idx = parent[current_idx]

    path_actions.reverse()
    return path_actions


def draw_final_path(img, start, goal, path_actions):
    out = img.copy()

    for ul, ur, curve_points, nxt in path_actions:
        for i in range(len(curve_points) - 1):
            p1 = world_to_img(curve_points[i][0], curve_points[i][1])
            p2 = world_to_img(curve_points[i + 1][0], curve_points[i + 1][1])
            cv2.line(out, p1, p2, (0, 0, 255), 2)

    sp = world_to_img(start[0], start[1])
    gp = world_to_img(goal[0], goal[1])
    cv2.circle(out, sp, 6, (0, 255, 0), -1)
    cv2.circle(out, gp, 6, (255, 0, 0), -1)

    arrow_len = 200.0
    hx = start[0] + arrow_len * math.cos(math.radians(start[2]))
    hy = start[1] + arrow_len * math.sin(math.radians(start[2]))
    hp = world_to_img(hx, hy)
    cv2.arrowedLine(out, sp, hp, (0, 128, 0), 2)

    return out


# =========================
# ROS open-loop execution
# =========================
def execute_in_gazebo(path_actions):
    import time
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist

    class CmdVelPublisher(Node):
        def __init__(self):
            super().__init__('astar_path_executor')
            self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

    WHEEL_RADIUS_M = 0.033
    WHEEL_DISTANCE_M = 0.287

    rclpy.init()
    node = CmdVelPublisher()

    try:
        print("execute_in_gazebo called")
        print("num actions:", len(path_actions))
        # time.sleep(1.0)
        time.sleep(1.0)

        for i, action in enumerate(path_actions):
            ul_rpm = action[0]
            ur_rpm = action[1]

            ul = ul_rpm * 2.0 * math.pi / 60.0
            ur = ur_rpm * 2.0 * math.pi / 60.0

            msg = Twist()
            msg.linear.x = (WHEEL_RADIUS_M / 2.0) * (ul + ur)
            msg.angular.z = (WHEEL_RADIUS_M / WHEEL_DISTANCE_M) * (ur - ul)

            action_time = primitive_action_time(ul_rpm, ur_rpm)

            print(
                f"Publishing action {i}: UL={ul_rpm}, UR={ur_rpm}, "
                f"v={msg.linear.x:.3f}, w={msg.angular.z:.3f}, T={action_time:.2f}s"
            )

            t0 = time.time()
            while time.time() - t0 < action_time:
                node.pub.publish(msg)
                rclpy.spin_once(node, timeout_sec=0.0)
                # time.sleep(0.05)
                time.sleep(0.01)

            node.pub.publish(Twist())
            # time.sleep(0.05)
            time.sleep(0.01)

        node.pub.publish(Twist())

    finally:
        node.destroy_node()
        rclpy.shutdown()


# =========================
# Inputs
# =========================
def get_float(prompt, default=None):
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("Invalid number. Try again.")


def print_gazebo_alignment_summary():
    print("\n=== Gazebo-aligned 2x planner ===")
    print(f"Planner map size: {MAP_W_MM} mm x {MAP_H_MM} mm  ({MAP_W_MM/1000:.1f} m x {MAP_H_MM/1000:.1f} m)")
    print("Gazebo map corners assumed:")
    print("  upper-left  = (0.5,  2.0) m")
    print("  lower-left  = (0.5, -2.0) m")
    print("  upper-right = (8.5,  2.0) m")
    print("  lower-right = (8.5, -2.0) m")
    print("Robot spawn given by user: (0.5, 0.0) m")
    print("This corresponds to planner start position: (0 mm, 2000 mm)")
    print("Left wall modification: 1.0 m opening in Gazebo, centered at spawn height")
    print("Heading convention: 0 deg = +x (right), 90 deg = +y (up)")
    print("Open-loop timing: adaptive per action, using wheel kinematics\n")


def get_inputs():
    print_gazebo_alignment_summary()
    print("Enter START / GOAL in GAZEBO METERS and heading in DEGREES.\n")

    sx_gz = get_float("Start x in Gazebo (m) [default 0.5]: ", default=0.5)
    sy_gz = get_float("Start y in Gazebo (m) [default 0.0]: ", default=0.0)
    st = get_float("Start theta (deg) [default 0]: ", default=0.0)

    gx_gz = get_float("Goal x in Gazebo (m): ")
    gy_gz = get_float("Goal y in Gazebo (m): ")
    gt = get_float("Goal theta (deg, can be 0 if unused) [default 0]: ", default=0.0)

    rpm1 = get_float("RPM1: ")
    rpm2 = get_float("RPM2: ")
    clearance = get_float("Clearance (mm): ")

    execute_flag = input("Execute in Gazebo after planning? (y/n): ").strip().lower() == 'y'

    sx_mm, sy_mm = gazebo_to_planner_mm(sx_gz, sy_gz)
    gx_mm, gy_mm = gazebo_to_planner_mm(gx_gz, gy_gz)

    start = (sx_mm, sy_mm, wrap_angle_deg(st))
    goal = (gx_mm, gy_mm, wrap_angle_deg(gt))

    return start, goal, rpm1, rpm2, clearance, execute_flag, (sx_gz, sy_gz, st), (gx_gz, gy_gz, gt)


# =========================
# Main
# =========================
def main():
    (
        start,
        goal,
        rpm1,
        rpm2,
        clearance,
        execute_flag,
        start_gz,
        goal_gz,
    ) = get_inputs()

    obstacle_free, squares, bars, inflate, left_gap = build_obstacle_check(clearance)

    while not obstacle_free(start[0], start[1]):
        print("Start is in obstacle space or invalid. Enter again.")
        sx_gz = get_float("Start x in Gazebo (m): ")
        sy_gz = get_float("Start y in Gazebo (m): ")
        st = get_float("Start theta (deg): ")
        sx_mm, sy_mm = gazebo_to_planner_mm(sx_gz, sy_gz)
        start = (sx_mm, sy_mm, wrap_angle_deg(st))

    while not obstacle_free(goal[0], goal[1]):
        print("Goal is in obstacle space or invalid. Enter again.")
        gx_gz = get_float("Goal x in Gazebo (m): ")
        gy_gz = get_float("Goal y in Gazebo (m): ")
        gt = get_float("Goal theta (deg, can be 0 if unused): ")
        gx_mm, gy_mm = gazebo_to_planner_mm(gx_gz, gy_gz)
        goal = (gx_mm, gy_mm, wrap_angle_deg(gt))

    print("\nConverted planner-frame coordinates (mm):")
    print(f"  Start: ({start[0]:.1f}, {start[1]:.1f}, {start[2]:.1f} deg)")
    print(f"  Goal : ({goal[0]:.1f}, {goal[1]:.1f}, {goal[2]:.1f} deg)")

    parent, parent_action, final_node, exploration_img = astar(
        start, goal, rpm1, rpm2, obstacle_free, squares, bars, inflate, left_gap
    )

    if final_node is None:
        print("No path found.")
        cv2.imshow("A* Exploration", exploration_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    path_actions = reconstruct_path(start, final_node, parent, parent_action)

    print("\nPath found.")
    print(f"Final reached node in planner frame: {final_node}")
    fx_gz, fy_gz = planner_mm_to_gazebo(final_node[0], final_node[1])
    print(f"Final reached node in Gazebo frame: ({fx_gz:.3f}, {fy_gz:.3f}, {final_node[2]:.2f} deg)")
    print(f"Number of action primitives: {len(path_actions)}")

    print("\nAction sequence [UL, UR] RPM:")
    for i, (ul, ur, _, nxt) in enumerate(path_actions):
        nx_gz, ny_gz = planner_mm_to_gazebo(nxt[0], nxt[1])
        action_time = primitive_action_time(ul, ur)
        print(
            f"{i:03d}: [{ul:.2f}, {ur:.2f}]  T={action_time:.2f}s -> "
            f"planner=({nxt[0]:.2f}, {nxt[1]:.2f}, {nxt[2]:.2f} deg), "
            f"gazebo=({nx_gz:.3f}, {ny_gz:.3f}, {nxt[2]:.2f} deg)"
        )

    final_img = draw_final_path(exploration_img, start, goal, path_actions)
    cv2.imshow("Optimal Path", final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if execute_flag:
        execute_in_gazebo(path_actions)


if __name__ == "__main__":
    main()#!/usr/bin/env python3

import math
import time
import heapq
import numpy as np
import cv2

# ============================================================
# Gazebo-aligned 2x planner version
# ------------------------------------------------------------
# Original project map: 4000 x 2000 mm
# Gazebo world map:     8000 x 4000 mm (2x scaled)
#
# Gazebo frame placement given by user:
# - Gazebo origin is 0.5 m left of the UPPER-LEFT map corner
# - Gazebo origin is 2.0 m below the UPPER-LEFT map corner
# - Robot spawn is at x=0.5, y=0.0 in Gazebo
#
# With a 2x-scaled 8m x 4m map, this means:
# - upper-left map corner in Gazebo = (0.5,  2.0)
# - lower-left map corner in Gazebo = (0.5, -2.0)
# - upper-right map corner in Gazebo = (8.5,  2.0)
# - lower-right map corner in Gazebo = (8.5, -2.0)
#
# Internal planning is still done in millimeters, but now on the
# FULL 2x-scaled geometry so that open-loop execution matches Gazebo.
#
# Coordinate conversion used here:
#   x_gazebo_m = 0.5 + x_mm / 1000
#   y_gazebo_m = -2.0 + y_mm / 1000
#
# and inverse:
#   x_mm = (x_gazebo_m - 0.5) * 1000
#   y_mm = (y_gazebo_m + 2.0) * 1000
# ============================================================

# waffle measurements
WHEEL_RADIUS_MM = 33.0
WHEEL_DISTANCE_MM = 287.0
ROBOT_RADIUS_MM = 150.0

# =========================
# Gazebo-aligned 2x map constants
# =========================
MAP_SCALE = 2.0
MAP_W_MM = int(4000 * MAP_SCALE)   # 8000 mm = 8 m
MAP_H_MM = int(2000 * MAP_SCALE)   # 4000 mm = 4 m

# Gazebo placement of map corners (meters)
GAZEBO_UPPER_LEFT_X_M = 0.5
GAZEBO_UPPER_LEFT_Y_M = 2.0
GAZEBO_LOWER_LEFT_X_M = 0.5
GAZEBO_LOWER_LEFT_Y_M = -2.0

# Planning resolution
XY_RES_MM = 50
# THETA_RES_DEG = 30
THETA_RES_DEG = 15
DT = 0.05
ACTION_TIME = 1.0
GOAL_THRESH_MM = 100.0

# =========================
# Open-loop action timing calibration
# =========================
# MIN_ACTION_TIME = 0.35
MIN_ACTION_TIME = 0.65
MAX_ACTION_TIME = 4.00

STRAIGHT_DISTANCE_TARGET_M = 0.24
# STRAIGHT_DISTANCE_TARGET_M = 0.54
ARC_DISTANCE_TARGET_M = 0.18
ARC_ANGLE_TARGET_DEG = 18.0
PIVOT_ANGLE_TARGET_DEG = 24.0

# STRAIGHT_TIME_GAIN = 1.25
STRAIGHT_TIME_GAIN = 2.25
ARC_TIME_GAIN = 1.00
PIVOT_TIME_GAIN = 0.90

# Display scale for OpenCV visualization
DISPLAY_SCALE = 0.10   # 8000x4000 mm -> 800x400 px


# =========================
# Utility
# =========================
def wrap_angle_deg(theta_deg):
    return theta_deg % 360.0


def wrap_angle_rad(theta_rad):
    return theta_rad % (2.0 * math.pi)


def rpm_to_rad_per_sec(rpm):
    return rpm * 2.0 * math.pi / 60.0


def world_to_img(x_mm, y_mm):
    """Map world coordinates (origin bottom-left) to image coordinates."""
    px = int(round(x_mm * DISPLAY_SCALE))
    py = int(round((MAP_H_MM - y_mm) * DISPLAY_SCALE))
    return px, py


def gazebo_to_planner_mm(x_gz_m, y_gz_m):
    """
    Gazebo -> planner conversion for the 2x-scaled map.

    Planner frame:
      - origin at bottom-left of map
      - units in mm
      - +x right, +y up

    Gazebo frame:
      - map lower-left corner is at (0.5, -2.0)
      - units in meters
      - +x right, +y up
    """
    x_mm = (x_gz_m - GAZEBO_LOWER_LEFT_X_M) * 1000.0
    y_mm = (y_gz_m - GAZEBO_LOWER_LEFT_Y_M) * 1000.0
    return x_mm, y_mm


def planner_mm_to_gazebo(x_mm, y_mm):
    x_gz_m = GAZEBO_LOWER_LEFT_X_M + x_mm / 1000.0
    y_gz_m = GAZEBO_LOWER_LEFT_Y_M + y_mm / 1000.0
    return x_gz_m, y_gz_m


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def primitive_action_time(ul_rpm, ur_rpm):
    """
    Open-loop action duration derived from wheel kinematics.
    Shared by planner simulation and Gazebo execution.
    """
    R = 0.033
    L = 0.287

    wl = rpm_to_rad_per_sec(ul_rpm)
    wr = rpm_to_rad_per_sec(ur_rpm)

    v = (R / 2.0) * (wl + wr)
    w = (R / L) * (wr - wl)

    abs_v = abs(v)
    abs_w = abs(w)

    if abs_w < 1e-6:
        if abs_v < 1e-6:
            return MIN_ACTION_TIME
        t = STRAIGHT_DISTANCE_TARGET_M / abs_v
        t *= STRAIGHT_TIME_GAIN
        return clamp(t, MIN_ACTION_TIME, MAX_ACTION_TIME)

    if abs(ul_rpm) < 1e-6 or abs(ur_rpm) < 1e-6:
        target_angle_rad = math.radians(PIVOT_ANGLE_TARGET_DEG)
        t = target_angle_rad / abs_w
        t *= PIVOT_TIME_GAIN
        return clamp(t, MIN_ACTION_TIME, MAX_ACTION_TIME)

    target_dist_time = ARC_DISTANCE_TARGET_M / max(abs_v, 1e-6)
    target_angle_time = math.radians(ARC_ANGLE_TARGET_DEG) / max(abs_w, 1e-6)
    t = max(target_dist_time, target_angle_time)
    t *= ARC_TIME_GAIN
    return clamp(t, MIN_ACTION_TIME, MAX_ACTION_TIME)


# =========================
# Obstacle helpers
# =========================
def point_in_rect(x, y, xmin, xmax, ymin, ymax):
    return xmin <= x <= xmax and ymin <= y <= ymax


def point_in_rotated_rect(x, y, cx, cy, length, width, angle_deg):
    theta = math.radians(angle_deg)
    dx = x - cx
    dy = y - cy

    # Rotate point into rectangle frame
    xr = dx * math.cos(theta) + dy * math.sin(theta)
    yr = -dx * math.sin(theta) + dy * math.cos(theta)

    return abs(xr) <= length / 2.0 and abs(yr) <= width / 2.0


def build_obstacle_check(clearance_mm):
    """
    Returns a closure obstacle_free(x, y) for the 2x-scaled project map.

    Modification for Gazebo:
    - the LEFT OUTER WALL now has a centered opening of 1.0 m in Gazebo
      (= 1000 mm in this 2x planner map)
    - that opening is centered at the robot spawn height, y = 0.0 m in Gazebo,
      which corresponds to planner y = 2000 mm
    """
    inflate = ROBOT_RADIUS_MM + clearance_mm

    # Original project-map approximations, now scaled by 2x.
    square_side = 400.0 * MAP_SCALE

    squares = [
        (420.0 * MAP_SCALE,  450.0 * MAP_SCALE, square_side),
        (1335.0 * MAP_SCALE, 1550.0 * MAP_SCALE, square_side),
        (2200.0 * MAP_SCALE, 1740.0 * MAP_SCALE, square_side),
    ]

    bars = [
        (800.0 * MAP_SCALE,  1400.0 * MAP_SCALE, 1400.0 * MAP_SCALE, 50.0 * MAP_SCALE, -60.0),
        (1700.0 * MAP_SCALE,  500.0 * MAP_SCALE, 1200.0 * MAP_SCALE, 50.0 * MAP_SCALE,  60.0),
        (2925.0 * MAP_SCALE, 1275.0 * MAP_SCALE, 1450.0 * MAP_SCALE, 50.0 * MAP_SCALE,  90.0),
    ]

    # Left wall opening: 1.0 m in Gazebo = 1000 mm in planner.
    left_gap_height_mm = 1000.0
    left_gap_center_y_mm = 2000.0   # spawn y = 0.0 m in Gazebo
    left_gap_ymin = left_gap_center_y_mm - left_gap_height_mm / 2.0
    left_gap_ymax = left_gap_center_y_mm + left_gap_height_mm / 2.0

    def obstacle_free(x, y):
        # Right, bottom, and top boundaries are still closed and inflated inward.
        if x > (MAP_W_MM - inflate):
            return False
        if y < inflate or y > (MAP_H_MM - inflate):
            return False

        # Left boundary is open only inside the requested gap.
        # Outside the gap, it behaves like a normal inflated wall.
        if x < inflate:
            if y < left_gap_ymin + inflate or y > left_gap_ymax - inflate:
                return False

        for cx, cy, side in squares:
            half = side / 2.0 + inflate
            if point_in_rect(x, y, cx - half, cx + half, cy - half, cy + half):
                return False

        for cx, cy, length, width, angle_deg in bars:
            if point_in_rotated_rect(
                x, y,
                cx, cy,
                length + 2.0 * inflate,
                width + 2.0 * inflate,
                angle_deg,
            ):
                return False

        return True

    return obstacle_free, squares, bars, inflate, (left_gap_ymin, left_gap_ymax)


# =========================
# Differential drive motion primitive
# =========================
def move_with_rpms(node, ul_rpm, ur_rpm, obstacle_free):
    """
    node = (x_mm, y_mm, theta_deg)
    returns:
        (x_new, y_new, theta_new_deg), distance_cost, curve_points
    or None if collision occurs during integration
    """
    x, y, theta_deg = node
    theta = math.radians(theta_deg)

    ul = rpm_to_rad_per_sec(ul_rpm)
    ur = rpm_to_rad_per_sec(ur_rpm)

    t = 0.0
    D = 0.0
    curve_points = [(x, y)]
    action_time = primitive_action_time(ul_rpm, ur_rpm)

    while t < action_time:
        t += DT

        x_prev = x
        y_prev = y

        x += 0.5 * WHEEL_RADIUS_MM * (ul + ur) * math.cos(theta) * DT
        y += 0.5 * WHEEL_RADIUS_MM * (ul + ur) * math.sin(theta) * DT
        theta += (WHEEL_RADIUS_MM / WHEEL_DISTANCE_MM) * (ur - ul) * DT

        if not obstacle_free(x, y):
            return None

        D += np.hypot(x - x_prev, y - y_prev)
        curve_points.append((x, y))

    theta_deg_new = wrap_angle_deg(math.degrees(theta))
    return (x, y, theta_deg_new), D, curve_points


# =========================
# A* bookkeeping
# =========================
def get_index(node):
    x, y, theta_deg = node
    ix = int(round(x / XY_RES_MM))
    iy = int(round(y / XY_RES_MM))
    it = int(round(wrap_angle_deg(theta_deg) / THETA_RES_DEG)) % int(360 / THETA_RES_DEG)
    return ix, iy, it


def heuristic(node, goal):
    return np.hypot(node[0] - goal[0], node[1] - goal[1])


def reached_goal(node, goal):
    return np.hypot(node[0] - goal[0], node[1] - goal[1]) <= GOAL_THRESH_MM


# =========================
# Map drawing
# =========================
def make_base_map(squares, bars, inflate, left_gap=None):
    w = int(MAP_W_MM * DISPLAY_SCALE)
    h = int(MAP_H_MM * DISPLAY_SCALE)
    img = np.ones((h, w, 3), dtype=np.uint8) * 255

    # Border with optional left-wall gap
    if left_gap is None:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 0), 2)
    else:
        gap_ymin, gap_ymax = left_gap
        # top wall
        cv2.line(img, (0, 0), (w - 1, 0), (0, 0, 0), 2)
        # bottom wall
        cv2.line(img, (0, h - 1), (w - 1, h - 1), (0, 0, 0), 2)
        # right wall
        cv2.line(img, (w - 1, 0), (w - 1, h - 1), (0, 0, 0), 2)

        # left wall split into two segments around the opening
        p_top = world_to_img(0.0, gap_ymax)
        p_bot = world_to_img(0.0, gap_ymin)
        cv2.line(img, (0, 0), (0, p_top[1]), (0, 0, 0), 2)
        cv2.line(img, (0, p_bot[1]), (0, h - 1), (0, 0, 0), 2)

    # Draw inflated squares
    for cx, cy, side in squares:
        half = side / 2.0 + inflate
        p1 = world_to_img(cx - half, cy - half)
        p2 = world_to_img(cx + half, cy + half)
        cv2.rectangle(
            img,
            (min(p1[0], p2[0]), min(p1[1], p2[1])),
            (max(p1[0], p2[0]), max(p1[1], p2[1])),
            (0, 0, 0),
            -1,
        )

    # Draw inflated bars
    for cx, cy, length, width, angle_deg in bars:
        box = cv2.boxPoints((
            (cx * DISPLAY_SCALE, (MAP_H_MM - cy) * DISPLAY_SCALE),
            ((length + 2.0 * inflate) * DISPLAY_SCALE, (width + 2.0 * inflate) * DISPLAY_SCALE),
            -angle_deg,
        ))
        box = np.round(box).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(img, [box], (0, 0, 0))

    # End line at right map edge
    endline_x = int(MAP_W_MM * DISPLAY_SCALE)
    cv2.line(img, (endline_x - 3, 0), (endline_x - 3, h - 1), (255, 0, 0), 3)

    return img


# =========================
# A*
# =========================
def astar(start, goal, rpm1, rpm2, obstacle_free, squares, bars, inflate, left_gap=None):
    # actions = [
    #     (0, rpm1),
    #     (rpm1, 0),
    #     (rpm1, rpm1),
    #     (0, rpm2),
    #     (rpm2, 0),
    #     (rpm2, rpm2),
    #     (rpm1, rpm2),
    #     (rpm2, rpm1),
    # ]
    actions = [
        (0, rpm1),
        (rpm1, 0),
        (rpm1, rpm1),
        (0, rpm2),
        (rpm2, 0),
        (rpm2, rpm1),
        (rpm1, rpm2),
        (rpm2, rpm2),
    ]
    

    open_list = []
    parent = {}
    parent_action = {}
    cost_to_come = {}
    visited = set()

    start_idx = get_index(start)
    cost_to_come[start_idx] = 0.0
    heapq.heappush(open_list, (heuristic(start, goal), 0.0, start))

    img = make_base_map(squares, bars, inflate, left_gap)

    while open_list:
        f, g, current = heapq.heappop(open_list)
        cidx = get_index(current)

        if cidx in visited:
            continue
        visited.add(cidx)

        if reached_goal(current, goal):
            return parent, parent_action, current, img

        for ul, ur in actions:
            result = move_with_rpms(current, ul, ur, obstacle_free)
            if result is None:
                continue

            nxt, edge_cost, curve_points = result
            nidx = get_index(nxt)

            if nidx in visited:
                continue

            new_g = g + edge_cost

            if (nidx not in cost_to_come) or (new_g < cost_to_come[nidx]):
                cost_to_come[nidx] = new_g
                new_f = new_g + heuristic(nxt, goal)

                parent[nidx] = cidx
                parent_action[nidx] = (ul, ur, curve_points, nxt)
                heapq.heappush(open_list, (new_f, new_g, nxt))

                for i in range(len(curve_points) - 1):
                    p1 = world_to_img(curve_points[i][0], curve_points[i][1])
                    p2 = world_to_img(curve_points[i + 1][0], curve_points[i + 1][1])
                    cv2.line(img, p1, p2, (150, 150, 150), 1)

        cv2.imshow("A* Exploration", img)
        cv2.waitKey(1)

    return None, None, None, img


def reconstruct_path(start, final_node, parent, parent_action):
    path_actions = []
    current_idx = get_index(final_node)
    start_idx = get_index(start)

    while current_idx != start_idx:
        action = parent_action[current_idx]
        path_actions.append(action)
        current_idx = parent[current_idx]

    path_actions.reverse()
    return path_actions


def draw_final_path(img, start, goal, path_actions):
    out = img.copy()

    for ul, ur, curve_points, nxt in path_actions:
        for i in range(len(curve_points) - 1):
            p1 = world_to_img(curve_points[i][0], curve_points[i][1])
            p2 = world_to_img(curve_points[i + 1][0], curve_points[i + 1][1])
            cv2.line(out, p1, p2, (0, 0, 255), 2)

    sp = world_to_img(start[0], start[1])
    gp = world_to_img(goal[0], goal[1])
    cv2.circle(out, sp, 6, (0, 255, 0), -1)
    cv2.circle(out, gp, 6, (255, 0, 0), -1)

    arrow_len = 200.0
    hx = start[0] + arrow_len * math.cos(math.radians(start[2]))
    hy = start[1] + arrow_len * math.sin(math.radians(start[2]))
    hp = world_to_img(hx, hy)
    cv2.arrowedLine(out, sp, hp, (0, 128, 0), 2)

    return out


# =========================
# ROS open-loop execution
# =========================
def execute_in_gazebo(path_actions):
    import time
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist

    class CmdVelPublisher(Node):
        def __init__(self):
            super().__init__('astar_path_executor')
            self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

    WHEEL_RADIUS_M = 0.033
    WHEEL_DISTANCE_M = 0.287

    rclpy.init()
    node = CmdVelPublisher()

    try:
        print("execute_in_gazebo called")
        print("num actions:", len(path_actions))
        # time.sleep(1.0)
        time.sleep(1.0)

        for i, action in enumerate(path_actions):
            ul_rpm = action[0]
            ur_rpm = action[1]

            ul = ul_rpm * 2.0 * math.pi / 60.0
            ur = ur_rpm * 2.0 * math.pi / 60.0

            msg = Twist()
            msg.linear.x = (WHEEL_RADIUS_M / 2.0) * (ul + ur)
            msg.angular.z = (WHEEL_RADIUS_M / WHEEL_DISTANCE_M) * (ur - ul)

            action_time = primitive_action_time(ul_rpm, ur_rpm)

            print(
                f"Publishing action {i}: UL={ul_rpm}, UR={ur_rpm}, "
                f"v={msg.linear.x:.3f}, w={msg.angular.z:.3f}, T={action_time:.2f}s"
            )

            t0 = time.time()
            while time.time() - t0 < action_time:
                node.pub.publish(msg)
                rclpy.spin_once(node, timeout_sec=0.0)
                # time.sleep(0.05)
                time.sleep(0.01)

            node.pub.publish(Twist())
            # time.sleep(0.05)
            time.sleep(0.01)

        node.pub.publish(Twist())

    finally:
        node.destroy_node()
        rclpy.shutdown()


# =========================
# Inputs
# =========================
def get_float(prompt, default=None):
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("Invalid number. Try again.")


def print_gazebo_alignment_summary():
    print("\n=== Gazebo-aligned 2x planner ===")
    print(f"Planner map size: {MAP_W_MM} mm x {MAP_H_MM} mm  ({MAP_W_MM/1000:.1f} m x {MAP_H_MM/1000:.1f} m)")
    print("Gazebo map corners assumed:")
    print("  upper-left  = (0.5,  2.0) m")
    print("  lower-left  = (0.5, -2.0) m")
    print("  upper-right = (8.5,  2.0) m")
    print("  lower-right = (8.5, -2.0) m")
    print("Robot spawn given by user: (0.5, 0.0) m")
    print("This corresponds to planner start position: (0 mm, 2000 mm)")
    print("Left wall modification: 1.0 m opening in Gazebo, centered at spawn height")
    print("Heading convention: 0 deg = +x (right), 90 deg = +y (up)")
    print("Open-loop timing: adaptive per action, using wheel kinematics\n")


def get_inputs():
    print_gazebo_alignment_summary()
    print("Enter START / GOAL in GAZEBO METERS and heading in DEGREES.\n")

    sx_gz = get_float("Start x in Gazebo (m) [default 0.5]: ", default=0.5)
    sy_gz = get_float("Start y in Gazebo (m) [default 0.0]: ", default=0.0)
    st = get_float("Start theta (deg) [default 0]: ", default=0.0)

    gx_gz = get_float("Goal x in Gazebo (m): ")
    gy_gz = get_float("Goal y in Gazebo (m): ")
    gt = get_float("Goal theta (deg, can be 0 if unused) [default 0]: ", default=0.0)

    rpm1 = get_float("RPM1: ")
    rpm2 = get_float("RPM2: ")
    clearance = get_float("Clearance (mm): ")

    execute_flag = input("Execute in Gazebo after planning? (y/n): ").strip().lower() == 'y'

    sx_mm, sy_mm = gazebo_to_planner_mm(sx_gz, sy_gz)
    gx_mm, gy_mm = gazebo_to_planner_mm(gx_gz, gy_gz)

    start = (sx_mm, sy_mm, wrap_angle_deg(st))
    goal = (gx_mm, gy_mm, wrap_angle_deg(gt))

    return start, goal, rpm1, rpm2, clearance, execute_flag, (sx_gz, sy_gz, st), (gx_gz, gy_gz, gt)


# =========================
# Main
# =========================
def main():
    (
        start,
        goal,
        rpm1,
        rpm2,
        clearance,
        execute_flag,
        start_gz,
        goal_gz,
    ) = get_inputs()

    obstacle_free, squares, bars, inflate, left_gap = build_obstacle_check(clearance)

    while not obstacle_free(start[0], start[1]):
        print("Start is in obstacle space or invalid. Enter again.")
        sx_gz = get_float("Start x in Gazebo (m): ")
        sy_gz = get_float("Start y in Gazebo (m): ")
        st = get_float("Start theta (deg): ")
        sx_mm, sy_mm = gazebo_to_planner_mm(sx_gz, sy_gz)
        start = (sx_mm, sy_mm, wrap_angle_deg(st))

    while not obstacle_free(goal[0], goal[1]):
        print("Goal is in obstacle space or invalid. Enter again.")
        gx_gz = get_float("Goal x in Gazebo (m): ")
        gy_gz = get_float("Goal y in Gazebo (m): ")
        gt = get_float("Goal theta (deg, can be 0 if unused): ")
        gx_mm, gy_mm = gazebo_to_planner_mm(gx_gz, gy_gz)
        goal = (gx_mm, gy_mm, wrap_angle_deg(gt))

    print("\nConverted planner-frame coordinates (mm):")
    print(f"  Start: ({start[0]:.1f}, {start[1]:.1f}, {start[2]:.1f} deg)")
    print(f"  Goal : ({goal[0]:.1f}, {goal[1]:.1f}, {goal[2]:.1f} deg)")

    parent, parent_action, final_node, exploration_img = astar(
        start, goal, rpm1, rpm2, obstacle_free, squares, bars, inflate, left_gap
    )

    if final_node is None:
        print("No path found.")
        cv2.imshow("A* Exploration", exploration_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    path_actions = reconstruct_path(start, final_node, parent, parent_action)

    print("\nPath found.")
    print(f"Final reached node in planner frame: {final_node}")
    fx_gz, fy_gz = planner_mm_to_gazebo(final_node[0], final_node[1])
    print(f"Final reached node in Gazebo frame: ({fx_gz:.3f}, {fy_gz:.3f}, {final_node[2]:.2f} deg)")
    print(f"Number of action primitives: {len(path_actions)}")

    print("\nAction sequence [UL, UR] RPM:")
    for i, (ul, ur, _, nxt) in enumerate(path_actions):
        nx_gz, ny_gz = planner_mm_to_gazebo(nxt[0], nxt[1])
        action_time = primitive_action_time(ul, ur)
        print(
            f"{i:03d}: [{ul:.2f}, {ur:.2f}]  T={action_time:.2f}s -> "
            f"planner=({nxt[0]:.2f}, {nxt[1]:.2f}, {nxt[2]:.2f} deg), "
            f"gazebo=({nx_gz:.3f}, {ny_gz:.3f}, {nxt[2]:.2f} deg)"
        )

    final_img = draw_final_path(exploration_img, start, goal, path_actions)
    cv2.imshow("Optimal Path", final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if execute_flag:
        execute_in_gazebo(path_actions)


if __name__ == "__main__":
    main()#!/usr/bin/env python3

import math
import time
import heapq
import numpy as np
import cv2

# ============================================================
# Gazebo-aligned 2x planner version
# ------------------------------------------------------------
# Original project map: 4000 x 2000 mm
# Gazebo world map:     8000 x 4000 mm (2x scaled)
#
# Gazebo frame placement given by user:
# - Gazebo origin is 0.5 m left of the UPPER-LEFT map corner
# - Gazebo origin is 2.0 m below the UPPER-LEFT map corner
# - Robot spawn is at x=0.5, y=0.0 in Gazebo
#
# With a 2x-scaled 8m x 4m map, this means:
# - upper-left map corner in Gazebo = (0.5,  2.0)
# - lower-left map corner in Gazebo = (0.5, -2.0)
# - upper-right map corner in Gazebo = (8.5,  2.0)
# - lower-right map corner in Gazebo = (8.5, -2.0)
#
# Internal planning is still done in millimeters, but now on the
# FULL 2x-scaled geometry so that open-loop execution matches Gazebo.
#
# Coordinate conversion used here:
#   x_gazebo_m = 0.5 + x_mm / 1000
#   y_gazebo_m = -2.0 + y_mm / 1000
#
# and inverse:
#   x_mm = (x_gazebo_m - 0.5) * 1000
#   y_mm = (y_gazebo_m + 2.0) * 1000
# ============================================================

# waffle measurements
WHEEL_RADIUS_MM = 33.0
WHEEL_DISTANCE_MM = 287.0
ROBOT_RADIUS_MM = 150.0

# =========================
# Gazebo-aligned 2x map constants
# =========================
MAP_SCALE = 2.0
MAP_W_MM = int(4000 * MAP_SCALE)   # 8000 mm = 8 m
MAP_H_MM = int(2000 * MAP_SCALE)   # 4000 mm = 4 m

# Gazebo placement of map corners (meters)
GAZEBO_UPPER_LEFT_X_M = 0.5
GAZEBO_UPPER_LEFT_Y_M = 2.0
GAZEBO_LOWER_LEFT_X_M = 0.5
GAZEBO_LOWER_LEFT_Y_M = -2.0

# Planning resolution
XY_RES_MM = 50
# THETA_RES_DEG = 30
THETA_RES_DEG = 15
DT = 0.05
ACTION_TIME = 1.0
GOAL_THRESH_MM = 100.0

# =========================
# Open-loop action timing calibration
# =========================
# MIN_ACTION_TIME = 0.35
MIN_ACTION_TIME = 0.65
MAX_ACTION_TIME = 4.00

STRAIGHT_DISTANCE_TARGET_M = 0.24
# STRAIGHT_DISTANCE_TARGET_M = 0.54
ARC_DISTANCE_TARGET_M = 0.18
ARC_ANGLE_TARGET_DEG = 18.0
PIVOT_ANGLE_TARGET_DEG = 24.0

# STRAIGHT_TIME_GAIN = 1.25
STRAIGHT_TIME_GAIN = 2.25
ARC_TIME_GAIN = 1.00
PIVOT_TIME_GAIN = 0.90

# Display scale for OpenCV visualization
DISPLAY_SCALE = 0.10   # 8000x4000 mm -> 800x400 px


# =========================
# Utility
# =========================
def wrap_angle_deg(theta_deg):
    return theta_deg % 360.0


def wrap_angle_rad(theta_rad):
    return theta_rad % (2.0 * math.pi)


def rpm_to_rad_per_sec(rpm):
    return rpm * 2.0 * math.pi / 60.0


def world_to_img(x_mm, y_mm):
    """Map world coordinates (origin bottom-left) to image coordinates."""
    px = int(round(x_mm * DISPLAY_SCALE))
    py = int(round((MAP_H_MM - y_mm) * DISPLAY_SCALE))
    return px, py


def gazebo_to_planner_mm(x_gz_m, y_gz_m):
    """
    Gazebo -> planner conversion for the 2x-scaled map.

    Planner frame:
      - origin at bottom-left of map
      - units in mm
      - +x right, +y up

    Gazebo frame:
      - map lower-left corner is at (0.5, -2.0)
      - units in meters
      - +x right, +y up
    """
    x_mm = (x_gz_m - GAZEBO_LOWER_LEFT_X_M) * 1000.0
    y_mm = (y_gz_m - GAZEBO_LOWER_LEFT_Y_M) * 1000.0
    return x_mm, y_mm


def planner_mm_to_gazebo(x_mm, y_mm):
    x_gz_m = GAZEBO_LOWER_LEFT_X_M + x_mm / 1000.0
    y_gz_m = GAZEBO_LOWER_LEFT_Y_M + y_mm / 1000.0
    return x_gz_m, y_gz_m


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def primitive_action_time(ul_rpm, ur_rpm):
    """
    Open-loop action duration derived from wheel kinematics.
    Shared by planner simulation and Gazebo execution.
    """
    R = 0.033
    L = 0.287

    wl = rpm_to_rad_per_sec(ul_rpm)
    wr = rpm_to_rad_per_sec(ur_rpm)

    v = (R / 2.0) * (wl + wr)
    w = (R / L) * (wr - wl)

    abs_v = abs(v)
    abs_w = abs(w)

    if abs_w < 1e-6:
        if abs_v < 1e-6:
            return MIN_ACTION_TIME
        t = STRAIGHT_DISTANCE_TARGET_M / abs_v
        t *= STRAIGHT_TIME_GAIN
        return clamp(t, MIN_ACTION_TIME, MAX_ACTION_TIME)

    if abs(ul_rpm) < 1e-6 or abs(ur_rpm) < 1e-6:
        target_angle_rad = math.radians(PIVOT_ANGLE_TARGET_DEG)
        t = target_angle_rad / abs_w
        t *= PIVOT_TIME_GAIN
        return clamp(t, MIN_ACTION_TIME, MAX_ACTION_TIME)

    target_dist_time = ARC_DISTANCE_TARGET_M / max(abs_v, 1e-6)
    target_angle_time = math.radians(ARC_ANGLE_TARGET_DEG) / max(abs_w, 1e-6)
    t = max(target_dist_time, target_angle_time)
    t *= ARC_TIME_GAIN
    return clamp(t, MIN_ACTION_TIME, MAX_ACTION_TIME)


# =========================
# Obstacle helpers
# =========================
def point_in_rect(x, y, xmin, xmax, ymin, ymax):
    return xmin <= x <= xmax and ymin <= y <= ymax


def point_in_rotated_rect(x, y, cx, cy, length, width, angle_deg):
    theta = math.radians(angle_deg)
    dx = x - cx
    dy = y - cy

    # Rotate point into rectangle frame
    xr = dx * math.cos(theta) + dy * math.sin(theta)
    yr = -dx * math.sin(theta) + dy * math.cos(theta)

    return abs(xr) <= length / 2.0 and abs(yr) <= width / 2.0


def build_obstacle_check(clearance_mm):
    """
    Returns a closure obstacle_free(x, y) for the 2x-scaled project map.

    Modification for Gazebo:
    - the LEFT OUTER WALL now has a centered opening of 1.0 m in Gazebo
      (= 1000 mm in this 2x planner map)
    - that opening is centered at the robot spawn height, y = 0.0 m in Gazebo,
      which corresponds to planner y = 2000 mm
    """
    inflate = ROBOT_RADIUS_MM + clearance_mm

    # Original project-map approximations, now scaled by 2x.
    square_side = 400.0 * MAP_SCALE

    squares = [
        (420.0 * MAP_SCALE,  450.0 * MAP_SCALE, square_side),
        (1335.0 * MAP_SCALE, 1550.0 * MAP_SCALE, square_side),
        (2200.0 * MAP_SCALE, 1740.0 * MAP_SCALE, square_side),
    ]

    bars = [
        (800.0 * MAP_SCALE,  1400.0 * MAP_SCALE, 1400.0 * MAP_SCALE, 50.0 * MAP_SCALE, -60.0),
        (1700.0 * MAP_SCALE,  500.0 * MAP_SCALE, 1200.0 * MAP_SCALE, 50.0 * MAP_SCALE,  60.0),
        (2925.0 * MAP_SCALE, 1275.0 * MAP_SCALE, 1450.0 * MAP_SCALE, 50.0 * MAP_SCALE,  90.0),
    ]

    # Left wall opening: 1.0 m in Gazebo = 1000 mm in planner.
    left_gap_height_mm = 1000.0
    left_gap_center_y_mm = 2000.0   # spawn y = 0.0 m in Gazebo
    left_gap_ymin = left_gap_center_y_mm - left_gap_height_mm / 2.0
    left_gap_ymax = left_gap_center_y_mm + left_gap_height_mm / 2.0

    def obstacle_free(x, y):
        # Right, bottom, and top boundaries are still closed and inflated inward.
        if x > (MAP_W_MM - inflate):
            return False
        if y < inflate or y > (MAP_H_MM - inflate):
            return False

        # Left boundary is open only inside the requested gap.
        # Outside the gap, it behaves like a normal inflated wall.
        if x < inflate:
            if y < left_gap_ymin + inflate or y > left_gap_ymax - inflate:
                return False

        for cx, cy, side in squares:
            half = side / 2.0 + inflate
            if point_in_rect(x, y, cx - half, cx + half, cy - half, cy + half):
                return False

        for cx, cy, length, width, angle_deg in bars:
            if point_in_rotated_rect(
                x, y,
                cx, cy,
                length + 2.0 * inflate,
                width + 2.0 * inflate,
                angle_deg,
            ):
                return False

        return True

    return obstacle_free, squares, bars, inflate, (left_gap_ymin, left_gap_ymax)


# =========================
# Differential drive motion primitive
# =========================
def move_with_rpms(node, ul_rpm, ur_rpm, obstacle_free):
    """
    node = (x_mm, y_mm, theta_deg)
    returns:
        (x_new, y_new, theta_new_deg), distance_cost, curve_points
    or None if collision occurs during integration
    """
    x, y, theta_deg = node
    theta = math.radians(theta_deg)

    ul = rpm_to_rad_per_sec(ul_rpm)
    ur = rpm_to_rad_per_sec(ur_rpm)

    t = 0.0
    D = 0.0
    curve_points = [(x, y)]
    action_time = primitive_action_time(ul_rpm, ur_rpm)

    while t < action_time:
        t += DT

        x_prev = x
        y_prev = y

        x += 0.5 * WHEEL_RADIUS_MM * (ul + ur) * math.cos(theta) * DT
        y += 0.5 * WHEEL_RADIUS_MM * (ul + ur) * math.sin(theta) * DT
        theta += (WHEEL_RADIUS_MM / WHEEL_DISTANCE_MM) * (ur - ul) * DT

        if not obstacle_free(x, y):
            return None

        D += np.hypot(x - x_prev, y - y_prev)
        curve_points.append((x, y))

    theta_deg_new = wrap_angle_deg(math.degrees(theta))
    return (x, y, theta_deg_new), D, curve_points


# =========================
# A* bookkeeping
# =========================
def get_index(node):
    x, y, theta_deg = node
    ix = int(round(x / XY_RES_MM))
    iy = int(round(y / XY_RES_MM))
    it = int(round(wrap_angle_deg(theta_deg) / THETA_RES_DEG)) % int(360 / THETA_RES_DEG)
    return ix, iy, it


def heuristic(node, goal):
    return np.hypot(node[0] - goal[0], node[1] - goal[1])


def reached_goal(node, goal):
    return np.hypot(node[0] - goal[0], node[1] - goal[1]) <= GOAL_THRESH_MM


# =========================
# Map drawing
# =========================
def make_base_map(squares, bars, inflate, left_gap=None):
    w = int(MAP_W_MM * DISPLAY_SCALE)
    h = int(MAP_H_MM * DISPLAY_SCALE)
    img = np.ones((h, w, 3), dtype=np.uint8) * 255

    # Border with optional left-wall gap
    if left_gap is None:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 0), 2)
    else:
        gap_ymin, gap_ymax = left_gap
        # top wall
        cv2.line(img, (0, 0), (w - 1, 0), (0, 0, 0), 2)
        # bottom wall
        cv2.line(img, (0, h - 1), (w - 1, h - 1), (0, 0, 0), 2)
        # right wall
        cv2.line(img, (w - 1, 0), (w - 1, h - 1), (0, 0, 0), 2)

        # left wall split into two segments around the opening
        p_top = world_to_img(0.0, gap_ymax)
        p_bot = world_to_img(0.0, gap_ymin)
        cv2.line(img, (0, 0), (0, p_top[1]), (0, 0, 0), 2)
        cv2.line(img, (0, p_bot[1]), (0, h - 1), (0, 0, 0), 2)

    # Draw inflated squares
    for cx, cy, side in squares:
        half = side / 2.0 + inflate
        p1 = world_to_img(cx - half, cy - half)
        p2 = world_to_img(cx + half, cy + half)
        cv2.rectangle(
            img,
            (min(p1[0], p2[0]), min(p1[1], p2[1])),
            (max(p1[0], p2[0]), max(p1[1], p2[1])),
            (0, 0, 0),
            -1,
        )

    # Draw inflated bars
    for cx, cy, length, width, angle_deg in bars:
        box = cv2.boxPoints((
            (cx * DISPLAY_SCALE, (MAP_H_MM - cy) * DISPLAY_SCALE),
            ((length + 2.0 * inflate) * DISPLAY_SCALE, (width + 2.0 * inflate) * DISPLAY_SCALE),
            -angle_deg,
        ))
        box = np.round(box).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(img, [box], (0, 0, 0))

    # End line at right map edge
    endline_x = int(MAP_W_MM * DISPLAY_SCALE)
    cv2.line(img, (endline_x - 3, 0), (endline_x - 3, h - 1), (255, 0, 0), 3)

    return img


# =========================
# A*
# =========================
def astar(start, goal, rpm1, rpm2, obstacle_free, squares, bars, inflate, left_gap=None):
    # actions = [
    #     (0, rpm1),
    #     (rpm1, 0),
    #     (rpm1, rpm1),
    #     (0, rpm2),
    #     (rpm2, 0),
    #     (rpm2, rpm2),
    #     (rpm1, rpm2),
    #     (rpm2, rpm1),
    # ]
    actions = [
        (0, rpm1),
        (rpm1, 0),
        (rpm1, rpm1),
        (0, rpm2),
        (rpm2, 0),
        (rpm2, rpm1),
        (rpm1, rpm2),
        (rpm2, rpm2),
    ]
    

    open_list = []
    parent = {}
    parent_action = {}
    cost_to_come = {}
    visited = set()

    start_idx = get_index(start)
    cost_to_come[start_idx] = 0.0
    heapq.heappush(open_list, (heuristic(start, goal), 0.0, start))

    img = make_base_map(squares, bars, inflate, left_gap)

    while open_list:
        f, g, current = heapq.heappop(open_list)
        cidx = get_index(current)

        if cidx in visited:
            continue
        visited.add(cidx)

        if reached_goal(current, goal):
            return parent, parent_action, current, img

        for ul, ur in actions:
            result = move_with_rpms(current, ul, ur, obstacle_free)
            if result is None:
                continue

            nxt, edge_cost, curve_points = result
            nidx = get_index(nxt)

            if nidx in visited:
                continue

            new_g = g + edge_cost

            if (nidx not in cost_to_come) or (new_g < cost_to_come[nidx]):
                cost_to_come[nidx] = new_g
                new_f = new_g + heuristic(nxt, goal)

                parent[nidx] = cidx
                parent_action[nidx] = (ul, ur, curve_points, nxt)
                heapq.heappush(open_list, (new_f, new_g, nxt))

                for i in range(len(curve_points) - 1):
                    p1 = world_to_img(curve_points[i][0], curve_points[i][1])
                    p2 = world_to_img(curve_points[i + 1][0], curve_points[i + 1][1])
                    cv2.line(img, p1, p2, (150, 150, 150), 1)

        cv2.imshow("A* Exploration", img)
        cv2.waitKey(1)

    return None, None, None, img


def reconstruct_path(start, final_node, parent, parent_action):
    path_actions = []
    current_idx = get_index(final_node)
    start_idx = get_index(start)

    while current_idx != start_idx:
        action = parent_action[current_idx]
        path_actions.append(action)
        current_idx = parent[current_idx]

    path_actions.reverse()
    return path_actions


def draw_final_path(img, start, goal, path_actions):
    out = img.copy()

    for ul, ur, curve_points, nxt in path_actions:
        for i in range(len(curve_points) - 1):
            p1 = world_to_img(curve_points[i][0], curve_points[i][1])
            p2 = world_to_img(curve_points[i + 1][0], curve_points[i + 1][1])
            cv2.line(out, p1, p2, (0, 0, 255), 2)

    sp = world_to_img(start[0], start[1])
    gp = world_to_img(goal[0], goal[1])
    cv2.circle(out, sp, 6, (0, 255, 0), -1)
    cv2.circle(out, gp, 6, (255, 0, 0), -1)

    arrow_len = 200.0
    hx = start[0] + arrow_len * math.cos(math.radians(start[2]))
    hy = start[1] + arrow_len * math.sin(math.radians(start[2]))
    hp = world_to_img(hx, hy)
    cv2.arrowedLine(out, sp, hp, (0, 128, 0), 2)

    return out


# =========================
# ROS open-loop execution
# =========================
def execute_in_gazebo(path_actions):
    import time
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist

    class CmdVelPublisher(Node):
        def __init__(self):
            super().__init__('astar_path_executor')
            self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

    WHEEL_RADIUS_M = 0.033
    WHEEL_DISTANCE_M = 0.287

    rclpy.init()
    node = CmdVelPublisher()

    try:
        print("execute_in_gazebo called")
        print("num actions:", len(path_actions))
        # time.sleep(1.0)
        time.sleep(1.0)

        for i, action in enumerate(path_actions):
            ul_rpm = action[0]
            ur_rpm = action[1]

            ul = ul_rpm * 2.0 * math.pi / 60.0
            ur = ur_rpm * 2.0 * math.pi / 60.0

            msg = Twist()
            msg.linear.x = (WHEEL_RADIUS_M / 2.0) * (ul + ur)
            msg.angular.z = (WHEEL_RADIUS_M / WHEEL_DISTANCE_M) * (ur - ul)

            action_time = primitive_action_time(ul_rpm, ur_rpm)

            print(
                f"Publishing action {i}: UL={ul_rpm}, UR={ur_rpm}, "
                f"v={msg.linear.x:.3f}, w={msg.angular.z:.3f}, T={action_time:.2f}s"
            )

            t0 = time.time()
            while time.time() - t0 < action_time:
                node.pub.publish(msg)
                rclpy.spin_once(node, timeout_sec=0.0)
                # time.sleep(0.05)
                time.sleep(0.01)

            node.pub.publish(Twist())
            # time.sleep(0.05)
            time.sleep(0.01)

        node.pub.publish(Twist())

    finally:
        node.destroy_node()
        rclpy.shutdown()


# =========================
# Inputs
# =========================
def get_float(prompt, default=None):
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("Invalid number. Try again.")


def print_gazebo_alignment_summary():
    print("\n=== Gazebo-aligned 2x planner ===")
    print(f"Planner map size: {MAP_W_MM} mm x {MAP_H_MM} mm  ({MAP_W_MM/1000:.1f} m x {MAP_H_MM/1000:.1f} m)")
    print("Gazebo map corners assumed:")
    print("  upper-left  = (0.5,  2.0) m")
    print("  lower-left  = (0.5, -2.0) m")
    print("  upper-right = (8.5,  2.0) m")
    print("  lower-right = (8.5, -2.0) m")
    print("Robot spawn given by user: (0.5, 0.0) m")
    print("This corresponds to planner start position: (0 mm, 2000 mm)")
    print("Left wall modification: 1.0 m opening in Gazebo, centered at spawn height")
    print("Heading convention: 0 deg = +x (right), 90 deg = +y (up)")
    print("Open-loop timing: adaptive per action, using wheel kinematics\n")


def get_inputs():
    print_gazebo_alignment_summary()
    print("Enter START / GOAL in GAZEBO METERS and heading in DEGREES.\n")

    sx_gz = get_float("Start x in Gazebo (m) [default 0.5]: ", default=0.5)
    sy_gz = get_float("Start y in Gazebo (m) [default 0.0]: ", default=0.0)
    st = get_float("Start theta (deg) [default 0]: ", default=0.0)

    gx_gz = get_float("Goal x in Gazebo (m): ")
    gy_gz = get_float("Goal y in Gazebo (m): ")
    gt = get_float("Goal theta (deg, can be 0 if unused) [default 0]: ", default=0.0)

    rpm1 = get_float("RPM1: ")
    rpm2 = get_float("RPM2: ")
    clearance = get_float("Clearance (mm): ")

    execute_flag = input("Execute in Gazebo after planning? (y/n): ").strip().lower() == 'y'

    sx_mm, sy_mm = gazebo_to_planner_mm(sx_gz, sy_gz)
    gx_mm, gy_mm = gazebo_to_planner_mm(gx_gz, gy_gz)

    start = (sx_mm, sy_mm, wrap_angle_deg(st))
    goal = (gx_mm, gy_mm, wrap_angle_deg(gt))

    return start, goal, rpm1, rpm2, clearance, execute_flag, (sx_gz, sy_gz, st), (gx_gz, gy_gz, gt)


# =========================
# Main
# =========================
def main():
    (
        start,
        goal,
        rpm1,
        rpm2,
        clearance,
        execute_flag,
        start_gz,
        goal_gz,
    ) = get_inputs()

    obstacle_free, squares, bars, inflate, left_gap = build_obstacle_check(clearance)

    while not obstacle_free(start[0], start[1]):
        print("Start is in obstacle space or invalid. Enter again.")
        sx_gz = get_float("Start x in Gazebo (m): ")
        sy_gz = get_float("Start y in Gazebo (m): ")
        st = get_float("Start theta (deg): ")
        sx_mm, sy_mm = gazebo_to_planner_mm(sx_gz, sy_gz)
        start = (sx_mm, sy_mm, wrap_angle_deg(st))

    while not obstacle_free(goal[0], goal[1]):
        print("Goal is in obstacle space or invalid. Enter again.")
        gx_gz = get_float("Goal x in Gazebo (m): ")
        gy_gz = get_float("Goal y in Gazebo (m): ")
        gt = get_float("Goal theta (deg, can be 0 if unused): ")
        gx_mm, gy_mm = gazebo_to_planner_mm(gx_gz, gy_gz)
        goal = (gx_mm, gy_mm, wrap_angle_deg(gt))

    print("\nConverted planner-frame coordinates (mm):")
    print(f"  Start: ({start[0]:.1f}, {start[1]:.1f}, {start[2]:.1f} deg)")
    print(f"  Goal : ({goal[0]:.1f}, {goal[1]:.1f}, {goal[2]:.1f} deg)")

    parent, parent_action, final_node, exploration_img = astar(
        start, goal, rpm1, rpm2, obstacle_free, squares, bars, inflate, left_gap
    )

    if final_node is None:
        print("No path found.")
        cv2.imshow("A* Exploration", exploration_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    path_actions = reconstruct_path(start, final_node, parent, parent_action)

    print("\nPath found.")
    print(f"Final reached node in planner frame: {final_node}")
    fx_gz, fy_gz = planner_mm_to_gazebo(final_node[0], final_node[1])
    print(f"Final reached node in Gazebo frame: ({fx_gz:.3f}, {fy_gz:.3f}, {final_node[2]:.2f} deg)")
    print(f"Number of action primitives: {len(path_actions)}")

    print("\nAction sequence [UL, UR] RPM:")
    for i, (ul, ur, _, nxt) in enumerate(path_actions):
        nx_gz, ny_gz = planner_mm_to_gazebo(nxt[0], nxt[1])
        action_time = primitive_action_time(ul, ur)
        print(
            f"{i:03d}: [{ul:.2f}, {ur:.2f}]  T={action_time:.2f}s -> "
            f"planner=({nxt[0]:.2f}, {nxt[1]:.2f}, {nxt[2]:.2f} deg), "
            f"gazebo=({nx_gz:.3f}, {ny_gz:.3f}, {nxt[2]:.2f} deg)"
        )

    final_img = draw_final_path(exploration_img, start, goal, path_actions)
    cv2.imshow("Optimal Path", final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if execute_flag:
        execute_in_gazebo(path_actions)


if __name__ == "__main__":
    main()