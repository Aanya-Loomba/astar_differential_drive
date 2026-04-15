import math
import time
import heapq
import numpy as np
import cv2

WHEEL_RADIUS_MM = 33.0
WHEEL_DISTANCE_MM = 287.0
ROBOT_RADIUS_MM = 150.0

MAP_W_MM = 4000
MAP_H_MM = 2000

XY_RES_MM = 50
THETA_RES_DEG = 30
DT = 0.1
ACTION_TIME = 1.0
GOAL_THRESH_MM = 100.0
DISPLAY_SCALE = 0.20

def wrap_angle_deg(theta_deg):
    return theta_deg % 360.0

def rpm_to_rad_per_sec(rpm):
    return rpm * 2.0 * math.pi / 60.0

def world_to_img(x_mm, y_mm):
    px = int(round(x_mm * DISPLAY_SCALE))
    py = int(round((MAP_H_MM - y_mm) * DISPLAY_SCALE))
    return px, py

def point_in_rect(x, y, xmin, xmax, ymin, ymax):
    return xmin <= x <= xmax and ymin <= y <= ymax

def point_in_rotated_rect(x, y, cx, cy, length, width, angle_deg):
    theta = math.radians(angle_deg)
    dx = x - cx
    dy = y - cy
    xr = dx * math.cos(theta) + dy * math.sin(theta)
    yr = -dx * math.sin(theta) + dy * math.cos(theta)
    return abs(xr) <= length / 2.0 and abs(yr) <= width / 2.0

def build_obstacle_check(clearance_mm):
    inflate = ROBOT_RADIUS_MM + clearance_mm
    square_side = 400.0
    squares = [
        (420.0, 450.0, square_side),
        (1335.0, 1550.0, square_side),
        (2200.0, 1740.0, square_side),
    ]
    bars = [
        (800.0, 1400.0, 1400.0, 50.0, -60.0),
        (1700.0, 500.0, 1200.0, 50.0, 60.0),
        (2925.0, 1275.0, 1450.0, 50.0, 90.0),
    ]

    def obstacle_free(x, y):
        if x < inflate or x > (MAP_W_MM - inflate):
            return False
        if y < inflate or y > (MAP_H_MM - inflate):
            return False

        for cx, cy, side in squares:
            half = side / 2.0 + inflate
            if point_in_rect(x, y, cx - half, cx + half, cy - half, cy + half):
                return False

        for cx, cy, length, width, angle_deg in bars:
            if point_in_rotated_rect(
                x,
                y,
                cx,
                cy,
                length + 2.0 * inflate,
                width + 2.0 * inflate,
                angle_deg,
            ):
                return False

        return True

    return obstacle_free, squares, bars, inflate

def move_with_rpms(node, ul_rpm, ur_rpm, obstacle_free):
    x, y, theta_deg = node
    theta = math.radians(theta_deg)
    ul = rpm_to_rad_per_sec(ul_rpm)
    ur = rpm_to_rad_per_sec(ur_rpm)
    t = 0.0
    distance = 0.0
    curve_points = [(x, y)]

    while t < ACTION_TIME:
        t += DT
        x_prev = x
        y_prev = y
        x += 0.5 * WHEEL_RADIUS_MM * (ul + ur) * math.cos(theta) * DT
        y += 0.5 * WHEEL_RADIUS_MM * (ul + ur) * math.sin(theta) * DT
        theta += (WHEEL_RADIUS_MM / WHEEL_DISTANCE_MM) * (ur - ul) * DT

        if not obstacle_free(x, y):
            return None

        distance += np.hypot(x - x_prev, y - y_prev)
        curve_points.append((x, y))

    return (x, y, wrap_angle_deg(math.degrees(theta))), distance, curve_points

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

def make_base_map(squares, bars, inflate):
    w = int(MAP_W_MM * DISPLAY_SCALE)
    h = int(MAP_H_MM * DISPLAY_SCALE)
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 0), 2)

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

    for cx, cy, length, width, angle_deg in bars:
        box = cv2.boxPoints(
            (
                (cx * DISPLAY_SCALE, (MAP_H_MM - cy) * DISPLAY_SCALE),
                ((length + 2.0 * inflate) * DISPLAY_SCALE, (width + 2.0 * inflate) * DISPLAY_SCALE),
                -angle_deg,
            )
        )
        box = np.round(box).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(img, [box], (0, 0, 0))

    return img

def astar(start, goal, rpm1, rpm2, obstacle_free, squares, bars, inflate):
    actions = [
        (0, rpm1),
        (rpm1, 0),
        (rpm1, rpm1),
        (0, rpm2),
        (rpm2, 0),
        (rpm2, rpm2),
        (rpm1, rpm2),
        (rpm2, rpm1),
    ]

    open_list = []
    parent = {}
    parent_action = {}
    cost_to_come = {}
    visited = set()

    cost_to_come[get_index(start)] = 0.0
    heapq.heappush(open_list, (heuristic(start, goal), 0.0, start))
    img = make_base_map(squares, bars, inflate)

    while open_list:
        _, g, current = heapq.heappop(open_list)
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
                parent[nidx] = cidx
                parent_action[nidx] = (ul, ur, curve_points, nxt)
                heapq.heappush(open_list, (new_g + heuristic(nxt, goal), new_g, nxt))

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
        path_actions.append(parent_action[current_idx])
        current_idx = parent[current_idx]
    path_actions.reverse()
    return path_actions

def draw_final_path(img, start, goal, path_actions):
    out = img.copy()
    for _, _, curve_points, _ in path_actions:
        for i in range(len(curve_points) - 1):
            p1 = world_to_img(curve_points[i][0], curve_points[i][1])
            p2 = world_to_img(curve_points[i + 1][0], curve_points[i + 1][1])
            cv2.line(out, p1, p2, (0, 0, 255), 2)
    sp = world_to_img(start[0], start[1])
    gp = world_to_img(goal[0], goal[1])
    cv2.circle(out, sp, 6, (0, 255, 0), -1)
    cv2.circle(out, gp, 6, (255, 0, 0), -1)
    return out

def get_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid number. Try again.")

def get_inputs():
    print("\nEnter all map coordinates in mm and angles in degrees.\n")
    sx = get_float("Start x (mm): ")
    sy = get_float("Start y (mm): ")
    st = get_float("Start theta (deg): ")
    gx = get_float("Goal x (mm): ")
    gy = get_float("Goal y (mm): ")
    gt = get_float("Goal theta (deg, can be 0 if unused): ")
    rpm1 = get_float("RPM1: ")
    rpm2 = get_float("RPM2: ")
    clearance = get_float("Clearance (mm): ")
    return (sx, sy, wrap_angle_deg(st)), (gx, gy, wrap_angle_deg(gt)), rpm1, rpm2, clearance

def main():
    start, goal, rpm1, rpm2, clearance = get_inputs()
    obstacle_free, squares, bars, inflate = build_obstacle_check(clearance)

    while not obstacle_free(start[0], start[1]):
        print("Start is in obstacle space or invalid. Enter again.")
        sx = get_float("Start x (mm): ")
        sy = get_float("Start y (mm): ")
        st = get_float("Start theta (deg): ")
        start = (sx, sy, wrap_angle_deg(st))

    while not obstacle_free(goal[0], goal[1]):
        print("Goal is in obstacle space or invalid. Enter again.")
        gx = get_float("Goal x (mm): ")
        gy = get_float("Goal y (mm): ")
        gt = get_float("Goal theta (deg, can be 0 if unused): ")
        goal = (gx, gy, wrap_angle_deg(gt))

    parent, parent_action, final_node, exploration_img = astar(
        start, goal, rpm1, rpm2, obstacle_free, squares, bars, inflate
    )

    if final_node is None:
        print("No path found.")
        cv2.imshow("A* Exploration", exploration_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    path_actions = reconstruct_path(start, final_node, parent, parent_action)
    print("\nPath found.")
    print(f"Final reached node: {final_node}")
    print(f"Number of action primitives: {len(path_actions)}")
    final_img = draw_final_path(exploration_img, start, goal, path_actions)
    cv2.imshow("Optimal Path", final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
