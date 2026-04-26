"""A* path planner for the FalconSim TurtleBot3 warehouse task.

The public entry point is :func:`plan_path`.  The ROS 2 controller calls this
function once, receives a list of relative Falcon-local motion increments, and
then handles path tracking.

Returned path format:
    [dx_cm, dy_cm, dtheta_rad]

The planner searches over differential-drive motion primitives generated from
wheel RPM pairs.  Each primitive is collision checked against an inflated map,
which keeps the returned path feasible for a robot with nonzero radius and user
specified clearance.

Coordinate conventions:
    * Controller inputs are Falcon-local centimeters.
    * A* runs in a y-up planner frame.
    * Falcon TF reports meters and has the opposite sign for local Y.

The conversion helpers below keep those frame conventions explicit so that the
planner, controller, and debug logs stay consistent.
"""

from __future__ import annotations

import heapq
import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

State = Tuple[float, float, float]  # x_cm, y_cm, theta_deg
Index = Tuple[int, int, int]
ActionRecord = Tuple[float, float, Tuple[Tuple[float, float], ...], State]

# Static calibration between Falcon-local coordinates and the ROS TF frame.
# The closed-loop controller can estimate these offsets at runtime, but the
# planner keeps the nominal values for logging and debug-file generation.
#
#     tf_x = FALCON_TF_ORIGIN_X_M + local_x_cm / FALCON_CM_PER_TF_M
#     tf_y = FALCON_TF_ORIGIN_Y_M + FALCON_Y_SIGN * local_y_cm / FALCON_CM_PER_TF_M
FALCON_TF_ORIGIN_X_M = 16.40
FALCON_TF_ORIGIN_Y_M = -6.05
FALCON_CM_PER_TF_M = 100.0
FALCON_Y_SIGN = -1.0

# Scale applied to the obstacle map.  Motion commands remain in centimeters;
# only the internal obstacle geometry changes when this value is tuned.
MAP_SCALE = 1.0
MAP_W_CM = 400.0 * MAP_SCALE
MAP_H_CM = 200.0 * MAP_SCALE

# Store obstacles in the original y-up planner frame, then mirror Falcon-local
# inputs/outputs at the API boundary when the simulation frame requires it.
FLIP_FALCON_Y_FOR_PLANNING = True

XY_RES_CM = 10.0
THETA_RES_DEG = 15.0
INTEGRATION_DT = 0.05

# Motion primitive tuning.  These values determine the simulated duration of each
# RPM action during search; the controller still receives only geometric deltas.
MIN_ACTION_TIME = 0.65
MAX_ACTION_TIME = 4.00
STRAIGHT_DISTANCE_TARGET_CM = 24.0
ARC_DISTANCE_TARGET_CM = 18.0
ARC_ANGLE_TARGET_DEG = 18.0
PIVOT_ANGLE_TARGET_DEG = 24.0
STRAIGHT_TIME_GAIN = 2.25
ARC_TIME_GAIN = 1.00
PIVOT_TIME_GAIN = 0.90

MAX_EXPANSIONS = 250_000


def _log(logger, level: str, msg: str) -> None:
    """Log through ROS when available, otherwise fall back to stdout.

    ROS 2 identifies log calls by source location.  Calling different severities
    through the same dynamic helper line can raise a severity-cache error, so
    each severity is dispatched from a separate branch.
    """
    if logger is None:
        print(msg)
        return

    try:
        normalized = str(level).lower()
        if normalized in ("warn", "warning"):
            logger.warning(msg) if hasattr(logger, "warning") else logger.warn(msg)
        elif normalized == "error":
            logger.error(msg)
        elif normalized == "debug":
            logger.debug(msg)
        else:
            logger.info(msg)
    except Exception:
        print(f"[{str(level).upper()}] {msg}")


def wrap_angle_deg(theta_deg: float) -> float:
    return theta_deg % 360.0


def wrap_angle_rad(theta_rad: float) -> float:
    return theta_rad % (2.0 * math.pi)


def angle_diff_rad(to_angle: float, from_angle: float) -> float:
    """Smallest signed angular difference from from_angle to to_angle."""
    return (to_angle - from_angle + math.pi) % (2.0 * math.pi) - math.pi


def rpm_to_rad_per_sec(rpm: float) -> float:
    return float(rpm) * 2.0 * math.pi / 60.0


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _controller_theta_to_deg(theta_value: float) -> float:
    """Convert the theta value received from falcon_amr_controller.py.

    The supplied controller multiplies every start/end array element by 100.
    For the default project parameters theta is zero, so nothing changes.  This
    helper also handles the common nonzero case where a launch theta was given
    in radians and was accidentally converted to centi-radians.
    """
    theta_value = float(theta_value)
    if abs(theta_value) > 2.0 * math.pi:
        theta_value = theta_value / 100.0
    return wrap_angle_deg(math.degrees(theta_value))


def falcon_local_to_tf_xy(x_cm: float, y_cm: float) -> Tuple[float, float]:
    """Convert Falcon-local map centimeters to observed ROS TF meters.

    This is used for logging/debugging. The provided controller expects relative
    local centimeter increments, so absolute TF values are not returned.
    """
    tf_x = FALCON_TF_ORIGIN_X_M + float(x_cm) / FALCON_CM_PER_TF_M
    tf_y = FALCON_TF_ORIGIN_Y_M + FALCON_Y_SIGN * float(y_cm) / FALCON_CM_PER_TF_M
    return tf_x, tf_y


def falcon_local_state_to_planner_state(state: State) -> State:
    """Map controller/Falcon-local coordinates into the planner's y-up frame."""
    x_cm, y_cm, theta_deg = state
    if FLIP_FALCON_Y_FOR_PLANNING:
        return (float(x_cm), float(MAP_H_CM - y_cm), wrap_angle_deg(-theta_deg))
    return (float(x_cm), float(y_cm), wrap_angle_deg(theta_deg))


def planner_state_to_falcon_local_state(state: State) -> State:
    """Map internal planner coordinates back to controller/Falcon-local coordinates."""
    x_cm, y_cm, theta_deg = state
    if FLIP_FALCON_Y_FOR_PLANNING:
        return (float(x_cm), float(MAP_H_CM - y_cm), wrap_angle_deg(-theta_deg))
    return (float(x_cm), float(y_cm), wrap_angle_deg(theta_deg))


def primitive_action_time(
    ul_rpm: float,
    ur_rpm: float,
    wheel_radius_cm: float,
    wheel_distance_cm: float,
) -> float:
    """Return the planning duration for one RPM primitive."""
    wl = rpm_to_rad_per_sec(ul_rpm)
    wr = rpm_to_rad_per_sec(ur_rpm)

    v = (wheel_radius_cm / 2.0) * (wl + wr)
    w = (wheel_radius_cm / wheel_distance_cm) * (wr - wl)

    abs_v = abs(v)
    abs_w = abs(w)

    if abs_w < 1e-9:
        if abs_v < 1e-9:
            return MIN_ACTION_TIME
        t = STRAIGHT_DISTANCE_TARGET_CM / abs_v
        t *= STRAIGHT_TIME_GAIN
        return clamp(t, MIN_ACTION_TIME, MAX_ACTION_TIME)

    if abs(ul_rpm) < 1e-9 or abs(ur_rpm) < 1e-9:
        target_angle_rad = math.radians(PIVOT_ANGLE_TARGET_DEG)
        t = target_angle_rad / abs_w
        t *= PIVOT_TIME_GAIN
        return clamp(t, MIN_ACTION_TIME, MAX_ACTION_TIME)

    target_dist_time = ARC_DISTANCE_TARGET_CM / max(abs_v, 1e-9)
    target_angle_time = math.radians(ARC_ANGLE_TARGET_DEG) / max(abs_w, 1e-9)
    t = max(target_dist_time, target_angle_time)
    t *= ARC_TIME_GAIN
    return clamp(t, MIN_ACTION_TIME, MAX_ACTION_TIME)


def point_in_rect(x: float, y: float, xmin: float, xmax: float, ymin: float, ymax: float) -> bool:
    return xmin <= x <= xmax and ymin <= y <= ymax


def point_in_rotated_rect(
    x: float,
    y: float,
    cx: float,
    cy: float,
    length: float,
    width: float,
    angle_deg: float,
) -> bool:
    theta = math.radians(angle_deg)
    dx = x - cx
    dy = y - cy

    xr = dx * math.cos(theta) + dy * math.sin(theta)
    yr = -dx * math.sin(theta) + dy * math.cos(theta)

    return abs(xr) <= length / 2.0 and abs(yr) <= width / 2.0


def build_obstacle_check(robot_radius_cm: float, clearance_cm: float) -> Callable[[float, float], bool]:
    """Create the collision checker for the scaled warehouse map."""
    inflate = float(robot_radius_cm) + float(clearance_cm)
    square_side = 40.0 * MAP_SCALE

    squares = [
        (42.0 * MAP_SCALE, 45.0 * MAP_SCALE, square_side),
        (133.5 * MAP_SCALE, 155.0 * MAP_SCALE, square_side),
        (220.0 * MAP_SCALE, 174.0 * MAP_SCALE, square_side),
    ]

    bars = [
        (80.0 * MAP_SCALE, 140.0 * MAP_SCALE, 140.0 * MAP_SCALE, 5.0 * MAP_SCALE, -60.0),
        (170.0 * MAP_SCALE, 50.0 * MAP_SCALE, 120.0 * MAP_SCALE, 5.0 * MAP_SCALE, 60.0),
        (292.5 * MAP_SCALE, 127.5 * MAP_SCALE, 145.0 * MAP_SCALE, 5.0 * MAP_SCALE, 90.0),
    ]

    # Same left-wall opening as final_astar.py, now in cm.
    left_gap_height_cm = 100.0
    left_gap_center_y_cm = MAP_H_CM / 2.0
    left_gap_ymin = left_gap_center_y_cm - left_gap_height_cm / 2.0
    left_gap_ymax = left_gap_center_y_cm + left_gap_height_cm / 2.0

    def obstacle_free(x: float, y: float) -> bool:
        # Treat map boundaries as obstacles after accounting for robot radius and
        # requested clearance.  The left boundary keeps the spawn opening.
        if x > (MAP_W_CM - inflate):
            return False
        if y < inflate or y > (MAP_H_CM - inflate):
            return False
        if x < inflate:
            if y < left_gap_ymin + inflate or y > left_gap_ymax - inflate:
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

    return obstacle_free


def move_with_rpms(
    node: State,
    ul_rpm: float,
    ur_rpm: float,
    obstacle_free: Callable[[float, float], bool],
    wheel_radius_cm: float,
    wheel_distance_cm: float,
) -> Optional[Tuple[State, float, Tuple[Tuple[float, float], ...]]]:
    """Integrate one differential-drive primitive in the planner frame."""
    x, y, theta_deg = node
    theta = math.radians(theta_deg)

    ul = rpm_to_rad_per_sec(ul_rpm)
    ur = rpm_to_rad_per_sec(ur_rpm)

    t = 0.0
    distance = 0.0
    curve_points: List[Tuple[float, float]] = [(x, y)]
    action_time = primitive_action_time(ul_rpm, ur_rpm, wheel_radius_cm, wheel_distance_cm)

    while t < action_time:
        step = min(INTEGRATION_DT, action_time - t)
        t += step

        x_prev = x
        y_prev = y

        x += 0.5 * wheel_radius_cm * (ul + ur) * math.cos(theta) * step
        y += 0.5 * wheel_radius_cm * (ul + ur) * math.sin(theta) * step
        theta += (wheel_radius_cm / wheel_distance_cm) * (ur - ul) * step

        if not obstacle_free(x, y):
            return None

        distance += math.hypot(x - x_prev, y - y_prev)
        curve_points.append((x, y))

    theta_deg_new = wrap_angle_deg(math.degrees(theta))
    return (x, y, theta_deg_new), distance, tuple(curve_points)


def get_index(node: State) -> Index:
    x, y, theta_deg = node
    ix = int(round(x / XY_RES_CM))
    iy = int(round(y / XY_RES_CM))
    theta_bins = int(round(360.0 / THETA_RES_DEG))
    it = int(round(wrap_angle_deg(theta_deg) / THETA_RES_DEG)) % theta_bins
    return ix, iy, it


def heuristic(node: State, goal: State) -> float:
    return math.hypot(node[0] - goal[0], node[1] - goal[1])


def reached_goal(node: State, goal: State, goal_threshold_cm: float) -> bool:
    return heuristic(node, goal) <= goal_threshold_cm


def astar(
    start: State,
    goal: State,
    rpm1: float,
    rpm2: float,
    obstacle_free: Callable[[float, float], bool],
    goal_threshold_cm: float,
    wheel_radius_cm: float,
    wheel_distance_cm: float,
    logger=None,
) -> Optional[Tuple[Dict[Index, Index], Dict[Index, ActionRecord], State, int]]:
    """Run A* with differential-drive RPM actions."""
    actions = [
        (0.0, rpm1),
        (rpm1, 0.0),
        (rpm1, rpm1),
        (0.0, rpm2),
        (rpm2, 0.0),
        (rpm2, rpm1),
        (rpm1, rpm2),
        (rpm2, rpm2),
    ]

    open_list: List[Tuple[float, float, int, State]] = []
    parent: Dict[Index, Index] = {}
    parent_action: Dict[Index, ActionRecord] = {}
    cost_to_come: Dict[Index, float] = {}
    visited = set()

    start_idx = get_index(start)
    cost_to_come[start_idx] = 0.0
    push_count = 0
    heapq.heappush(open_list, (heuristic(start, goal), 0.0, push_count, start))

    expansions = 0
    while open_list:
        _, g, _, current = heapq.heappop(open_list)
        cidx = get_index(current)

        if cidx in visited:
            continue
        visited.add(cidx)
        expansions += 1

        if expansions % 5000 == 0:
            _log(logger, "info", f"A* expansions: {expansions}, open: {len(open_list)}")

        if reached_goal(current, goal, goal_threshold_cm):
            return parent, parent_action, current, expansions

        if expansions >= MAX_EXPANSIONS:
            _log(logger, "warn", f"A* stopped at expansion limit ({MAX_EXPANSIONS}).")
            return None

        for ul, ur in actions:
            result = move_with_rpms(
                current,
                ul,
                ur,
                obstacle_free,
                wheel_radius_cm,
                wheel_distance_cm,
            )
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
                push_count += 1
                heapq.heappush(open_list, (new_f, new_g, push_count, nxt))

    return None


def reconstruct_path(start: State, final_node: State, parent: Dict[Index, Index], parent_action: Dict[Index, ActionRecord]) -> List[ActionRecord]:
    path_actions: List[ActionRecord] = []
    current_idx = get_index(final_node)
    start_idx = get_index(start)

    while current_idx != start_idx:
        action = parent_action[current_idx]
        path_actions.append(action)
        current_idx = parent[current_idx]

    path_actions.reverse()
    return path_actions


def _line_is_free(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    obstacle_free: Callable[[float, float], bool],
    step_cm: float = 2.0,
) -> bool:
    distance = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    steps = max(1, int(math.ceil(distance / step_cm)))
    for i in range(1, steps + 1):
        a = i / steps
        x = p0[0] + a * (p1[0] - p0[0])
        y = p0[1] + a * (p1[1] - p0[1])
        if not obstacle_free(x, y):
            return False
    return True


def actions_to_falcon_moves(
    start_planner: State,
    goal_planner: State,
    final_node: State,
    path_actions: Sequence[ActionRecord],
    obstacle_free: Callable[[float, float], bool],
) -> List[List[float]]:
    """Convert internal A* states to Falcon's [dx_cm, dy_cm, dtheta_rad] format.

    A* runs in the original y-up planner frame. The returned moves must be in
    Falcon-local/controller coordinates, where the measured relationship is:

        tf_y = -6.05 - local_y_cm / 100

    Therefore dy and dtheta are mirrored back before returning them.
    """
    # The final node is not needed for conversion, but remains in the signature
    # because visualization and earlier helper versions pass it through.
    del final_node

    moves: List[List[float]] = []
    current_planner = start_planner
    current_falcon = planner_state_to_falcon_local_state(current_planner)

    for _, _, _, nxt_planner in path_actions:
        next_falcon = planner_state_to_falcon_local_state(nxt_planner)

        current_theta = math.radians(current_falcon[2])
        next_theta = math.radians(next_falcon[2])

        dx = next_falcon[0] - current_falcon[0]
        dy = next_falcon[1] - current_falcon[1]
        dtheta = angle_diff_rad(next_theta, current_theta)
        moves.append([float(dx), float(dy), float(dtheta)])

        current_planner = nxt_planner
        current_falcon = next_falcon

    # A* stops inside a goal radius.  When the remaining straight-line segment is
    # collision-free, append a small correction so the controller targets the
    # exact requested goal pose.
    if _line_is_free(
        (current_planner[0], current_planner[1]),
        (goal_planner[0], goal_planner[1]),
        obstacle_free,
    ):
        goal_falcon = planner_state_to_falcon_local_state(goal_planner)
        dx = goal_falcon[0] - current_falcon[0]
        dy = goal_falcon[1] - current_falcon[1]
        dtheta = angle_diff_rad(math.radians(goal_falcon[2]), math.radians(current_falcon[2]))
        if abs(dx) > 1e-6 or abs(dy) > 1e-6 or abs(dtheta) > 1e-6:
            moves.append([float(dx), float(dy), float(dtheta)])

    return moves

def plan_path(
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
):
    """Plan a path for FalconSim.

    Parameters are exactly the ones supplied by falcon_amr_controller.py after it
    converts launch parameters from meters to centimeters.

    Returns:
        List[List[float]]: Falcon-compatible moves [dx_cm, dy_cm, dtheta_rad].
    """
    del delta_time  # Falcon uses this while executing the returned increments.

    if wheel_radius <= 0.0 or wheel_distance <= 0.0:
        raise ValueError("wheel_radius and wheel_distance must be positive")

    # The controller converts launch-file meters into Falcon-local centimeters
    # before calling this function.
    start_falcon: State = (
        float(start[0]),
        float(start[1]),
        _controller_theta_to_deg(float(start[2]) if len(start) > 2 else 0.0),
    )
    goal_falcon: State = (
        float(end[0]),
        float(end[1]),
        _controller_theta_to_deg(float(end[2]) if len(end) > 2 else 0.0),
    )

    # Convert into the y-up frame used by the obstacle map and A* lattice.
    start_state = falcon_local_state_to_planner_state(start_falcon)
    goal_state = falcon_local_state_to_planner_state(goal_falcon)

    obstacle_free = build_obstacle_check(float(robot_radius), float(clearance))

    if not obstacle_free(start_state[0], start_state[1]):
        _log(logger, "warn", f"Start {start_state[:2]} is in obstacle/inflated boundary space.")
    if not obstacle_free(goal_state[0], goal_state[1]):
        _log(logger, "warn", f"Goal {goal_state[:2]} is in obstacle/inflated boundary space.")

    start_tf = falcon_local_to_tf_xy(start_falcon[0], start_falcon[1])
    goal_tf = falcon_local_to_tf_xy(goal_falcon[0], goal_falcon[1])

    _log(
        logger,
        "info",
        "Planning A* path with measured Falcon offset: "
        f"falcon_local_start={start_falcon}, falcon_local_goal={goal_falcon}, "
        f"expected_tf_start=({start_tf[0]:.3f}, {start_tf[1]:.3f}), "
        f"expected_tf_goal=({goal_tf[0]:.3f}, {goal_tf[1]:.3f}), "
        f"planner_start={start_state}, planner_goal={goal_state}, "
        f"rr={robot_radius:.2f} cm, clearance={clearance:.2f} cm, "
        f"rpms=({rpm1:.2f}, {rpm2:.2f})",
    )

    result = astar(
        start_state,
        goal_state,
        float(rpm1),
        float(rpm2),
        obstacle_free,
        float(goal_threshold),
        float(wheel_radius),
        float(wheel_distance),
        logger,
    )

    if result is None:
        _log(logger, "error", "A* could not find a path. Returning an empty path.")
        return []

    parent, parent_action, final_node, expansions = result
    path_actions = reconstruct_path(start_state, final_node, parent, parent_action)
    moves = actions_to_falcon_moves(start_state, goal_state, final_node, path_actions, obstacle_free)

    net_dx = sum(m[0] for m in moves)
    net_dy = sum(m[1] for m in moves)
    net_dtheta = sum(m[2] for m in moves)

    _log(
        logger,
        "info",
        f"A* path found with {len(moves)} Falcon moves after {expansions} expansions; "
        f"planner_reached={final_node}; "
        f"net_falcon_move=({net_dx:.2f}, {net_dy:.2f}, {net_dtheta:.3f})",
    )
    return moves
