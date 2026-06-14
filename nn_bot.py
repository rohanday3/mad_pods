"""
nn_bot.py — Codingame-ready Mad Pod Racing bot using trained NN weights.

USAGE: replace RUNNER_W and BLOCKER_W with trained weights from nn_train.py output.
The rest of the bot (physics helpers, heuristic fallback) works without any imports.

Architecture:
  Runner: 14 -> 32 -> 16 -> 4   (target_dx, target_dy, thrust, use_boost)
  Blocker: 12 -> 32 -> 16 -> 3  (target_dx, target_dy, thrust)

All numpy operations are hand-coded to comply with Codingame's pure Python constraint.
"""

import math
import sys

# ── TRAINED WEIGHTS — replace with output from nn_train.py ───────────────────
# These are placeholder zero weights; bot will use heuristic fallback until replaced.
RUNNER_W = None   # Will be a flat list of floats after training
BLOCKER_W = None  # Will be a flat list of floats after training

RUNNER_ARCH = [14, 32, 16, 4]
BLOCKER_ARCH = [12, 32, 16, 3]

# ── Constants ─────────────────────────────────────────────────────────────────
CHECKPOINT_RADIUS = 600
DRAG = 0.85
MAX_ROTATE = 18.0
POD_RADIUS = 400.0
MAP_W = 16000
MAP_H = 9000


# ── Pure-Python micro neural network ─────────────────────────────────────────

def _relu(v):
    return [max(0.0, x) for x in v]


def _tanh(v):
    return [math.tanh(x) for x in v]


def _matmul_add(W_flat, b, x, rows, cols):
    """rows x cols matrix (stored row-major) @ x + b."""
    out = []
    for r in range(rows):
        s = b[r]
        for c in range(cols):
            s += W_flat[r * cols + c] * x[c]
        out.append(s)
    return out


def nn_forward(flat, x, arch):
    """arch = [in, h1, h2, out]. flat = flattened weights."""
    in_sz, h1, h2, out_sz = arch
    idx = 0
    def take(n):
        nonlocal idx
        v = flat[idx:idx+n]
        idx += n
        return v
    W1 = take(h1 * in_sz);  b1 = take(h1)
    W2 = take(h2 * h1);     b2 = take(h2)
    W3 = take(out_sz * h2); b3 = take(out_sz)
    h = _relu(_matmul_add(W1, b1, x, h1, in_sz))
    h = _relu(_matmul_add(W2, b2, h, h2, h1))
    return _tanh(_matmul_add(W3, b3, h, out_sz, h2))


# ── Feature helpers ───────────────────────────────────────────────────────────

def _norm_pos(x, y):
    return x / (MAP_W * 0.5) - 1.0, y / (MAP_H * 0.5) - 1.0


def _norm_vel(vx, vy):
    return vx / 1000.0, vy / 1000.0


def _norm_angle(a):
    return (a / 180.0) - 1.0


def runner_features(x, y, vx, vy, angle, next_cp, cps_passed, shield_cd, checkpoints, n_cp):
    cp_x, cp_y = checkpoints[next_cp]
    ncp_x, ncp_y = checkpoints[(next_cp + 1) % n_cp]
    px, py = _norm_pos(x, y)
    rvx, rvy = _norm_vel(vx, vy)
    angle_n = _norm_angle(angle)
    dcx = (cp_x - x) / MAP_W
    dcy = (cp_y - y) / MAP_H
    dist_cp = math.hypot(cp_x - x, cp_y - y) / 20000.0
    dnx = (ncp_x - x) / MAP_W
    dny = (ncp_y - x) / MAP_H
    desired = math.degrees(math.atan2(cp_y - y, cp_x - x)) % 360.0
    ang_diff = ((desired - angle + 180) % 360) - 180
    ang_diff_n = ang_diff / 180.0
    speed_n = math.hypot(vx, vy) / 1000.0
    cps_n = cps_passed / 30.0
    shield_n = shield_cd / 3.0
    return [px, py, rvx, rvy, angle_n, dcx, dcy, dist_cp, ang_diff_n, dnx, dny, speed_n, cps_n, shield_n]


def blocker_features(bx, by, bvx, bvy, bangle, ox, oy, ovx, ovy, o_next_cp, checkpoints, n_cp):
    ocp_x, ocp_y = checkpoints[o_next_cp]
    bpx, bpy = _norm_pos(bx, by)
    nbvx, nbvy = _norm_vel(bvx, bvy)
    dx_opp = (ox - bx) / MAP_W
    dy_opp = (oy - by) / MAP_H
    dist_opp = math.hypot(ox - bx, oy - by) / 20000.0
    novx, novy = _norm_vel(ovx, ovy)
    dx_cp = (ocp_x - bx) / MAP_W
    dy_cp = (ocp_y - by) / MAP_H
    angle_n = _norm_angle(bangle)
    return [bpx, bpy, nbvx, nbvy, dx_opp, dy_opp, dist_opp, novx, novy, dx_cp, dy_cp, angle_n]


# ── Decode NN output to commands ──────────────────────────────────────────────

def decode_runner(out, px, py, boosts_left):
    tx = px + out[0] * 3000.0
    ty = py + out[1] * 2000.0
    thrust = max(0, min(100, int((out[2] + 1.0) * 50.0)))
    use_boost = out[3] > 0.5 and boosts_left[0] > 0
    if use_boost:
        boosts_left[0] -= 1
        return f"{int(tx)} {int(ty)} BOOST"
    return f"{int(tx)} {int(ty)} {thrust}"


def decode_blocker(out, px, py):
    tx = px + out[0] * 3000.0
    ty = py + out[1] * 2000.0
    thrust = max(0, min(100, int((out[2] + 1.0) * 50.0)))
    return f"{int(tx)} {int(ty)} {thrust}"


# ── Heuristic fallback (champion_bot_v3 logic) ────────────────────────────────

class PodState:
    def __init__(self):
        self.checkpoints_passed = 0
        self.prev_next_cp = -1
        self.shield_cooldown = 0


def update_progress(state, next_cp, checkpoint_count):
    if state.prev_next_cp == -1:
        state.prev_next_cp = next_cp
    elif next_cp != state.prev_next_cp:
        diff = (next_cp - state.prev_next_cp) % checkpoint_count
        state.checkpoints_passed += diff
        state.prev_next_cp = next_cp


def closest_point_on_line(ax, ay, bx, by, px, py):
    abx = bx - ax; aby = by - ay
    ab_sq = abx * abx + aby * aby
    if ab_sq == 0:
        return px, py
    t = ((px - ax) * abx + (py - ay) * aby) / ab_sq
    return ax + abx * t, ay + aby * t


def is_entering_cp_soon(x, y, vx, vy, cp_x, cp_y, frames=6):
    px, py, pvx, pvy = x, y, vx, vy
    for _ in range(frames):
        pvx *= DRAG; pvy *= DRAG; px += pvx; py += pvy
        if math.hypot(px - cp_x, py - cp_y) <= CHECKPOINT_RADIUS:
            return True
    return False


def get_progress_score(state, x, y, next_cp, checkpoints):
    cx, cy = checkpoints[next_cp]
    return state.checkpoints_passed * 20000.0 - math.hypot(cx - x, cy - y)


def heuristic_runner(x, y, vx, vy, angle, next_cp, state, checkpoints, n_cp, boosts_left, turn_number, boost_dist_thresh, total_cps):
    """champion_bot_v3 runner logic."""
    curr_cp = checkpoints[next_cp]
    next_cp_coord = checkpoints[(next_cp + 1) % n_cp]

    is_home_run = (state.checkpoints_passed >= total_cps)
    entering_soon = is_entering_cp_soon(x, y, vx, vy, curr_cp[0], curr_cp[1])

    if not is_home_run and entering_soon:
        dist_to_cp = math.hypot(curr_cp[0] - x, curr_cp[1] - y)
        blend = max(0.0, min(1.0, (dist_to_cp - 600.0) / 1400.0))
        tx = blend * curr_cp[0] + (1 - blend) * next_cp_coord[0]
        ty = blend * curr_cp[1] + (1 - blend) * next_cp_coord[1]
    else:
        tx, ty = float(curr_cp[0]), float(curr_cp[1])

    # Drift compensation
    fx = x + vx * DRAG; fy = y + vy * DRAG
    dt = math.hypot(vx * DRAG, vy * DRAG)
    desired = math.degrees(math.atan2(ty - y, tx - x)) % 360
    ang_diff = ((desired - angle + 180) % 360) - 180
    ang_abs = abs(ang_diff)
    dist_target = math.hypot(tx - x, ty - y)
    if dt > 50 and ang_abs < 70 and math.hypot(fx - tx, fy - ty) < dist_target:
        proj_x, proj_y = closest_point_on_line(x, y, tx, ty, fx, fy)
        tx = proj_x + (proj_x - fx)
        ty = proj_y + (proj_y - fy)
        desired = math.degrees(math.atan2(ty - y, tx - x)) % 360
        ang_diff = ((desired - angle + 180) % 360) - 180
        ang_abs = abs(ang_diff)

    if ang_abs >= 90:
        thrust = 0
    else:
        thrust = int(math.ceil(100.0 * math.cos(ang_abs / 180.0)))

    dist_cp = math.hypot(curr_cp[0] - x, curr_cp[1] - y)
    use_boost = (boosts_left[0] > 0 and turn_number == 1) or \
                (boosts_left[0] > 0 and turn_number > 30 and ang_abs < 5.0 and dist_cp > boost_dist_thresh)

    if use_boost:
        boosts_left[0] -= 1
        return f"{int(tx)} {int(ty)} BOOST"
    return f"{int(tx)} {int(ty)} {thrust}"


def heuristic_blocker(bx, by, bvx, bvy, ox, oy, ovx, ovy, o_next_cp, checkpoints, n_cp, runner_next_cp):
    """champion_bot_v3 blocker logic."""
    opp_dist = math.hypot(ox - bx, oy - by)

    # Path-aware intercept check
    opp_vel_h = math.hypot(ovx, ovy)
    is_in_front = False
    if opp_vel_h > 10.0:
        dot = ovx * (bx - ox) + ovy * (by - oy)
        if dot > 0.0:
            is_in_front = True

    should_intercept = (is_in_front and opp_dist < 3500.0) or (opp_dist < 1200.0)

    if should_intercept:
        est_turns = max(1.0, min(4.0, opp_dist / 350.0))
        tx = ox + ovx * est_turns
        ty = oy + ovy * est_turns
        thrust = 100
    else:
        ocp_x, ocp_y = checkpoints[o_next_cp]
        opp_dx = ocp_x - ox; opp_dy = ocp_y - oy
        d = math.hypot(opp_dx, opp_dy)
        if d > 0:
            post_x = ocp_x - (opp_dx / d) * 800.0
            post_y = ocp_y - (opp_dy / d) * 800.0
        else:
            post_x, post_y = ocp_x, ocp_y
        dist_to_post = math.hypot(post_x - bx, post_y - by)
        if dist_to_post > 800:
            tx, ty = post_x, post_y; thrust = 100
        elif dist_to_post > 150:
            tx, ty = post_x, post_y; thrust = int(min(100, max(30, dist_to_post * 0.3)))
        else:
            tx = ox + ovx; ty = oy + ovy
            thrust = 0 if math.hypot(bvx, bvy) > 30 else 10

    return f"{int(tx)} {int(ty)} {thrust}"


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    try:
        laps = int(input())
        checkpoint_count = int(input())
        checkpoints = []
        for _ in range(checkpoint_count):
            cx, cy = map(int, input().split())
            checkpoints.append((cx, cy))
    except Exception:
        sys.exit(0)

    our_states = [PodState(), PodState()]
    opp_states = [PodState(), PodState()]

    boosts_left_nn = [1]   # for NN runner
    boosts_left_h = [1]    # for heuristic fallback runner
    turn_number = 0
    prev_runner_idx = None

    # Pre-compute boost threshold
    max_stretch = 0.0
    for i in range(checkpoint_count):
        cp1 = checkpoints[i]; cp2 = checkpoints[(i + 1) % checkpoint_count]
        d = math.hypot(cp2[0] - cp1[0], cp2[1] - cp1[1])
        if d > max_stretch:
            max_stretch = d
    boost_dist_thresh = max_stretch * 0.70
    total_cps = checkpoint_count * laps

    use_nn = (RUNNER_W is not None and BLOCKER_W is not None)

    while True:
        try:
            x0, y0, vx0, vy0, angle0, next_cp0 = map(int, input().split())
            x1, y1, vx1, vy1, angle1, next_cp1 = map(int, input().split())
            ox0, oy0, ovx0, ovy0, oangle0, onext_cp0 = map(int, input().split())
            ox1, oy1, ovx1, ovy1, oangle1, onext_cp1 = map(int, input().split())
        except Exception:
            break

        turn_number += 1

        for s in our_states + opp_states:
            if s.shield_cooldown > 0:
                s.shield_cooldown -= 1

        update_progress(our_states[0], next_cp0, checkpoint_count)
        update_progress(our_states[1], next_cp1, checkpoint_count)
        update_progress(opp_states[0], onext_cp0, checkpoint_count)
        update_progress(opp_states[1], onext_cp1, checkpoint_count)

        # Determine runner/blocker roles
        score0 = get_progress_score(our_states[0], x0, y0, next_cp0, checkpoints)
        score1 = get_progress_score(our_states[1], x1, y1, next_cp1, checkpoints)
        ROLE_HYSTERESIS = 2000.0
        if prev_runner_idx is None:
            runner_idx = 0 if score0 >= score1 else 1
        else:
            runner_idx = prev_runner_idx
            if [score0, score1][1 - runner_idx] > [score0, score1][runner_idx] + ROLE_HYSTERESIS:
                runner_idx = 1 - runner_idx
        prev_runner_idx = runner_idx
        blocker_idx = 1 - runner_idx

        pods_data = [
            (x0, y0, vx0, vy0, angle0, next_cp0),
            (x1, y1, vx1, vy1, angle1, next_cp1),
        ]

        # Opponent runner is whoever is further ahead
        score_opp0 = get_progress_score(opp_states[0], ox0, oy0, onext_cp0, checkpoints)
        score_opp1 = get_progress_score(opp_states[1], ox1, oy1, onext_cp1, checkpoints)
        opp_runner_data = (ox0, oy0, ovx0, ovy0, oangle0, onext_cp0) if score_opp0 >= score_opp1 else (ox1, oy1, ovx1, ovy1, oangle1, onext_cp1)

        commands = ["", ""]
        rx, ry, rvx, rvy, rangle, rncp = pods_data[runner_idx]
        bx, by, bvx, bvy, bangle, bncp = pods_data[blocker_idx]
        ox, oy, ovx, ovy, oangle, oncp = opp_runner_data

        # ── Runner command ────────────────────────────────────────────────────
        if use_nn:
            feat = runner_features(rx, ry, rvx, rvy, rangle, rncp,
                                   our_states[runner_idx].checkpoints_passed,
                                   our_states[runner_idx].shield_cooldown,
                                   checkpoints, checkpoint_count)
            out = nn_forward(RUNNER_W, feat, RUNNER_ARCH)
            commands[runner_idx] = decode_runner(out, rx, ry, boosts_left_nn)
        else:
            commands[runner_idx] = heuristic_runner(
                rx, ry, rvx, rvy, rangle, rncp,
                our_states[runner_idx], checkpoints, checkpoint_count,
                boosts_left_h, turn_number, boost_dist_thresh, total_cps
            )

        # ── Blocker command ───────────────────────────────────────────────────
        if use_nn:
            feat = blocker_features(bx, by, bvx, bvy, bangle, ox, oy, ovx, ovy, oncp, checkpoints, checkpoint_count)
            out = nn_forward(BLOCKER_W, feat, BLOCKER_ARCH)
            commands[blocker_idx] = decode_blocker(out, bx, by)
        else:
            commands[blocker_idx] = heuristic_blocker(
                bx, by, bvx, bvy, ox, oy, ovx, ovy, oncp,
                checkpoints, checkpoint_count, rncp
            )

        print(commands[0])
        print(commands[1])


if __name__ == "__main__":
    main()
