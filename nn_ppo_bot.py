"""
nn_bot.py — CodinGame Mad Pod Racing submission bot.

Loads weights from nn_weights_ppo.json (produced by nn_train_ppo.py) and
runs inference in pure Python + numpy.  No torch dependency — safe to paste
directly into the CodinGame editor as a single file.

Architecture mirror of ActorCritic in nn_train_ppo.py:
  Shared trunk: Linear(obs) → Tanh → Linear(hidden) → Tanh
  Actor head:   Linear(hidden) → mean  (log_std ignored at inference)
  Action:       tanh(mean)  [deterministic]

Runner  : 14 inputs → 64 → 64 → 4 outputs
Blocker : 12 inputs → 64 → 64 → 3 outputs

Usage on CodinGame:
  1. Paste this file contents into your solution.
   OR
  2. Copy nn_weights_ppo.json contents into WEIGHTS_JSON below,
     then upload nn_bot.py as a single-file solution.

For local testing:
  python3 nn_bot.py
"""

import sys
import math
import json
import os

# ── Constants ─────────────────────────────────────────────────────────────────
MAP_W = 16000
MAP_H =  9000
CP_RADIUS = 600

RUNNER_IN   = 14
RUNNER_H    = 64
RUNNER_ACT  = 4

BLOCKER_IN  = 12
BLOCKER_H   = 64
BLOCKER_ACT = 3

# ── Paste your nn_weights_ppo.json content here for single-file submission ────
# Leave as None to load from file at runtime (useful for local testing).
WEIGHTS_JSON = None   # e.g. WEIGHTS_JSON = {"runner": {...}, "blocker": {...}}

# ── Tiny numpy-free matrix ops ────────────────────────────────────────────────

def tanh(x):
    # Fast element-wise tanh using list comprehension (no numpy needed)
    return [math.tanh(v) for v in x]

def matmul_add(W, b, x):
    """y = W @ x + b  (W is list-of-rows, x and b are lists)."""
    return [sum(W[i][j] * x[j] for j in range(len(x))) + b[i]
            for i in range(len(W))]

def forward(weights, x):
    """
    Two-layer trunk + actor head.  Deterministic (tanh of mean).
    weights: dict with keys trunk_w0, trunk_b0, trunk_w2, trunk_b2,
                              actor_w, actor_b
    x: list[float]  (obs vector)
    returns: list[float]  (tanh-squashed action)
    """
    h = tanh(matmul_add(weights["trunk_w0"], weights["trunk_b0"], x))
    h = tanh(matmul_add(weights["trunk_w2"], weights["trunk_b2"], h))
    mean = matmul_add(weights["actor_w"], weights["actor_b"], h)
    return tanh(mean)


# ── Weight loading ─────────────────────────────────────────────────────────────

def load_weights():
    global WEIGHTS_JSON
    if WEIGHTS_JSON is None:
        # Try loading from file (local testing)
        weight_file = "nn_weights_ppo.json"
        if not os.path.exists(weight_file):
            sys.stderr.write(f"[nn_bot] Weight file '{weight_file}' not found.\n")
            sys.stderr.write("[nn_bot] Falling back to heuristic bot.\n")
            return None, None
        with open(weight_file) as f:
            WEIGHTS_JSON = json.load(f)
    return WEIGHTS_JSON["runner"], WEIGHTS_JSON["blocker"]


# ── Feature extraction (mirrors nn_train_ppo.py exactly) ──────────────────────

def _norm_pos(x, y):
    return x / (MAP_W * 0.5) - 1.0, y / (MAP_H * 0.5) - 1.0

def _norm_vel(vx, vy):
    return vx / 1000.0, vy / 1000.0

def _norm_angle(a):
    return (a / 180.0) - 1.0

def _dist_norm(d):
    return d / 20000.0

def _dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def runner_features(pod, checkpoints, n_cp):
    """
    pod: dict with keys x, y, vx, vy, angle, next_cp, cps_passed, shield_cooldown
    checkpoints: list of (x, y)
    """
    cp_x, cp_y   = checkpoints[pod["next_cp"]]
    ncp_idx      = (pod["next_cp"] + 1) % n_cp
    ncp_x, ncp_y = checkpoints[ncp_idx]

    px, py   = _norm_pos(pod["x"], pod["y"])
    vx, vy   = _norm_vel(pod["vx"], pod["vy"])
    angle_n  = _norm_angle(pod["angle"])

    dcx      = (cp_x - pod["x"]) / MAP_W
    dcy      = (cp_y - pod["y"]) / MAP_H
    dist_cp  = _dist_norm(_dist(cp_x, cp_y, pod["x"], pod["y"]))

    dnx      = (ncp_x - pod["x"]) / MAP_W
    dny      = (ncp_y - pod["y"]) / MAP_H

    desired     = math.degrees(math.atan2(cp_y - pod["y"], cp_x - pod["x"])) % 360.0
    ang_diff    = ((desired - pod["angle"] + 180) % 360) - 180
    ang_diff_n  = ang_diff / 180.0

    speed_n  = _dist(0, 0, pod["vx"], pod["vy"]) / 1000.0
    cps_n    = pod["cps_passed"] / 30.0
    shield_n = pod.get("shield_cooldown", 0) / 3.0

    return [px, py, vx, vy, angle_n,
            dcx, dcy, dist_cp, ang_diff_n,
            dnx, dny,
            speed_n, cps_n, shield_n]


def blocker_features(blocker, opp_runner, checkpoints, n_cp):
    ocp_x, ocp_y = checkpoints[opp_runner["next_cp"]]

    bpx, bpy = _norm_pos(blocker["x"], blocker["y"])
    bvx, bvy = _norm_vel(blocker["vx"], blocker["vy"])

    dx_opp   = (opp_runner["x"] - blocker["x"]) / MAP_W
    dy_opp   = (opp_runner["y"] - blocker["y"]) / MAP_H
    dist_opp = _dist_norm(_dist(opp_runner["x"], opp_runner["y"],
                                 blocker["x"], blocker["y"]))

    ovx, ovy = _norm_vel(opp_runner["vx"], opp_runner["vy"])

    dx_cp    = (ocp_x - blocker["x"]) / MAP_W
    dy_cp    = (ocp_y - blocker["y"]) / MAP_H
    angle_n  = _norm_angle(blocker["angle"])
    speed_n  = _dist(0, 0, blocker["vx"], blocker["vy"]) / 1000.0

    return [bpx, bpy, bvx, bvy,
            dx_opp, dy_opp, dist_opp,
            ovx, ovy,
            dx_cp, dy_cp,
            angle_n]


# ── Action decoding (mirrors nn_train_ppo.py) ─────────────────────────────────

def decode_runner(raw, pod, boosts_left):
    """
    raw: list[4] in [-1,1]
    Returns: (target_x, target_y, thrust_or_BOOST)
    """
    tx     = pod["x"] + raw[0] * 3000.0
    ty     = pod["y"] + raw[1] * 2000.0
    thrust = int((raw[2] + 1.0) * 50.0)
    thrust = max(0, min(100, thrust))

    use_boost = raw[3] > 0.0 and boosts_left[0] > 0
    if use_boost:
        boosts_left[0] -= 1
        return int(tx), int(ty), "BOOST"
    return int(tx), int(ty), thrust


def decode_blocker(raw, pod):
    tx     = pod["x"] + raw[0] * 3000.0
    ty     = pod["y"] + raw[1] * 2000.0
    thrust = int((raw[2] + 1.0) * 50.0)
    thrust = max(0, min(100, thrust))
    return int(tx), int(ty), thrust


# ── Heuristic fallback (used when weights unavailable) ────────────────────────

def heuristic(pod, checkpoints, n_cp, boosts_left, turn):
    cp_x, cp_y   = checkpoints[pod["next_cp"]]
    ncp_x, ncp_y = checkpoints[(pod["next_cp"] + 1) % n_cp]

    dist_to_cp = _dist(pod["x"], pod["y"], cp_x, cp_y)

    px, py, pvx, pvy = pod["x"], pod["y"], pod["vx"], pod["vy"]
    entering = False
    for _ in range(6):
        pvx *= 0.85; pvy *= 0.85; px += pvx; py += pvy
        if _dist(px, py, cp_x, cp_y) <= CP_RADIUS:
            entering = True
            break

    if entering:
        blend = max(0.0, min(1.0, (dist_to_cp - CP_RADIUS) / 1400.0))
        tx = blend * cp_x + (1 - blend) * ncp_x
        ty = blend * cp_y + (1 - blend) * ncp_y
    else:
        tx, ty = float(cp_x), float(cp_y)

    desired  = math.degrees(math.atan2(ty - pod["y"], tx - pod["x"])) % 360.0
    ang_diff = abs(((desired - pod["angle"] + 180) % 360) - 180)
    thrust   = 0 if ang_diff >= 90 else int(math.ceil(100.0 * math.cos(math.radians(ang_diff))))

    use_boost = boosts_left[0] > 0 and turn > 10 and ang_diff < 5.0 and dist_to_cp > 5000
    if use_boost:
        boosts_left[0] -= 1
        return int(tx), int(ty), "BOOST"
    return int(tx), int(ty), thrust


# ── CodinGame game loop ────────────────────────────────────────────────────────

def main():
    runner_w, blocker_w = load_weights()
    use_nn = runner_w is not None

    # First turn: read laps + checkpoint count
    laps   = int(input())
    n_cp   = int(input())
    checkpoints = []
    for _ in range(n_cp):
        cx, cy = map(int, input().split())
        checkpoints.append((cx, cy))

    boosts_left = [1]   # one boost per pod; we only track runner's boost here
    turn = 0
    # Track cps_passed ourselves (CodinGame doesn't give it directly)
    cp_counts = [0, 0]   # our runner, our blocker

    # Pod state from previous turn for cps_passed tracking
    prev_next_cp = [None, None]

    # We need opp pods for blocker features — store from input
    pods = [None, None, None, None]  # [our_runner, our_blocker, opp_runner, opp_blocker]

    while True:
        # CodinGame input: 4 pods per turn
        # Order: our pod 1, our pod 2, opp pod 1, opp pod 2
        for i in range(4):
            line = input().split()
            x, y, vx, vy, angle, next_cp = (int(line[0]), int(line[1]),
                                              int(line[2]), int(line[3]),
                                              int(line[4]), int(line[5]))
            # Track cps_passed for our pods
            cps = 0
            if i < 2:
                if prev_next_cp[i] is None:
                    cps = 0
                else:
                    if next_cp != prev_next_cp[i]:
                        cp_counts[i] += 1
                    cps = cp_counts[i]
                prev_next_cp[i] = next_cp

            pods[i] = {
                "x": x, "y": y,
                "vx": vx, "vy": vy,
                "angle": angle,
                "next_cp": next_cp,
                "cps_passed": cps,
                "shield_cooldown": 0,   # not exposed by CG; safe to leave 0
            }

        # ── Runner output ─────────────────────────────────────────────────────
        if use_nn:
            obs = runner_features(pods[0], checkpoints, n_cp)
            raw = forward(runner_w, obs)
            tx, ty, thrust = decode_runner(raw, pods[0], boosts_left)
        else:
            tx, ty, thrust = heuristic(pods[0], checkpoints, n_cp, boosts_left, turn)

        print(f"{tx} {ty} {thrust}")

        # ── Blocker output ────────────────────────────────────────────────────
        if use_nn:
            obs = blocker_features(pods[1], pods[2], checkpoints, n_cp)
            raw = forward(blocker_w, obs)
            tx, ty, thrust = decode_blocker(raw, pods[1])
        else:
            # Blocker heuristic: aim at opponent runner's next checkpoint to intercept
            opp_cp_x, opp_cp_y = checkpoints[pods[2]["next_cp"]]
            tx = int((pods[2]["x"] + opp_cp_x) / 2)
            ty = int((pods[2]["y"] + opp_cp_y) / 2)
            thrust = 100

        print(f"{tx} {ty} {thrust}")

        turn += 1


if __name__ == "__main__":
    main()
