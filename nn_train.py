"""
nn_train.py — Train a neural network for Mad Pod Racing using Evolutionary Strategy (ES).

Key improvements over v1:
 - Dense per-step shaping rewards (distance-closing + CP hits) so the NN
   gets a gradient signal every turn, not just at race end.
 - Curriculum training: phase 1 = steer to one CP, phase 2 = full track no
   opponent, phase 3 = full race with heuristic opponent + blocker.
 - Runner and blocker are trained separately (runner first, then blocker),
   halving the search-space dimensionality at each stage.
 - Imitation pre-training: runner is warm-started by cloning heuristic_action
   for ~50 ES gens before switching to race fitness. This puts weights in a
   sensible region immediately.
 - Larger population (100), fewer generations needed.

Network architecture (runner):
  Input (14 features) -> Hidden (32) -> Hidden (16) -> Output (4)
  Output: [target_dx_norm, target_dy_norm, thrust_norm, use_boost]

Network architecture (blocker):
  Input (12 features) -> Hidden (32) -> Hidden (16) -> Output (3)
  Output: [target_dx_norm, target_dy_norm, thrust_norm]

Usage:
  python3 nn_train.py
"""

import math
import random
import copy
import json
import sys
import os
import time
import numpy as np
from simulator import Simulator, Action, PodPhysics, MAP_W, MAP_H
from champion_opponent import ChampionOpponent

# ── Network config ────────────────────────────────────────────────────────────
RUNNER_IN  = 14
RUNNER_H1  = 32
RUNNER_H2  = 16
RUNNER_OUT = 4

BLOCKER_IN  = 12
BLOCKER_H1  = 32
BLOCKER_H2  = 16
BLOCKER_OUT = 3

# ── ES Hyper-parameters ───────────────────────────────────────────────────────
POPULATION   = 100          # λ: number of perturbations per generation
SIGMA_INIT   = 0.3          # initial noise std dev
SIGMA_DECAY  = 0.9995       # noise annealing
SIGMA_MIN    = 0.01
LEARNING_RATE = 0.05

# Curriculum phases (in generations):
#   Phase 1 (imitation)  : 0  .. PHASE2_START-1  — clone heuristic, runner only
#   Phase 2 (navigation) : PHASE2_START .. PHASE3_START-1 — full track, no opp
#   Phase 3 (racing)     : PHASE3_START .. PHASE4_START-1 — full race + opp, runner only
#   Phase 4 (blocker)    : PHASE4_START .. MAX_GENERATIONS-1 — freeze runner, train blocker
PHASE2_START = 50
PHASE3_START = 150
PHASE4_START = 300
MAX_GENERATIONS = 500

ROLLOUTS_PER_EVAL = 3   # circuits per fitness evaluation (phases 2-4)
LAPS = 3
MAX_TURNS = 1200        # cap per rollout

# Shaping reward weights
W_DIST_CLOSE  = 0.02    # reward per unit of distance closed toward next CP per turn
W_CP_HIT      = 800.0   # reward per checkpoint passed (our runner)
W_LAP_BONUS   = 3000.0  # extra reward per lap completed
W_WIN_BONUS   = 15000.0 # win/loss bonus (reduced vs v1 so shaping dominates early)
W_SPEED       = 0.5     # reward for turns saved vs MAX_TURNS


# ── Neural Network helpers ────────────────────────────────────────────────────

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def make_params(in_sz, h1, h2, out_sz) -> np.ndarray:
    """Random Xavier-initialized flat parameter vector."""
    shapes = [
        (h1, in_sz), (h1,),
        (h2, h1),   (h2,),
        (out_sz, h2),(out_sz,)
    ]
    params = []
    for shape in shapes:
        if len(shape) == 2:
            scale = math.sqrt(2.0 / shape[1])
            params.append(np.random.randn(*shape) * scale)
        else:
            params.append(np.zeros(shape))
    return np.concatenate([p.flatten() for p in params])


def unpack_params(flat: np.ndarray, in_sz, h1, h2, out_sz):
    idx = 0
    def take(shape):
        nonlocal idx
        n = int(np.prod(shape))
        arr = flat[idx:idx+n].reshape(shape)
        idx += n
        return arr
    W1 = take((h1, in_sz));  b1 = take((h1,))
    W2 = take((h2, h1));     b2 = take((h2,))
    W3 = take((out_sz, h2)); b3 = take((out_sz,))
    return W1, b1, W2, b2, W3, b3


def forward(flat: np.ndarray, x: np.ndarray, in_sz, h1, h2, out_sz) -> np.ndarray:
    W1, b1, W2, b2, W3, b3 = unpack_params(flat, in_sz, h1, h2, out_sz)
    h = relu(W1 @ x + b1)
    h = relu(W2 @ h + b2)
    return tanh(W3 @ h + b3)


# ── Feature extraction ────────────────────────────────────────────────────────

def _norm_pos(x, y):
    return x / (MAP_W * 0.5) - 1.0, y / (MAP_H * 0.5) - 1.0


def _norm_vel(vx, vy):
    return vx / 1000.0, vy / 1000.0


def _norm_angle(a):
    return (a / 180.0) - 1.0


def _dist_norm(d):
    return d / 20000.0


def runner_features(pod: PodPhysics, checkpoints, n_cp) -> np.ndarray:
    cp_x, cp_y = checkpoints[pod.next_cp]
    ncp_idx = (pod.next_cp + 1) % n_cp
    ncp_x, ncp_y = checkpoints[ncp_idx]

    px, py = _norm_pos(pod.x, pod.y)
    vx, vy = _norm_vel(pod.vx, pod.vy)
    angle_n = _norm_angle(pod.angle)

    dcx = (cp_x - pod.x) / MAP_W
    dcy = (cp_y - pod.y) / MAP_H
    dist_cp = _dist_norm(math.hypot(cp_x - pod.x, cp_y - pod.y))

    dnx = (ncp_x - pod.x) / MAP_W
    dny = (ncp_y - pod.y) / MAP_H

    desired = math.degrees(math.atan2(cp_y - pod.y, cp_x - pod.x)) % 360.0
    ang_diff = ((desired - pod.angle + 180) % 360) - 180
    ang_diff_n = ang_diff / 180.0

    speed_n   = math.hypot(pod.vx, pod.vy) / 1000.0
    cps_n     = pod.cps_passed / 30.0
    shield_n  = pod.shield_cooldown / 3.0

    return np.array([
        px, py, vx, vy, angle_n,
        dcx, dcy, dist_cp, ang_diff_n,
        dnx, dny,
        speed_n, cps_n, shield_n
    ], dtype=np.float32)


def blocker_features(blocker: PodPhysics, opp_runner: PodPhysics, checkpoints, n_cp) -> np.ndarray:
    ocp_x, ocp_y = checkpoints[opp_runner.next_cp]

    bpx, bpy = _norm_pos(blocker.x, blocker.y)
    bvx, bvy = _norm_vel(blocker.vx, blocker.vy)

    dx_opp  = (opp_runner.x - blocker.x) / MAP_W
    dy_opp  = (opp_runner.y - blocker.y) / MAP_H
    dist_opp = _dist_norm(math.hypot(opp_runner.x - blocker.x, opp_runner.y - blocker.y))

    ovx, ovy = _norm_vel(opp_runner.vx, opp_runner.vy)

    dx_cp = (ocp_x - blocker.x) / MAP_W
    dy_cp = (ocp_y - blocker.y) / MAP_H

    angle_n = _norm_angle(blocker.angle)
    speed_n = math.hypot(blocker.vx, blocker.vy) / 1000.0

    return np.array([
        bpx, bpy, bvx, bvy,
        dx_opp, dy_opp, dist_opp,
        ovx, ovy,
        dx_cp, dy_cp,
        angle_n
    ], dtype=np.float32)


# ── Heuristic action (opponent + imitation target) ────────────────────────────

def heuristic_action(pod: PodPhysics, checkpoints, n_cp,
                     boosts_left_ref: list, turn: int) -> Action:
    """Simplified champion_bot_v3 runner logic."""
    cp_x, cp_y   = checkpoints[pod.next_cp]
    ncp_x, ncp_y = checkpoints[(pod.next_cp + 1) % n_cp]

    dist_to_cp  = math.hypot(cp_x - pod.x, cp_y - pod.y)
    entering_soon = False
    px, py, pvx, pvy = pod.x, pod.y, pod.vx, pod.vy
    for _ in range(6):
        pvx *= 0.85; pvy *= 0.85; px += pvx; py += pvy
        if math.hypot(px - cp_x, py - cp_y) <= 600:
            entering_soon = True
            break

    if entering_soon:
        blend = max(0.0, min(1.0, (dist_to_cp - 600.0) / 1400.0))
        tx = blend * cp_x + (1 - blend) * ncp_x
        ty = blend * cp_y + (1 - blend) * ncp_y
    else:
        tx, ty = float(cp_x), float(cp_y)

    desired  = math.degrees(math.atan2(ty - pod.y, tx - pod.x)) % 360.0
    ang_diff = abs(((desired - pod.angle + 180) % 360) - 180)

    thrust = 0 if ang_diff >= 90 else int(math.ceil(100.0 * math.cos(ang_diff / 180.0)))

    use_boost = (boosts_left_ref[0] > 0 and turn > 10
                 and ang_diff < 5.0 and dist_to_cp > 5000)
    if use_boost:
        boosts_left_ref[0] -= 1

    return Action(target_x=tx, target_y=ty, thrust=thrust, boost=use_boost)


# ── NN action decoders ────────────────────────────────────────────────────────

def runner_action(flat: np.ndarray, features: np.ndarray,
                  boosts_left: list, sim: Simulator) -> Action:
    out = forward(flat, features, RUNNER_IN, RUNNER_H1, RUNNER_H2, RUNNER_OUT)
    tx  = float(features[0] * MAP_W * 0.5 + MAP_W * 0.5) + out[0] * 3000.0
    ty  = float(features[1] * MAP_H * 0.5 + MAP_H * 0.5) + out[1] * 2000.0
    thrust = int((out[2] + 1.0) * 50.0)
    thrust = max(0, min(100, thrust))
    use_boost = bool(out[3] > 0.5 and boosts_left[0] > 0)
    if use_boost:
        boosts_left[0] -= 1
    return Action(target_x=tx, target_y=ty, thrust=thrust, boost=use_boost)


def blocker_action(flat: np.ndarray, features: np.ndarray) -> Action:
    out = forward(flat, features, BLOCKER_IN, BLOCKER_H1, BLOCKER_H2, BLOCKER_OUT)
    tx  = float(features[0] * MAP_W * 0.5 + MAP_W * 0.5) + out[0] * 3000.0
    ty  = float(features[1] * MAP_H * 0.5 + MAP_H * 0.5) + out[1] * 2000.0
    thrust = int((out[2] + 1.0) * 50.0)
    thrust = max(0, min(100, thrust))
    return Action(target_x=tx, target_y=ty, thrust=thrust)


# ── Fitness functions ─────────────────────────────────────────────────────────

def _dist_to_cp(pod, checkpoints):
    cp_x, cp_y = checkpoints[pod.next_cp]
    return math.hypot(cp_x - pod.x, cp_y - pod.y)


def evaluate_imitation(runner_flat: np.ndarray, n_rollouts: int = 2) -> float:
    """
    Phase 1 fitness: negative MSE between NN output and heuristic action.
    We sample random pod states so the NN learns to steer correctly
    across the whole state space, not just one trajectory.
    """
    total_loss = 0.0
    for _ in range(n_rollouts):
        checkpoints = Simulator.random_checkpoints(random.randint(3, 6))
        sim = Simulator(checkpoints, laps=LAPS)
        sim.reset()
        boosts_h = [1]
        boosts_nn = [1]

        for turn in range(300):          # short rollout for speed
            pod = sim.pods[0]
            feat = runner_features(pod, checkpoints, sim.n_cp)
            h_act = heuristic_action(pod, checkpoints, sim.n_cp, boosts_h, turn)
            nn_act = runner_action(runner_flat, feat, boosts_nn, sim)

            # Normalise heuristic targets the same way the NN decodes them
            # so we can compare in a common space.
            h_thrust_norm = h_act.thrust / 100.0       # [0, 1]
            nn_thrust_norm = nn_act.thrust / 100.0

            h_tx_norm  = (h_act.target_x - pod.x) / 3000.0
            h_ty_norm  = (h_act.target_y - pod.y) / 2000.0
            nn_tx_norm = (nn_act.target_x - pod.x) / 3000.0
            nn_ty_norm = (nn_act.target_y - pod.y) / 2000.0

            loss = ((h_tx_norm  - nn_tx_norm) ** 2 +
                    (h_ty_norm  - nn_ty_norm) ** 2 +
                    (h_thrust_norm - nn_thrust_norm) ** 2)
            total_loss += loss

            # Advance sim with heuristic action so states evolve naturally
            dummy_act = Action(target_x=h_act.target_x, target_y=h_act.target_y,
                               thrust=h_act.thrust, boost=h_act.boost)
            sim.step([dummy_act, dummy_act], [dummy_act, dummy_act])

    return -(total_loss / n_rollouts)   # higher = better


def evaluate_navigation(runner_flat: np.ndarray, n_rollouts: int = ROLLOUTS_PER_EVAL) -> float:
    """
    Phase 2 fitness: pure navigation, no opponent, dense shaping.
    Blocker just mimics heuristic so it doesn't interfere.
    """
    total = 0.0
    for _ in range(n_rollouts):
        checkpoints = Simulator.random_checkpoints(random.randint(3, 6))
        sim = Simulator(checkpoints, laps=LAPS)
        sim.reset()
        our_boosts = [1]
        dummy_boosts = [0]

        prev_dist = _dist_to_cp(sim.pods[0], checkpoints)
        shaping = 0.0

        for turn in range(MAX_TURNS):
            pods = sim.pods
            r_feat = runner_features(pods[0], checkpoints, sim.n_cp)
            r_act  = runner_action(runner_flat, r_feat, our_boosts, sim)

            # Blocker follows heuristic so it doesn't trip on our runner
            b_act = heuristic_action(pods[1], checkpoints, sim.n_cp, dummy_boosts, turn)

            # No real opponent — feed dummy actions for opponent pods
            dummy = Action(target_x=pods[2].x, target_y=pods[2].y, thrust=0)
            _, done, info = sim.step([r_act, b_act], [dummy, dummy])

            # ── Per-step shaping ──────────────────────────────────────────
            new_dist = _dist_to_cp(pods[0], checkpoints)
            shaping += (prev_dist - new_dist) * W_DIST_CLOSE
            prev_dist = new_dist

            if done:
                break

        cp_score  = pods[0].cps_passed * W_CP_HIT
        lap_score = (pods[0].cps_passed // sim.n_cp) * W_LAP_BONUS
        total += shaping + cp_score + lap_score

    return total / n_rollouts


def evaluate_racing(runner_flat: np.ndarray, blocker_flat: np.ndarray,
                    n_rollouts: int = ROLLOUTS_PER_EVAL) -> float:
    """
    Phase 3 / 4 fitness: full race vs heuristic opponent with dense shaping.
    In phase 3 the blocker_flat weights may be the fixed warm-start.
    In phase 4 runner_flat is frozen and blocker_flat is being optimised.
    """
    total = 0.0
    for _ in range(n_rollouts):
        checkpoints = Simulator.random_checkpoints(random.randint(3, 6))
        sim = Simulator(checkpoints, laps=LAPS)
        sim.reset()
        our_boosts = [1]
        opp = ChampionOpponent(checkpoints, laps=LAPS)

        prev_our_dist = _dist_to_cp(sim.pods[0], checkpoints)
        prev_opp_dist = _dist_to_cp(sim.pods[2], checkpoints)
        shaping = 0.0
        prev_our_cps = 0

        for turn in range(MAX_TURNS):
            pods = sim.pods

            r_feat = runner_features(pods[0], checkpoints, sim.n_cp)
            r_act  = runner_action(runner_flat, r_feat, our_boosts, sim)

            b_feat = blocker_features(pods[1], pods[2], checkpoints, sim.n_cp)
            b_act  = blocker_action(blocker_flat, b_feat)

            o_act1, o_act2 = opp.get_actions(pods)
            _, done, info = sim.step([r_act, b_act], [o_act1, o_act2])

            # ── Per-step shaping ──────────────────────────────────────────
            new_our_dist = _dist_to_cp(pods[0], checkpoints)
            new_opp_dist = _dist_to_cp(pods[2], checkpoints)

            # Reward our runner closing distance faster than opponent
            our_gain = (prev_our_dist - new_our_dist) * W_DIST_CLOSE
            opp_gain = (prev_opp_dist - new_opp_dist) * W_DIST_CLOSE * 0.5  # penalise less
            shaping += our_gain - opp_gain

            prev_our_dist = new_our_dist
            prev_opp_dist = new_opp_dist

            # Reward each CP our runner passes
            new_cps = pods[0].cps_passed
            if new_cps > prev_our_cps:
                shaping += (new_cps - prev_our_cps) * W_CP_HIT
                prev_our_cps = new_cps

            if done:
                break

        our_best  = max(pods[0].cps_passed, pods[1].cps_passed)
        opp_best  = max(pods[2].cps_passed, pods[3].cps_passed)
        cp_advantage = (our_best - opp_best) * W_CP_HIT
        speed_bonus  = (MAX_TURNS - sim.turn) * W_SPEED
        win_bonus    = (W_WIN_BONUS if info["winner"] == 0 else
                       -W_WIN_BONUS if info["winner"] == 1 else 0.0)

        total += shaping + cp_advantage + speed_bonus + win_bonus

    return total / n_rollouts


# ── ES update (shared) ────────────────────────────────────────────────────────

def es_update(flat: np.ndarray, noise_list, fitnesses: np.ndarray,
              sigma: float) -> np.ndarray:
    """Rank-normalised ES gradient step."""
    ranks = np.argsort(np.argsort(fitnesses)).astype(float)
    ranks = (ranks / (len(fitnesses) - 1)) - 0.5   # [-0.5, 0.5]
    grad  = sum(ranks[i] * noise_list[i] for i in range(len(noise_list)))
    grad /= len(noise_list)
    return flat + LEARNING_RATE / sigma * grad


# ── Live state writer ────────────────────────────────────────────────────────

def record_live_race(runner_flat, blocker_flat, seed=0):
    random.seed(seed)
    checkpoints = Simulator.random_checkpoints(random.randint(3, 5))
    sim = Simulator(checkpoints, laps=LAPS)
    sim.reset()
    our_boosts = [1]
    opp = ChampionOpponent(checkpoints, laps=LAPS)
    frames = []
    for turn in range(MAX_TURNS):
        pods = sim.pods
        frames.append({"t": turn, "pods": [
            {"x": p.x, "y": p.y, "vx": p.vx, "vy": p.vy,
             "angle": p.angle, "next_cp": p.next_cp,
             "cps": p.cps_passed, "shield": p.shield_active}
            for p in pods
        ]})
        r_feat = runner_features(pods[0], checkpoints, sim.n_cp)
        r_act  = runner_action(runner_flat, r_feat, our_boosts, sim)
        b_feat = blocker_features(pods[1], pods[2], checkpoints, sim.n_cp)
        b_act  = blocker_action(blocker_flat, b_feat)
        o_act1, o_act2 = opp.get_actions(pods)
        _, done, info = sim.step([r_act, b_act], [o_act1, o_act2])
        if done:
            break
    return checkpoints, frames, sim.winner


def write_live_state(gen, gen_fitness, best_fitness, sigma, elapsed,
                     history, runner_flat, blocker_flat, phase):
    try:
        checkpoints, frames, winner = record_live_race(
            runner_flat, blocker_flat, seed=gen % 20)
        state = {
            "gen": gen,
            "phase": phase,
            "gen_fitness": float(gen_fitness),
            "best_fitness": float(best_fitness),
            "sigma": float(sigma),
            "elapsed": float(elapsed),
            "history": history,
            "checkpoints": checkpoints,
            "laps": LAPS,
            "total_cps": len(checkpoints) * LAPS,
            "map_w": MAP_W,
            "map_h": MAP_H,
            "frames": frames,
            "winner": winner,
            "team_names": ["NN Bot", "Heuristic"],
            "max_gens": MAX_GENERATIONS,
            "runner_weights": runner_flat.tolist(),
            "blocker_weights": blocker_flat.tolist(),
        }
        tmp = "live_state.tmp.json"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, "live_state.json")
    except Exception as e:
        pass  # Never let the viz crash training


# ── ES Training Loop ──────────────────────────────────────────────────────────

def get_phase(gen: int) -> int:
    if gen < PHASE2_START:  return 1
    if gen < PHASE3_START:  return 2
    if gen < PHASE4_START:  return 3
    return 4


def phase_label(phase: int) -> str:
    return {1: "imitation", 2: "navigation", 3: "racing (runner)",
            4: "racing (blocker)"}[phase]


def train():
    print("Initialising parameter vectors...")
    runner_flat  = make_params(RUNNER_IN,  RUNNER_H1,  RUNNER_H2,  RUNNER_OUT)
    blocker_flat = make_params(BLOCKER_IN, BLOCKER_H1, BLOCKER_H2, BLOCKER_OUT)

    best_fitness  = -1e18
    best_runner   = runner_flat.copy()
    best_blocker  = blocker_flat.copy()
    sigma         = SIGMA_INIT
    history       = []   # [(gen, gen_fitness, best_fitness)]

    print(f"Starting ES training: {MAX_GENERATIONS} generations, pop={POPULATION}")
    print(f"Runner params : {len(runner_flat)}")
    print(f"Blocker params: {len(blocker_flat)}")
    print(f"Curriculum: phase1=imitation(0-{PHASE2_START-1}), "
          f"phase2=navigation({PHASE2_START}-{PHASE3_START-1}), "
          f"phase3=racing-runner({PHASE3_START}-{PHASE4_START-1}), "
          f"phase4=racing-blocker({PHASE4_START}-{MAX_GENERATIONS-1})")
    print("Live dashboard: open race_viewer.html in your browser\n")

    t0 = time.time()

    for gen in range(MAX_GENERATIONS):
        sigma = max(SIGMA_MIN, sigma * SIGMA_DECAY)
        phase = get_phase(gen)

        # ── Reset best tracker when phase changes so comparisons are fair ──
        if gen in (PHASE2_START, PHASE3_START, PHASE4_START):
            best_fitness = -1e18
            print(f"\n>>> Entering phase {phase}: {phase_label(phase)} <<<\n")

        # ── Generate perturbations (only perturb the active network) ────────
        if phase <= 3:
            # Optimise runner only
            runner_noise = [np.random.randn(*runner_flat.shape) for _ in range(POPULATION)]
            fitnesses    = np.array([
                evaluate_imitation(runner_flat + sigma * runner_noise[i])
                if phase == 1
                else evaluate_navigation(runner_flat + sigma * runner_noise[i])
                if phase == 2
                else evaluate_racing(runner_flat + sigma * runner_noise[i], blocker_flat, n_rollouts=2)
                for i in range(POPULATION)
            ])
            runner_flat = es_update(runner_flat, runner_noise, fitnesses, sigma)
            # Evaluate mean (no noise) for tracking
            if phase == 1:
                gen_fitness = evaluate_imitation(runner_flat)
            elif phase == 2:
                gen_fitness = evaluate_navigation(runner_flat, n_rollouts=ROLLOUTS_PER_EVAL)
            else:
                gen_fitness = evaluate_racing(runner_flat, blocker_flat, n_rollouts=ROLLOUTS_PER_EVAL)

        else:
            # Phase 4: freeze runner, optimise blocker
            blocker_noise = [np.random.randn(*blocker_flat.shape) for _ in range(POPULATION)]
            fitnesses     = np.array([
                evaluate_racing(runner_flat, blocker_flat + sigma * blocker_noise[i], n_rollouts=2)
                for i in range(POPULATION)
            ])
            blocker_flat = es_update(blocker_flat, blocker_noise, fitnesses, sigma)
            gen_fitness  = evaluate_racing(runner_flat, blocker_flat, n_rollouts=ROLLOUTS_PER_EVAL)

        elapsed = time.time() - t0

        if gen_fitness > best_fitness:
            best_fitness = gen_fitness
            best_runner  = runner_flat.copy()
            best_blocker = blocker_flat.copy()
            mark = " *** NEW BEST ***"
        else:
            mark = ""

        history.append([gen, float(gen_fitness), float(best_fitness)])
        if len(history) > 500:
            history = history[-500:]

        if gen % 5 == 0 or mark:
            write_live_state(gen, gen_fitness, best_fitness, sigma,
                             elapsed, history, best_runner, best_blocker, phase)

        if gen % 10 == 0 or mark:
            print(f"Gen {gen:4d} [ph{phase}] | fit={gen_fitness:9.1f} "
                  f"| best={best_fitness:9.1f} | σ={sigma:.4f} | {elapsed:.0f}s{mark}")

        sys.stdout.flush()

    print("\n=== Training complete ===")
    print(f"Best fitness: {best_fitness:.1f}")

    # ── Save weights ──────────────────────────────────────────────────────────
    weights = {
        "runner":        best_runner.tolist(),
        "blocker":       best_blocker.tolist(),
        "runner_arch":   [RUNNER_IN,  RUNNER_H1,  RUNNER_H2,  RUNNER_OUT],
        "blocker_arch":  [BLOCKER_IN, BLOCKER_H1, BLOCKER_H2, BLOCKER_OUT],
    }
    with open("nn_weights.json", "w") as f:
        json.dump(weights, f)
    print("Weights saved to nn_weights.json")

    print("\n# === RUNNER WEIGHTS (paste into nn_bot.py) ===")
    print(f"RUNNER_W = {best_runner.tolist()}")
    print(f"\n# === BLOCKER WEIGHTS ===")
    print(f"BLOCKER_W = {best_blocker.tolist()}")


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    train()