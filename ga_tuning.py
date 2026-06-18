"""
Grid-search tuning harness for ga_simulation.py
------------------------------------------------
Pits every parameter combo against a fixed "baseline" bot and records
wins (first to complete all laps).  Runs matches in parallel across CPU cores.

Usage:
    python tune.py                  # uses built-in grid
    python tune.py --laps 3 --turns 400 --matches 10 --jobs 4
"""

import math, random, time, copy, itertools, argparse, sys
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ── Re-import sim primitives from your file ──────────────────────────────────
sys.path.insert(0, "user-data/uploads")
from ga_simulation import (
    SimPod, Move, Solution, CHECKPOINT_RADIUS, DRAG, MAX_ROTATE,
    MAX_THRUST, BOOST_THRUST, MIN_IMPULSE, SHIELD_MASS_MULT, BASE_MASS,
    MAP_DIAGONAL, POD_RADIUS,
    _norm_angle, _clamp, _apply_move, _simulate_movement,
    random_move, fitness,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Parameter bundle
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Params:
    sim_turns:      int   = 6
    population:     int   = 10
    time_limit_ms:  float = 70.0
    k_ahead:        float = 2.0

    def label(self):
        return (f"T{self.sim_turns}_P{self.population}_"
                f"ms{int(self.time_limit_ms)}_K{self.k_ahead:.1f}")


BASELINE = Params()   # your current config — never mutated


# ═══════════════════════════════════════════════════════════════════════════════
# Self-contained GA (reads Params at runtime — no module-level globals)
# ═══════════════════════════════════════════════════════════════════════════════

def _score_sol(sol, our_pods, opp_pods, checkpoints, boosts_left, p: Params):
    """Simulate and score one solution under the given Params."""
    my  = [pod.copy() for pod in our_pods]
    opp = [pod.copy() for pod in opp_pods]
    bl  = boosts_left

    for t in range(p.sim_turns):
        for pi in range(2):
            bl = _apply_move(my[pi], sol.moves[pi][t], False, bl)

        for op in opp:
            cp_x, cp_y = checkpoints[op.next_cp]
            dx = cp_x - op.x;  dy = cp_y - op.y
            dist = math.sqrt(dx*dx + dy*dy) or 1.0
            target = math.degrees(math.atan2(dy, dx)) % 360.0
            rot = _clamp(_norm_angle(target - op.angle), -MAX_ROTATE, MAX_ROTATE)
            op.angle = (op.angle + rot) % 360.0
            rad = math.radians(op.angle)
            op.vx += math.cos(rad) * MAX_THRUST
            op.vy += math.sin(rad) * MAX_THRUST

        _simulate_movement(my + opp, checkpoints)

    return fitness(my, opp, checkpoints), my, opp


def run_ga_params(our_pods, opp_pods, checkpoints, boosts_left,
                  warm_solutions, p: Params):
    """GA loop parameterised by a Params object."""
    start = time.perf_counter()
    limit = p.time_limit_ms / 1000.0

    if warm_solutions:
        population = list(warm_solutions[:p.population])
        while len(population) < p.population:
            population.append(Solution(p.sim_turns))
    else:
        population = [Solution(p.sim_turns) for _ in range(p.population)]

    scored = [(
        _score_sol(s, our_pods, opp_pods, checkpoints, boosts_left, p)[0], s
    ) for s in population]
    scored.sort(key=lambda x: x[0], reverse=True)

    while (time.perf_counter() - start) < limit:
        children_scored = []
        for sc, s in scored:
            child = s.mutate()
            cscore = _score_sol(child, our_pods, opp_pods, checkpoints, boosts_left, p)[0]
            children_scored.append((cscore, child))
        combined = scored + children_scored
        combined.sort(key=lambda x: x[0], reverse=True)
        scored = combined[:p.population]

    best = scored[0][1]
    next_pop = []
    for _, s in scored:
        sc2 = s.copy(); sc2.shift(); next_pop.append(sc2)
    return best, next_pop


# ═══════════════════════════════════════════════════════════════════════════════
# Map generator
# ═══════════════════════════════════════════════════════════════════════════════

def random_map(n_checkpoints=5, seed=None):
    rng = random.Random(seed)
    margin = 1500
    cps = []
    for _ in range(n_checkpoints):
        x = rng.randint(margin, 16000 - margin)
        y = rng.randint(margin, 9000  - margin)
        cps.append((x, y))
    return cps


def initial_pods(checkpoints, angle=0):
    """Place two of our pods and two opponent pods near CP 0, facing it."""
    cx, cy = checkpoints[0]
    our = [
        SimPod(cx - 1000, cy,       0, 0, angle, 0, 0),
        SimPod(cx - 1000, cy + 600, 0, 0, angle, 0, 0),
    ]
    opp = [
        SimPod(cx - 1000, cy - 300, 0, 0, angle, 0, 0),
        SimPod(cx - 1000, cy - 900, 0, 0, angle, 0, 0),
    ]
    return our, opp


# ═══════════════════════════════════════════════════════════════════════════════
# Full race simulation between two Params configs
# ═══════════════════════════════════════════════════════════════════════════════

def _pods_done(pods, laps, checkpoint_count):
    """Return True if any pod has completed all laps."""
    target = laps * checkpoint_count
    return any(p.cps_done >= target for p in pods)


def _apply_best_move(pods, best_sol, checkpoints, boosts_left, our_states_boosted):
    """Apply first move of best solution to pods, return updated boosts_left."""
    for pi in range(2):
        move = best_sol.moves[pi][0]
        pod  = pods[pi]
        new_angle = (pod.angle + _clamp(move.rot, -MAX_ROTATE, MAX_ROTATE)) % 360.0
        pod.angle = new_angle
        if move.shield:
            pod.shield_cd = 4; pod.mass = SHIELD_MASS_MULT
            continue
        pod.mass = BASE_MASS
        if pod.shield_cd > 0:
            pod.shield_cd -= 1
            pod.mass = SHIELD_MASS_MULT if pod.shield_cd > 0 else BASE_MASS
        if move.boost and boosts_left > 0 and not our_states_boosted[pi]:
            thrust = BOOST_THRUST; boosts_left -= 1; our_states_boosted[pi] = True
        else:
            thrust = move.thrust
        rad = math.radians(pod.angle)
        pod.vx += math.cos(rad) * thrust
        pod.vy += math.sin(rad) * thrust
    return boosts_left


def run_match(p_a: Params, p_b: Params,
              laps=3, max_turns=400, seed=None) -> Optional[str]:
    """
    Race A (uses p_a) vs B (uses p_b) on the same random map.
    Returns 'A', 'B', or 'draw'.
    """
    rng = random.Random(seed)
    n_cp = rng.randint(3, 8)
    checkpoints = random_map(n_cp, seed=seed)

    pods_a, pods_b = initial_pods(checkpoints)
    # Give each side their own copies
    pods_a = [p.copy() for p in pods_a]
    pods_b = [p.copy() for p in pods_b]

    warm_a, warm_b = [], []
    bl_a = bl_b = 1
    boost_a = [False, False]
    boost_b = [False, False]

    target = laps * n_cp

    for turn in range(max_turns):
        # --- Side A chooses moves (sees pods_a as ours, pods_b as opponent) ---
        best_a, warm_a = run_ga_params(pods_a, pods_b, checkpoints, bl_a, warm_a, p_a)
        # --- Side B chooses moves (sees pods_b as ours, pods_a as opponent) ---
        best_b, warm_b = run_ga_params(pods_b, pods_a, checkpoints, bl_b, warm_b, p_b)

        # Apply both sides' moves (pre-movement acceleration only)
        bl_a = _apply_best_move(pods_a, best_a, checkpoints, bl_a, boost_a)
        bl_b = _apply_best_move(pods_b, best_b, checkpoints, bl_b, boost_b)

        # Joint physics step (all 4 pods interact)
        _simulate_movement(pods_a + pods_b, checkpoints)

        # Count checkpoints
        for pods in (pods_a, pods_b):
            for p in pods:
                cp_x, cp_y = checkpoints[p.next_cp]
                if math.hypot(p.x - cp_x, p.y - cp_y) <= CHECKPOINT_RADIUS:
                    p.cps_done += 1
                    p.next_cp   = (p.next_cp + 1) % n_cp

        a_done = any(p.cps_done >= target for p in pods_a)
        b_done = any(p.cps_done >= target for p in pods_b)

        if a_done and b_done: return "draw"
        if a_done:            return "A"
        if b_done:            return "B"

    # Tie-break by progress
    def best_progress(pods):
        return max(p.cps_done for p in pods)

    pa, pb = best_progress(pods_a), best_progress(pods_b)
    if pa > pb:   return "A"
    if pb > pa:   return "B"
    return "draw"


# ═══════════════════════════════════════════════════════════════════════════════
# Grid search
# ═══════════════════════════════════════════════════════════════════════════════

GRID = {
    "sim_turns":     [4, 6, 8],
    "population":    [8, 10, 14],
    "time_limit_ms": [70, 80, 90],
    "k_ahead":       [1.5, 2.0, 3.0],
}


def build_combos(grid):
    keys   = list(grid.keys())
    values = list(grid.values())
    combos = []
    for combo in itertools.product(*values):
        p = Params(**dict(zip(keys, combo)))
        combos.append(p)
    return combos


def _worker(args):
    """Run one match; used by multiprocessing Pool."""
    p_candidate, baseline, laps, max_turns, seed = args
    result = run_match(p_candidate, baseline, laps=laps,
                       max_turns=max_turns, seed=seed)
    return result   # 'A'=candidate won, 'B'=baseline won, 'draw'


def grid_search(grid, baseline: Params, matches_per_combo=8,
                laps=3, max_turns=400, jobs=None):
    combos = build_combos(grid)
    # Remove combos identical to baseline
    combos = [c for c in combos if c != baseline]

    total = len(combos) * matches_per_combo
    print(f"\n{'═'*60}")
    print(f"  Grid search: {len(combos)} combos × {matches_per_combo} matches = {total} races")
    print(f"  Baseline: {baseline.label()}")
    print(f"{'═'*60}\n")

    results = {}   # label → {wins, losses, draws}
    jobs = jobs or max(1, cpu_count() - 1)

    for idx, p in enumerate(combos, 1):
        label = p.label()
        # Use different seeds per match for variety
        args_list = [
            (p, baseline, laps, max_turns, idx * 1000 + m)
            for m in range(matches_per_combo)
        ]

        t0 = time.time()
        with Pool(jobs) as pool:
            match_results = pool.map(_worker, args_list)
        elapsed = time.time() - t0

        wins = match_results.count("A")
        losses = match_results.count("B")
        draws = match_results.count("draw")
        win_rate = wins / matches_per_combo

        results[label] = dict(p=p, wins=wins, losses=losses,
                              draws=draws, win_rate=win_rate)

        bar = "█" * wins + "░" * losses + "·" * draws
        print(f"[{idx:3d}/{len(combos)}] {label}")
        print(f"         {bar}  W={wins} L={losses} D={draws}  "
              f"win%={win_rate*100:.0f}%  ({elapsed:.1f}s)\n")

    return results


def print_leaderboard(results, top_n=10):
    ranked = sorted(results.values(), key=lambda r: r["win_rate"], reverse=True)
    print(f"\n{'═'*60}")
    print(f"  TOP {top_n} CONFIGURATIONS (vs baseline)")
    print(f"{'═'*60}")
    print(f"  {'Config':<42} {'W':>3} {'L':>3} {'D':>3} {'Win%':>6}")
    print(f"  {'-'*56}")
    for r in ranked[:top_n]:
        p = r["p"]
        print(f"  {p.label():<42} {r['wins']:>3} {r['losses']:>3} "
              f"{r['draws']:>3} {r['win_rate']*100:>5.0f}%")
    print()

    best = ranked[0]
    bp   = best["p"]
    print("  ★ Best config:")
    print(f"    SIM_TURNS      = {bp.sim_turns}")
    print(f"    POPULATION     = {bp.population}")
    print(f"    TIME_LIMIT_MS  = {bp.time_limit_ms}")
    print(f"    K_AHEAD        = {bp.k_ahead}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser(description="Grid-search tuner for ga_simulation.py")
    ap.add_argument("--laps",    type=int,   default=3,  help="Laps to win (default 3)")
    ap.add_argument("--turns",   type=int,   default=400, help="Max turns per match (default 400)")
    ap.add_argument("--matches", type=int,   default=8,  help="Matches per combo (default 8)")
    ap.add_argument("--jobs",    type=int,   default=None, help="Parallel workers (default: nCPU-1)")
    ap.add_argument("--top",     type=int,   default=10, help="Show top N results")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    results = grid_search(
        GRID,
        BASELINE,
        matches_per_combo=args.matches,
        laps=args.laps,
        max_turns=args.turns,
        jobs=args.jobs,
    )
    print_leaderboard(results, top_n=args.top)