"""
record_replay.py — Run a sim race and save frames to replay.json for the visualizer.

Usage:
    python3 record_replay.py           # heuristic vs heuristic
    python3 record_replay.py --nn      # NN (if weights ready) vs heuristic
"""

import math, random, json, sys
from simulator import Simulator, Action, MAP_W, MAP_H

# Import heuristic logic from nn_bot
from nn_bot import (
    PodState, update_progress, get_progress_score,
    heuristic_runner, heuristic_blocker,
    closest_point_on_line, is_entering_cp_soon,
)

LAPS = 3
MAX_TURNS = 600
SEED = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 42


def run_replay(seed=SEED):
    random.seed(seed)
    n_cps = random.randint(3, 6)
    checkpoints = Simulator.random_checkpoints(n_cps)

    sim = Simulator(checkpoints, laps=LAPS)
    sim.reset()

    our_states  = [PodState(), PodState()]
    opp_states  = [PodState(), PodState()]
    boosts_a = [1]
    boosts_b = [1]
    prev_runner_a = None

    # Pre-compute boost threshold
    max_stretch = max(
        math.hypot(checkpoints[(i+1) % n_cps][0] - checkpoints[i][0],
                   checkpoints[(i+1) % n_cps][1] - checkpoints[i][1])
        for i in range(n_cps)
    )
    boost_thresh = max_stretch * 0.70
    total_cps = n_cps * LAPS

    frames = []
    turn = 0

    while turn < MAX_TURNS:
        turn += 1
        pods = sim.pods

        for s in our_states + opp_states:
            if s.shield_cooldown > 0:
                s.shield_cooldown -= 1

        update_progress(our_states[0], pods[0].next_cp, n_cps)
        update_progress(our_states[1], pods[1].next_cp, n_cps)
        update_progress(opp_states[0], pods[2].next_cp, n_cps)
        update_progress(opp_states[1], pods[3].next_cp, n_cps)

        # Role assignment team A
        s0 = get_progress_score(our_states[0], pods[0].x, pods[0].y, pods[0].next_cp, checkpoints)
        s1 = get_progress_score(our_states[1], pods[1].x, pods[1].y, pods[1].next_cp, checkpoints)
        if prev_runner_a is None:
            runner_a = 0 if s0 >= s1 else 1
        else:
            runner_a = prev_runner_a
            if [s0,s1][1-runner_a] > [s0,s1][runner_a] + 2000:
                runner_a = 1 - runner_a
        prev_runner_a = runner_a
        blocker_a = 1 - runner_a

        # Role assignment team B
        sb0 = get_progress_score(opp_states[0], pods[2].x, pods[2].y, pods[2].next_cp, checkpoints)
        sb1 = get_progress_score(opp_states[1], pods[3].x, pods[3].y, pods[3].next_cp, checkpoints)
        runner_b = 0 if sb0 >= sb1 else 1
        blocker_b = 1 - runner_b

        rax, ray = pods[runner_a].x, pods[runner_a].y
        rbx, rby = pods[runner_b + 2].x, pods[runner_b + 2].y

        # Team A commands
        r_cmd_a = heuristic_runner(
            pods[runner_a].x, pods[runner_a].y, pods[runner_a].vx, pods[runner_a].vy,
            pods[runner_a].angle, pods[runner_a].next_cp, our_states[runner_a],
            checkpoints, n_cps, boosts_a, turn, boost_thresh, total_cps
        )
        b_cmd_a = heuristic_blocker(
            pods[blocker_a].x, pods[blocker_a].y, pods[blocker_a].vx, pods[blocker_a].vy,
            pods[runner_b+2].x, pods[runner_b+2].y, pods[runner_b+2].vx, pods[runner_b+2].vy,
            pods[runner_b+2].next_cp, checkpoints, n_cps, pods[runner_a].next_cp
        )

        # Team B commands
        r_cmd_b = heuristic_runner(
            pods[runner_b+2].x, pods[runner_b+2].y, pods[runner_b+2].vx, pods[runner_b+2].vy,
            pods[runner_b+2].angle, pods[runner_b+2].next_cp, opp_states[runner_b],
            checkpoints, n_cps, boosts_b, turn, boost_thresh, total_cps
        )
        b_cmd_b = heuristic_blocker(
            pods[blocker_b+2].x, pods[blocker_b+2].y, pods[blocker_b+2].vx, pods[blocker_b+2].vy,
            pods[runner_a].x, pods[runner_a].y, pods[runner_a].vx, pods[runner_a].vy,
            pods[runner_a].next_cp, checkpoints, n_cps, pods[runner_b+2].next_cp
        )

        def parse_cmd(cmd_str):
            parts = cmd_str.split()
            tx, ty = int(parts[0]), int(parts[1])
            thrust_raw = parts[2]
            boost = thrust_raw == "BOOST"
            shield = thrust_raw == "SHIELD"
            thrust = 100 if boost else (0 if shield else int(thrust_raw))
            return Action(tx, ty, thrust, boost, shield)

        acts_a = [None, None]
        acts_a[runner_a] = parse_cmd(r_cmd_a)
        acts_a[blocker_a] = parse_cmd(b_cmd_a)

        acts_b = [None, None]
        acts_b[runner_b] = parse_cmd(r_cmd_b)
        acts_b[blocker_b] = parse_cmd(b_cmd_b)

        # Record frame BEFORE stepping
        frame = {
            "turn": turn,
            "pods": [
                {"x": p.x, "y": p.y, "vx": p.vx, "vy": p.vy,
                 "angle": p.angle, "next_cp": p.next_cp,
                 "cps": p.cps_passed, "shield": p.shield_active}
                for p in pods
            ],
        }
        frames.append(frame)

        _, done, info = sim.step(acts_a, acts_b)
        if done:
            # Record final frame
            frame = {
                "turn": turn + 1,
                "pods": [
                    {"x": p.x, "y": p.y, "vx": p.vx, "vy": p.vy,
                     "angle": p.angle, "next_cp": p.next_cp,
                     "cps": p.cps_passed, "shield": p.shield_active}
                    for p in sim.pods
                ],
                "done": True,
                "winner": info["winner"],
            }
            frames.append(frame)
            break

    replay = {
        "checkpoints": checkpoints,
        "laps": LAPS,
        "total_cps": total_cps,
        "map_w": MAP_W,
        "map_h": MAP_H,
        "frames": frames,
        "winner": sim.winner,
        "total_turns": turn,
        "team_names": ["daemon_slayer", "DarthBoss"],
    }

    with open("replay.json", "w") as f:
        json.dump(replay, f)

    print(f"Saved {len(frames)} frames to replay.json")
    print(f"Winner: team {sim.winner} ({replay['team_names'][sim.winner] if sim.winner is not None else 'none'})")
    print(f"Total turns: {turn}")


if __name__ == "__main__":
    run_replay()
