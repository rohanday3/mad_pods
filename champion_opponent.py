"""
champion_opponent.py — Wraps champion_bot_current_best.py logic into a stateful
class that the simulator can call each turn.

Usage:
    opp = ChampionOpponent(checkpoints, laps)
    opp.reset()
    act0, act1 = opp.get_actions(pods)  # pods = sim.pods[2:4] (opponent pods)
                                         # our_pods = sim.pods[0:2] (NN pods)
"""

import math
from simulator import Action, PodPhysics

CHECKPOINT_RADIUS = 600
DRAG = 0.85
POD_RADIUS = 400.0


# ── Helpers (verbatim from champion_bot_current_best.py) ─────────────────────

def _update_progress(state, next_cp, checkpoint_count):
    if state['prev_next_cp'] == -1:
        state['prev_next_cp'] = next_cp
    elif next_cp != state['prev_next_cp']:
        diff = (next_cp - state['prev_next_cp']) % checkpoint_count
        state['checkpoints_passed'] += diff
        state['prev_next_cp'] = next_cp


def _get_progress_score(state, x, y, next_cp, checkpoints):
    cx, cy = checkpoints[next_cp]
    return state['checkpoints_passed'] * 20000.0 - math.hypot(cx - x, cy - y)


def _closest_point_on_line(ax, ay, bx, by, px, py):
    abx, aby = bx - ax, by - ay
    ab_sq = abx * abx + aby * aby
    if ab_sq == 0:
        return px, py
    t = ((px - ax) * abx + (py - ay) * aby) / ab_sq
    return ax + abx * t, ay + aby * t


def _entering_soon(x, y, vx, vy, cp_x, cp_y, frames=6):
    px, py, pvx, pvy = x, y, vx, vy
    for _ in range(frames):
        pvx *= DRAG; pvy *= DRAG; px += pvx; py += pvy
        if math.hypot(px - cp_x, py - cp_y) <= CHECKPOINT_RADIUS:
            return True
    return False


def _collision_niceness(my_x, my_y, my_vx, my_vy, other_vx, other_vy, cp_x, cp_y):
    base = math.hypot(cp_x - my_x, cp_y - my_y)
    hx = my_x + my_vx * DRAG + other_vx * DRAG * 10.0
    hy = my_y + my_vy * DRAG + other_vy * DRAG * 10.0
    return base - math.hypot(cp_x - hx, cp_y - hy)


def _will_collide(x1, y1, vx1, vy1, x2, y2, vx2, vy2):
    nx1, ny1 = x1 + vx1 * DRAG, y1 + vy1 * DRAG
    nx2, ny2 = x2 + vx2 * DRAG, y2 + vy2 * DRAG
    return math.hypot(nx1 - nx2, ny1 - ny2) <= POD_RADIUS * 2


def _get_intercept_cp(blocker_pod, opp_runner_pod, runner_pod, checkpoints):
    n = len(checkpoints)
    curr = opp_runner_pod.next_cp
    runner_cps = {runner_pod.next_cp, (runner_pod.next_cp + 1) % n}
    for offset in range(3):
        idx = (curr + offset) % n
        if idx in runner_cps:
            continue
        cp_x, cp_y = checkpoints[idx]
        dist_opp = 0.0
        cx, cy = opp_runner_pod.x, opp_runner_pod.y
        for i in range(offset + 1):
            nidx = (curr + i) % n
            nx, ny = checkpoints[nidx]
            dist_opp += math.hypot(nx - cx, ny - cy)
            cx, cy = nx, ny
        opp_spd = max(100.0, math.hypot(opp_runner_pod.vx, opp_runner_pod.vy))
        blocker_spd = max(100.0, math.hypot(blocker_pod.vx, blocker_pod.vy))
        dist_blocker = math.hypot(cp_x - blocker_pod.x, cp_y - blocker_pod.y)
        if dist_blocker / blocker_spd < dist_opp / opp_spd + 2.0:
            return idx
    return curr


# ── Lightweight pod wrapper (mirrors PodInfo from champion_bot_current_best) ──

class _Pod:
    """Thin wrapper around PodPhysics that matches champion_bot's PodInfo API."""
    def __init__(self, phys: PodPhysics, state: dict):
        self.x = phys.x; self.y = phys.y
        self.vx = phys.vx; self.vy = phys.vy
        self.angle = phys.angle
        self.next_cp = phys.next_cp
        self.speed = math.hypot(phys.vx, phys.vy)
        self.state = state  # mutable dict with shield_cooldown etc.


# ── Main class ────────────────────────────────────────────────────────────────

class ChampionOpponent:
    """
    Stateful wrapper around champion_bot_current_best.py.
    Controls pods[2] and pods[3] of the simulator (opponent team).
    pods[0:2] are passed in as "our" pods so the blocker can target them.
    """

    def __init__(self, checkpoints, laps: int):
        self.checkpoints = checkpoints
        self.n_cp = len(checkpoints)
        self.laps = laps
        self.reset()

        # Pre-compute boost threshold
        max_s = max(
            math.hypot(checkpoints[(i+1) % self.n_cp][0] - checkpoints[i][0],
                       checkpoints[(i+1) % self.n_cp][1] - checkpoints[i][1])
            for i in range(self.n_cp)
        )
        self.boost_dist_thresh = max_s * 0.70
        self.total_cps = self.n_cp * laps

    def reset(self):
        # Per-pod mutable state dicts
        self._states = [
            {'checkpoints_passed': 0, 'prev_next_cp': -1, 'shield_cooldown': 0}
            for _ in range(2)
        ]
        self._boosts_left = 1
        self._prev_runner_idx = None
        self._turn = 0

    def get_actions(self, all_pods) -> tuple:
        """
        all_pods: list of 4 PodPhysics from sim.pods.
          [0,1] = NN team (opponent from this bot's perspective)
          [2,3] = this bot's pods

        Returns (Action for pod2, Action for pod3).
        """
        self._turn += 1
        n_cp = self.n_cp
        checkpoints = self.checkpoints

        # Decrement shield cooldowns
        for s in self._states:
            if s['shield_cooldown'] > 0:
                s['shield_cooldown'] -= 1

        # Update checkpoint progress for our two pods (indices 2,3)
        for i in range(2):
            _update_progress(self._states[i], all_pods[2 + i].next_cp, n_cp)

        # Score for role assignment
        s0 = _get_progress_score(self._states[0],
                                  all_pods[2].x, all_pods[2].y, all_pods[2].next_cp, checkpoints)
        s1 = _get_progress_score(self._states[1],
                                  all_pods[3].x, all_pods[3].y, all_pods[3].next_cp, checkpoints)

        HYSTERESIS = 2000.0
        if self._prev_runner_idx is None:
            runner_local = 0 if s0 >= s1 else 1
        else:
            runner_local = self._prev_runner_idx
            if [s0, s1][1 - runner_local] > [s0, s1][runner_local] + HYSTERESIS:
                runner_local = 1 - runner_local
        self._prev_runner_idx = runner_local
        blocker_local = 1 - runner_local

        # Build pod wrappers
        our_pods = [_Pod(all_pods[2 + i], self._states[i]) for i in range(2)]
        # "enemy" from this bot's perspective = NN pods [0,1]
        enemy_pods = [_Pod(all_pods[i], {'checkpoints_passed': 0, 'prev_next_cp': -1, 'shield_cooldown': 0}) for i in range(2)]

        runner = our_pods[runner_local]
        blocker = our_pods[blocker_local]

        # Enemy runner = whichever NN pod is further ahead
        es0 = all_pods[0].cps_passed * 20000.0 - math.hypot(
            checkpoints[all_pods[0].next_cp][0] - all_pods[0].x,
            checkpoints[all_pods[0].next_cp][1] - all_pods[0].y)
        es1 = all_pods[1].cps_passed * 20000.0 - math.hypot(
            checkpoints[all_pods[1].next_cp][0] - all_pods[1].x,
            checkpoints[all_pods[1].next_cp][1] - all_pods[1].y)
        opp_runner = enemy_pods[0] if es0 >= es1 else enemy_pods[1]
        opp_blocker = enemy_pods[1] if es0 >= es1 else enemy_pods[0]

        score_our_runner = s0 if runner_local == 0 else s1
        score_opp_runner = max(es0, es1)

        commands = [None, None]

        # ── RUNNER LOGIC (verbatim champion_bot_current_best) ─────────────────
        curr_cp = checkpoints[runner.next_cp]
        next_cp_coord = checkpoints[(runner.next_cp + 1) % n_cp]
        is_home_run = (runner.state['checkpoints_passed'] >= self.total_cps)
        entering_soon = _entering_soon(runner.x, runner.y, runner.vx, runner.vy,
                                       curr_cp[0], curr_cp[1])

        if not is_home_run and entering_soon:
            target_x = float(next_cp_coord[0])
            target_y = float(next_cp_coord[1])
            angle_to_target = math.degrees(math.atan2(target_y - runner.y, target_x - runner.x))
            angle_diff = (angle_to_target - runner.angle + 180) % 360 - 180
            angle_to_target_abs = abs(angle_diff)
        else:
            target_x = float(curr_cp[0])
            target_y = float(curr_cp[1])
            future_x = runner.x + runner.vx * DRAG
            future_y = runner.y + runner.vy * DRAG
            dt = math.hypot(runner.vx * DRAG, runner.vy * DRAG)
            angle_to_target = math.degrees(math.atan2(target_y - runner.y, target_x - runner.x))
            angle_diff = (angle_to_target - runner.angle + 180) % 360 - 180
            angle_to_target_abs = abs(angle_diff)
            dist_to_cp = math.hypot(curr_cp[0] - runner.x, curr_cp[1] - runner.y)
            if (dt > 50 and angle_to_target_abs < 70 and
                    math.hypot(future_x - curr_cp[0], future_y - curr_cp[1]) < dist_to_cp):
                proj_x, proj_y = _closest_point_on_line(
                    runner.x, runner.y, curr_cp[0], curr_cp[1], future_x, future_y)
                target_x = proj_x + (proj_x - future_x)
                target_y = proj_y + (proj_y - future_y)
                angle_to_target = math.degrees(math.atan2(target_y - runner.y, target_x - runner.x))
                angle_diff = (angle_to_target - runner.angle + 180) % 360 - 180
                angle_to_target_abs = abs(angle_diff)

        if angle_to_target_abs >= 90:
            thrust = 0
        else:
            thrust = int(math.ceil(100.0 * math.cos(angle_to_target_abs / 180.0)))

        dist = math.hypot(curr_cp[0] - runner.x, curr_cp[1] - runner.y)
        use_boost = False
        use_shield = False

        if self._boosts_left > 0:
            if self._turn == 1:
                use_boost = True
                self._boosts_left -= 1
            elif self._turn > 30 and angle_to_target_abs < 5.0 and dist > self.boost_dist_thresh:
                opp_blocking = False
                for ep in enemy_pods:
                    d_opp = math.hypot(ep.x - runner.x, ep.y - runner.y)
                    if d_opp < 1500:
                        a_opp = math.degrees(math.atan2(ep.y - runner.y, ep.x - runner.x))
                        if abs((a_opp - runner.angle + 180) % 360 - 180) < 25:
                            opp_blocking = True
                            break
                if not opp_blocking:
                    use_boost = True
                    self._boosts_left -= 1

        # Shield check for runner
        if not use_boost and not use_shield and runner.state['shield_cooldown'] == 0:
            for ep in enemy_pods:
                if _will_collide(runner.x, runner.y, runner.vx, runner.vy,
                                 ep.x, ep.y, ep.vx, ep.vy):
                    my_score = _collision_niceness(runner.x, runner.y, runner.vx, runner.vy,
                                                   ep.vx, ep.vy, curr_cp[0], curr_cp[1])
                    ep_cp = checkpoints[ep.next_cp]
                    enemy_score = _collision_niceness(ep.x, ep.y, ep.vx, ep.vy,
                                                      runner.vx, runner.vy,
                                                      ep_cp[0], ep_cp[1])
                    if my_score < -10.0 or enemy_score > 10.0:
                        use_shield = True
                        runner.state['shield_cooldown'] = 4
                        break

        runner_action = Action(
            target_x=target_x, target_y=target_y,
            thrust=thrust, boost=use_boost, shield=use_shield
        )

        # ── BLOCKER LOGIC (verbatim champion_bot_current_best) ────────────────
        block_cp_idx = _get_intercept_cp(blocker, opp_runner, runner, checkpoints)
        block_cp = checkpoints[block_cp_idx]
        opp_dx = block_cp[0] - opp_runner.x
        opp_dy = block_cp[1] - opp_runner.y
        opp_dist = math.hypot(opp_dx, opp_dy)

        if opp_dist > 0:
            post_x = block_cp[0] - (opp_dx / opp_dist) * 800.0
            post_y = block_cp[1] - (opp_dy / opp_dist) * 800.0
        else:
            post_x, post_y = block_cp

        dist_to_post = math.hypot(post_x - blocker.x, post_y - blocker.y)
        we_trail = (score_opp_runner > score_our_runner + 1500)

        if we_trail or opp_dist < 3000:
            bx, by = opp_runner.x + opp_runner.vx * 2.0, opp_runner.y + opp_runner.vy * 2.0
            bt = 100
        elif dist_to_post > 800:
            bx, by = post_x, post_y; bt = 100
        elif dist_to_post > 150:
            bx, by = post_x, post_y; bt = int(min(100, max(30, dist_to_post * 0.3)))
        else:
            bx = opp_runner.x + opp_runner.vx
            by = opp_runner.y + opp_runner.vy
            bt = 0 if blocker.speed > 30 else 10

        # Blocker avoids own runner
        for t in [1, 2, 3]:
            eb_x = blocker.x + blocker.vx * t
            eb_y = blocker.y + blocker.vy * t
            er_x = runner.x + runner.vx * t
            er_y = runner.y + runner.vy * t
            if math.hypot(eb_x - er_x, eb_y - er_y) < 950:
                esc_x = eb_x - er_x; esc_y = eb_y - er_y
                esc_d = math.hypot(esc_x, esc_y)
                if esc_d > 0:
                    bx = blocker.x + (esc_x / esc_d) * 2000.0
                    by = blocker.y + (esc_y / esc_d) * 2000.0
                    if t <= 2:
                        bt = 0
                break

        b_shield = False
        if blocker.state['shield_cooldown'] == 0:
            for ep in enemy_pods:
                if _will_collide(blocker.x, blocker.y, blocker.vx, blocker.vy,
                                 ep.x, ep.y, ep.vx, ep.vy):
                    if _will_collide(blocker.x, blocker.y, blocker.vx, blocker.vy,
                                     runner.x, runner.y, runner.vx, runner.vy):
                        break
                    ep_cp = checkpoints[ep.next_cp]
                    e_score = _collision_niceness(ep.x, ep.y, ep.vx, ep.vy,
                                                  blocker.vx, blocker.vy,
                                                  ep_cp[0], ep_cp[1])
                    if e_score < -10.0:
                        b_shield = True
                        blocker.state['shield_cooldown'] = 4
                        break
                    b_cp = checkpoints[blocker.next_cp if blocker.next_cp < n_cp else 0]
                    m_score = _collision_niceness(blocker.x, blocker.y, blocker.vx, blocker.vy,
                                                  ep.vx, ep.vy, b_cp[0], b_cp[1])
                    if m_score < -10.0:
                        b_shield = True
                        blocker.state['shield_cooldown'] = 4
                        break

        blocker_action = Action(target_x=bx, target_y=by, thrust=bt, shield=b_shield)

        commands[runner_local] = runner_action
        commands[blocker_local] = blocker_action
        return commands[0], commands[1]
