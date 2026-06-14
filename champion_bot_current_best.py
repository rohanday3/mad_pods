import math
import sys

# ── Configuration & Constants ──────────────────────────────────────────────────
CHECKPOINT_RADIUS = 600
DRAG = 0.85
MAX_ROTATE = 18.0
POD_RADIUS = 400.0

# ── Classes & State Tracking ──────────────────────────────────────────────────

class PodState:
    def __init__(self, pod_id):
        self.pod_id = pod_id
        self.checkpoints_passed = 0
        self.prev_next_cp = -1
        self.shield_cooldown = 0


class PodInfo:
    def __init__(self, x, y, vx, vy, angle, next_cp, state):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.angle = angle
        self.next_cp = next_cp
        self.state = state
        self.speed = math.hypot(vx, vy)


# ── Helper Functions ──────────────────────────────────────────────────────────

def update_progress(state, next_cp, checkpoint_count):
    if state.prev_next_cp == -1:
        state.prev_next_cp = next_cp
        state.checkpoints_passed = 0
    elif next_cp != state.prev_next_cp:
        diff = (next_cp - state.prev_next_cp) % checkpoint_count
        state.checkpoints_passed += diff
        state.prev_next_cp = next_cp


def get_progress_score(state, x, y, next_cp, checkpoints):
    cx, cy = checkpoints[next_cp]
    dist = math.hypot(cx - x, cy - y)
    return state.checkpoints_passed * 20000.0 - dist


def closest_point_on_line(ax, ay, bx, by, px, py):
    """Project point P onto line A->B, return the closest point on the line."""
    abx = bx - ax
    aby = by - ay
    ab_sq = abx * abx + aby * aby
    if ab_sq == 0:
        return px, py
    apx = px - ax
    apy = py - ay
    t = (apx * abx + apy * aby) / ab_sq
    return ax + abx * t, ay + aby * t


def is_going_to_enter_checkpoint_soon(x, y, vx, vy, cp_x, cp_y, frames=6):
    """Predict if the pod will enter the checkpoint radius within N frames, coasting on inertia."""
    pos_x, pos_y = x, y
    vel_x, vel_y = vx, vy
    for _ in range(frames):
        vel_x *= DRAG
        vel_y *= DRAG
        pos_x += vel_x
        pos_y += vel_y
        if math.hypot(pos_x - cp_x, pos_y - cp_y) <= CHECKPOINT_RADIUS:
            return True
    return False


def collision_niceness_score(my_x, my_y, my_vx, my_vy, other_vx, other_vy, cp_x, cp_y):
    """
    Heuristic: estimate how beneficial a collision would be.
    Positive = collision helps us get closer to our checkpoint.
    Negative = collision hurts us (pushes us away).
    Based on gold solution's approach: estimate heuristic post-collision position
    using our velocity + other's velocity * 10 (mass-weighted impulse approximation).
    """
    base_dist = math.hypot(cp_x - my_x, cp_y - my_y)
    # Heuristic next position: our movement + amplified opponent impulse
    heuristic_x = my_x + my_vx * DRAG + other_vx * DRAG * 10.0
    heuristic_y = my_y + my_vy * DRAG + other_vy * DRAG * 10.0
    new_dist = math.hypot(cp_x - heuristic_x, cp_y - heuristic_y)
    return base_dist - new_dist  # positive = beneficial


def will_collide_next_frame(x1, y1, vx1, vy1, x2, y2, vx2, vy2):
    """Check if two pods will be within collision distance after applying friction to velocities."""
    next_x1 = x1 + vx1 * DRAG
    next_y1 = y1 + vy1 * DRAG
    next_x2 = x2 + vx2 * DRAG
    next_y2 = y2 + vy2 * DRAG
    return math.hypot(next_x1 - next_x2, next_y1 - next_y2) <= POD_RADIUS * 2


def get_intercept_checkpoint(blocker, target_opp, runner, checkpoints):
    """Find the best checkpoint to intercept the opponent runner, avoiding our runner's path."""
    num_cp = len(checkpoints)
    curr_opp_cp = target_opp.next_cp

    # Don't block at checkpoints our runner needs
    runner_cps = {runner.next_cp, (runner.next_cp + 1) % num_cp}

    for offset in range(3):
        cp_idx = (curr_opp_cp + offset) % num_cp
        if cp_idx in runner_cps:
            continue

        cp_x, cp_y = checkpoints[cp_idx]

        # Estimate opponent arrival time
        dist_opp = 0.0
        curr_x, curr_y = target_opp.x, target_opp.y
        for i in range(offset + 1):
            next_idx = (curr_opp_cp + i) % num_cp
            nx, ny = checkpoints[next_idx]
            dist_opp += math.hypot(nx - curr_x, ny - curr_y)
            curr_x, curr_y = nx, ny

        opp_speed = math.hypot(target_opp.vx, target_opp.vy)
        turns_opp = dist_opp / max(100.0, opp_speed)

        dist_blocker = math.hypot(cp_x - blocker.x, cp_y - blocker.y)
        blocker_speed = math.hypot(blocker.vx, blocker.vy)
        turns_blocker = dist_blocker / max(100.0, blocker_speed)

        if turns_blocker < turns_opp + 2.0:
            return cp_idx

    return curr_opp_cp


# ── Main Controller ───────────────────────────────────────────────────────────

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

    our_states = [PodState(0), PodState(1)]
    opp_states = [PodState(2), PodState(3)]

    boosts_left = 1
    prev_runner_idx = None
    turn_number = 0

    # Pre-compute: longest stretch for boost threshold (gold uses 70% of max)
    max_stretch = 0.0
    best_boost_cp_idx = 0
    for i in range(checkpoint_count):
        cp1 = checkpoints[i]
        cp2 = checkpoints[(i + 1) % checkpoint_count]
        d = math.hypot(cp2[0] - cp1[0], cp2[1] - cp1[1])
        if d > max_stretch:
            max_stretch = d
            best_boost_cp_idx = i
    boost_distance_threshold = max_stretch * 0.70

    # Total checkpoints to pass for a complete race
    total_checkpoints_for_race = checkpoint_count * laps

    while True:
        try:
            x0, y0, vx0, vy0, angle0, next_cp0 = map(int, input().split())
            x1, y1, vx1, vy1, angle1, next_cp1 = map(int, input().split())
            ox0, oy0, ovx0, ovy0, oangle0, onext_cp0 = map(int, input().split())
            ox1, oy1, ovx1, ovy1, oangle1, onext_cp1 = map(int, input().split())
        except Exception:
            break

        turn_number += 1

        for state in our_states + opp_states:
            if state.shield_cooldown > 0:
                state.shield_cooldown -= 1

        update_progress(our_states[0], next_cp0, checkpoint_count)
        update_progress(our_states[1], next_cp1, checkpoint_count)
        update_progress(opp_states[0], onext_cp0, checkpoint_count)
        update_progress(opp_states[1], onext_cp1, checkpoint_count)

        our_pods = [
            PodInfo(x0, y0, vx0, vy0, angle0, next_cp0, our_states[0]),
            PodInfo(x1, y1, vx1, vy1, angle1, next_cp1, our_states[1])
        ]
        opp_pods = [
            PodInfo(ox0, oy0, ovx0, ovy0, oangle0, onext_cp0, opp_states[0]),
            PodInfo(ox1, oy1, ovx1, ovy1, oangle1, onext_cp1, opp_states[1])
        ]

        score_our0 = get_progress_score(our_states[0], x0, y0, next_cp0, checkpoints)
        score_our1 = get_progress_score(our_states[1], x1, y1, next_cp1, checkpoints)
        score_opp0 = get_progress_score(opp_states[0], ox0, oy0, onext_cp0, checkpoints)
        score_opp1 = get_progress_score(opp_states[1], ox1, oy1, onext_cp1, checkpoints)

        # Role assignment with hysteresis
        ROLE_HYSTERESIS = 2000.0
        if prev_runner_idx is None:
            runner_idx = 0 if score_our0 >= score_our1 else 1
        else:
            runner_idx = prev_runner_idx
            blocker_idx_tmp = 1 - runner_idx
            scores = [score_our0, score_our1]
            if scores[blocker_idx_tmp] > scores[runner_idx] + ROLE_HYSTERESIS:
                runner_idx = blocker_idx_tmp
        prev_runner_idx = runner_idx
        blocker_idx = 1 - runner_idx

        opp_runner_idx = 0 if score_opp0 >= score_opp1 else 1
        opp_blocker_idx = 1 - opp_runner_idx

        runner = our_pods[runner_idx]
        blocker = our_pods[blocker_idx]
        opp_runner = opp_pods[opp_runner_idx]
        opp_blocker = opp_pods[opp_blocker_idx]

        score_our_runner = score_our0 if runner_idx == 0 else score_our1
        score_opp_runner = score_opp0 if opp_runner_idx == 0 else score_opp1

        commands = ["", ""]

        # ── RUNNER LOGIC ──────────────────────────────────────────────────────
        curr_cp = checkpoints[runner.next_cp]
        next_cp_idx = (runner.next_cp + 1) % checkpoint_count
        next_cp = checkpoints[next_cp_idx]

        # Home run detection: on the very last checkpoint, go straight (no drift)
        is_home_run = (runner.state.checkpoints_passed >= total_checkpoints_for_race)

        # Check if we are going to enter the current checkpoint soon (6-frame lookahead)
        entering_soon = is_going_to_enter_checkpoint_soon(
            runner.x, runner.y, runner.vx, runner.vy,
            curr_cp[0], curr_cp[1]
        )

        if not is_home_run and entering_soon:
            # ── INTENTIONAL DRIFT: target next checkpoint early ──
            target_x = float(next_cp[0])
            target_y = float(next_cp[1])
            # Recalculate angle based on next checkpoint
            angle_to_target = math.degrees(math.atan2(target_y - runner.y, target_x - runner.x))
            angle_diff = (angle_to_target - runner.angle + 180) % 360 - 180
            angle_to_target_abs = abs(angle_diff)
        else:
            # ── Standard approach: target current checkpoint ──
            target_x = float(curr_cp[0])
            target_y = float(curr_cp[1])

            # ── DRIFT COMPENSATION (gold technique) ──
            # Project our future position onto the line from current pos to checkpoint
            # Then mirror the lateral deviation to cancel drift
            future_x = runner.x + runner.vx * DRAG
            future_y = runner.y + runner.vy * DRAG
            distance_travelled = math.hypot(runner.vx * DRAG, runner.vy * DRAG)

            angle_to_target = math.degrees(math.atan2(target_y - runner.y, target_x - runner.x))
            angle_diff = (angle_to_target - runner.angle + 180) % 360 - 180
            angle_to_target_abs = abs(angle_diff)

            dist_to_cp = math.hypot(curr_cp[0] - runner.x, curr_cp[1] - runner.y)

            if (distance_travelled > 50 and
                    angle_to_target_abs < 70 and
                    math.hypot(future_x - curr_cp[0], future_y - curr_cp[1]) < dist_to_cp):
                # Find closest point on the line (currentPos -> checkpoint) to futurePos
                proj_x, proj_y = closest_point_on_line(
                    runner.x, runner.y, curr_cp[0], curr_cp[1],
                    future_x, future_y
                )
                # Mirror: target = projection + (projection - future) = 2*projection - future
                target_x = proj_x + (proj_x - future_x)
                target_y = proj_y + (proj_y - future_y)

            # Recalculate angle after drift compensation
            angle_to_target = math.degrees(math.atan2(target_y - runner.y, target_x - runner.x))
            angle_diff = (angle_to_target - runner.angle + 180) % 360 - 180
            angle_to_target_abs = abs(angle_diff)

        # ── THRUST: gold's non-linear cos formula ──
        # thrust = ceil(100 * cos(angle / 180))
        # Note: intentionally NOT using PI — this creates a gentler curve that works better empirically
        if angle_to_target_abs >= 90:
            thrust = 0
        else:
            thrust = int(math.ceil(100.0 * math.cos(angle_to_target_abs / 180.0)))
            thrust = max(0, min(100, thrust))

        # ── BOOST LOGIC ──
        dist = math.hypot(curr_cp[0] - runner.x, curr_cp[1] - runner.y)

        if boosts_left > 0:
            if turn_number == 1:
                # Boost on first frame (gold strategy: Pod 1 always boosts first frame)
                thrust = "BOOST"
                boosts_left -= 1
            elif turn_number > 30 and angle_to_target_abs < 5.0 and dist > boost_distance_threshold:
                # Wait at least 30 frames, then boost on long straights when well-aimed
                # Check no opponent directly in front
                opp_blocking = False
                for opp in opp_pods:
                    dist_opp = math.hypot(opp.x - runner.x, opp.y - runner.y)
                    if dist_opp < 1500:
                        opp_angle = math.degrees(math.atan2(opp.y - runner.y, opp.x - runner.x))
                        angle_to_opp = (opp_angle - runner.angle + 180) % 360 - 180
                        if abs(angle_to_opp) < 25:
                            opp_blocking = True
                            break
                if not opp_blocking:
                    thrust = "BOOST"
                    boosts_left -= 1

        # ── RUNNER SHIELD (collision niceness heuristic) ──
        if thrust != "SHIELD" and runner.state.shield_cooldown == 0:
            runner_cp_x, runner_cp_y = curr_cp
            for opp in opp_pods:
                if will_collide_next_frame(runner.x, runner.y, runner.vx, runner.vy,
                                           opp.x, opp.y, opp.vx, opp.vy):
                    # How does this collision affect US?
                    my_score = collision_niceness_score(
                        runner.x, runner.y, runner.vx, runner.vy,
                        opp.vx, opp.vy,
                        runner_cp_x, runner_cp_y
                    )
                    # How does this collision affect the ENEMY?
                    opp_cp = checkpoints[opp.next_cp]
                    enemy_score = collision_niceness_score(
                        opp.x, opp.y, opp.vx, opp.vy,
                        runner.vx, runner.vy,
                        opp_cp[0], opp_cp[1]
                    )

                    # Shield if: collision hurts us significantly, OR helps enemy significantly
                    BENEFIT_THRESHOLD = 10.0
                    if my_score < -BENEFIT_THRESHOLD or enemy_score > BENEFIT_THRESHOLD:
                        thrust = "SHIELD"
                        runner.state.shield_cooldown = 4
                        break
                    # If collision helps us, keep shield off (let it push us closer)

        commands[runner_idx] = f"{int(target_x)} {int(target_y)} {thrust}"

        # ── BLOCKER LOGIC ─────────────────────────────────────────────────────
        block_cp_idx = get_intercept_checkpoint(blocker, opp_runner, runner, checkpoints)
        block_cp = checkpoints[block_cp_idx]

        opp_dx = block_cp[0] - opp_runner.x
        opp_dy = block_cp[1] - opp_runner.y
        opp_dist = math.hypot(opp_dx, opp_dy)

        # Position the blocker between the opponent runner and their next checkpoint
        if opp_dist > 0:
            post_x = block_cp[0] - (opp_dx / opp_dist) * 800.0
            post_y = block_cp[1] - (opp_dy / opp_dist) * 800.0
        else:
            post_x, post_y = block_cp[0], block_cp[1]

        dist_to_post = math.hypot(post_x - blocker.x, post_y - blocker.y)

        # Decide blocker behaviour
        we_are_trailing = (score_opp_runner > score_our_runner + 1500)

        if we_are_trailing or opp_dist < 3000:
            # AGGRESSIVE: intercept opponent runner's future position
            target_x = opp_runner.x + opp_runner.vx * 2.0
            target_y = opp_runner.y + opp_runner.vy * 2.0
            thrust = 100
        elif dist_to_post > 800:
            # TRAVEL to post at full speed (no more crawling at thrust 20)
            target_x = post_x
            target_y = post_y
            thrust = 100
        elif dist_to_post > 150:
            # Close to post, slow down
            target_x = post_x
            target_y = post_y
            thrust = int(min(100, max(30, dist_to_post * 0.3)))
        else:
            # AT the post: face the opponent runner
            target_x = opp_runner.x + opp_runner.vx
            target_y = opp_runner.y + opp_runner.vy
            thrust = 0 if blocker.speed > 30 else 10

        # ── BLOCKER: avoid colliding with our own runner ──
        for t in [1, 2, 3]:
            eb_x = blocker.x + blocker.vx * t
            eb_y = blocker.y + blocker.vy * t
            er_x = runner.x + runner.vx * t
            er_y = runner.y + runner.vy * t
            if math.hypot(eb_x - er_x, eb_y - er_y) < 950:
                escape_x = eb_x - er_x
                escape_y = eb_y - er_y
                escape_d = math.hypot(escape_x, escape_y)
                if escape_d > 0:
                    target_x = blocker.x + (escape_x / escape_d) * 2000.0
                    target_y = blocker.y + (escape_y / escape_d) * 2000.0
                    if t <= 2:
                        thrust = 0
                break

        # ── BLOCKER SHIELD (collision niceness heuristic) ──
        if thrust != "SHIELD" and blocker.state.shield_cooldown == 0:
            for opp in opp_pods:
                if will_collide_next_frame(blocker.x, blocker.y, blocker.vx, blocker.vy,
                                           opp.x, opp.y, opp.vx, opp.vy):
                    # How does this collision affect the ENEMY?
                    opp_cp = checkpoints[opp.next_cp]
                    enemy_score = collision_niceness_score(
                        opp.x, opp.y, opp.vx, opp.vy,
                        blocker.vx, blocker.vy,
                        opp_cp[0], opp_cp[1]
                    )
                    # How does this collision affect our ally (runner)?
                    ally_collision = will_collide_next_frame(
                        blocker.x, blocker.y, blocker.vx, blocker.vy,
                        runner.x, runner.y, runner.vx, runner.vy
                    )
                    if ally_collision:
                        # Don't shield if it would hurt our runner
                        break

                    BENEFIT_THRESHOLD = 10.0
                    # Shield if collision will HURT the enemy (push them away from their CP)
                    if enemy_score < -BENEFIT_THRESHOLD:
                        thrust = "SHIELD"
                        blocker.state.shield_cooldown = 4
                        break
                    # Also shield if collision would hurt us and not help
                    blocker_cp = checkpoints[blocker.next_cp] if blocker.next_cp < checkpoint_count else checkpoints[0]
                    my_score = collision_niceness_score(
                        blocker.x, blocker.y, blocker.vx, blocker.vy,
                        opp.vx, opp.vy,
                        blocker_cp[0], blocker_cp[1]
                    )
                    if my_score < -BENEFIT_THRESHOLD:
                        thrust = "SHIELD"
                        blocker.state.shield_cooldown = 4
                        break

        commands[blocker_idx] = f"{int(target_x)} {int(target_y)} {thrust}"

        # ── OUTPUT ────────────────────────────────────────────────────────────
        print(commands[0])
        print(commands[1])


if __name__ == "__main__":
    main()
