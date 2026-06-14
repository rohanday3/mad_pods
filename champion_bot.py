import math
import sys

# ── Configuration & Constants ──────────────────────────────────────────────────
CHECKPOINT_RADIUS = 600
DRAG = 0.85
MAX_ROTATE = 18.0

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
        # Calculate checkpoints passed using modular difference
        diff = (next_cp - state.prev_next_cp) % checkpoint_count
        state.checkpoints_passed += diff
        state.prev_next_cp = next_cp


def get_progress_score(state, x, y, next_cp, checkpoints):
    cx, cy = checkpoints[next_cp]
    dist = math.hypot(cx - x, cy - y)
    # Give high weight to checkpoints passed, subtract distance to next checkpoint
    return state.checkpoints_passed * 20000.0 - dist


def turn_angle_between(prev_cp, curr_cp, next_cp):
    """
    Compute turn angle at curr_cp in degrees.
    0.0 = straight line, 180.0 = complete U-turn.
    """
    v1_x = curr_cp[0] - prev_cp[0]
    v1_y = curr_cp[1] - prev_cp[1]
    v2_x = next_cp[0] - curr_cp[0]
    v2_y = next_cp[1] - curr_cp[1]
    
    l1 = math.hypot(v1_x, v1_y)
    l2 = math.hypot(v2_x, v2_y)
    if l1 == 0 or l2 == 0:
        return 0.0
        
    dot = (v1_x * v2_x + v1_y * v2_y) / (l1 * l2)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def get_braking_distance(current_speed, target_speed):
    """Calculate the distance covered while coasting (thrust=0) down to target_speed."""
    d = 0.0
    v = current_speed
    while v > target_speed:
        d += v
        v *= DRAG
        if v < 1.0:
            break
    return d


def will_collide_this_turn(x1, y1, vx1, vy1, x2, y2, vx2, vy2):
    """
    Find if a collision occurs between two circles of radius 400 during the next turn.
    Returns (True, t) if collision occurs, where t is the fraction of the turn.
    """
    dx = x1 - x2
    dy = y1 - y2
    dvx = vx1 - vx2
    dvy = vy1 - vy2
    
    # Already colliding
    if math.hypot(dx, dy) < 800:
        return True, 0.0
        
    a = dvx*dvx + dvy*dvy
    if a == 0:
        return False, 0.0
        
    b = 2.0 * (dx*dvx + dy*dvy)
    t = -b / (2.0 * a)
    
    if 0.0 <= t <= 1.0:
        min_dist_sq = (dx + t*dvx)**2 + (dy + t*dvy)**2
        if min_dist_sq < 640000:  # 800^2
            return True, t
            
    return False, 0.0


def get_intercept_checkpoint(blocker, target_opp, runner, checkpoints):
    """
    Select the optimal checkpoint to intercept the opponent.
    NEVER pick a checkpoint that our runner is heading towards soon.
    """
    num_cp = len(checkpoints)
    curr_opp_cp = target_opp.next_cp
    
    runner_cps = {runner.next_cp, (runner.next_cp + 1) % num_cp}
    
    for offset in range(3):
        cp_idx = (curr_opp_cp + offset) % num_cp
        if cp_idx in runner_cps:
            continue
            
        cp_x, cp_y = checkpoints[cp_idx]
        
        # Estimate turns for opponent to reach cp_idx along the track
        dist_opp = 0.0
        curr_x, curr_y = target_opp.x, target_opp.y
        for i in range(offset + 1):
            next_idx = (curr_opp_cp + i) % num_cp
            nx, ny = checkpoints[next_idx]
            dist_opp += math.hypot(nx - curr_x, ny - curr_y)
            curr_x, curr_y = nx, ny
        
        opp_speed = math.hypot(target_opp.vx, target_opp.vy)
        turns_opp = dist_opp / max(100.0, opp_speed)
        
        # Estimate turns for blocker to reach cp_idx directly
        dist_blocker = math.hypot(cp_x - blocker.x, cp_y - blocker.y)
        blocker_speed = math.hypot(blocker.vx, blocker.vy)
        turns_blocker = dist_blocker / max(100.0, blocker_speed)
        
        # If we can reach the checkpoint before or shortly after the opponent, target it
        if turns_blocker < turns_opp + 2.0:
            return cp_idx
            
    # Fallback to the current opponent checkpoint if no other matches
    return curr_opp_cp


# ── Main Controller ───────────────────────────────────────────────────────────

def main():
    # Read Laps and Checkpoints
    laps = int(input())
    checkpoint_count = int(input())
    checkpoints = []
    for _ in range(checkpoint_count):
        cx, cy = map(int, input().split())
        checkpoints.append((cx, cy))

    # Initialize long-term persistent states
    our_states = [PodState(0), PodState(1)]
    opp_states = [PodState(2), PodState(3)]
    
    boosts_left = 1
    prev_runner_idx = None
    
    # Find the best checkpoint for boost (longest straight section)
    best_boost_cp_idx = 0
    max_straight_dist = 0.0
    for i in range(checkpoint_count):
        cp1 = checkpoints[i]
        cp2 = checkpoints[(i + 1) % checkpoint_count]
        d = math.hypot(cp2[0] - cp1[0], cp2[1] - cp1[1])
        if d > max_straight_dist:
            max_straight_dist = d
            best_boost_cp_idx = i

    while True:
        # Read Pod states
        # Our pods
        x0, y0, vx0, vy0, angle0, next_cp0 = map(int, input().split())
        x1, y1, vx1, vy1, angle1, next_cp1 = map(int, input().split())
        # Opponent pods
        ox0, oy0, ovx0, ovy0, oangle0, onext_cp0 = map(int, input().split())
        ox1, oy1, ovx1, ovy1, oangle1, onext_cp1 = map(int, input().split())

        # Update persistent state cooldowns
        for state in our_states + opp_states:
            if state.shield_cooldown > 0:
                state.shield_cooldown -= 1

        # Update checkpoints passed
        update_progress(our_states[0], next_cp0, checkpoint_count)
        update_progress(our_states[1], next_cp1, checkpoint_count)
        update_progress(opp_states[0], onext_cp0, checkpoint_count)
        update_progress(opp_states[1], onext_cp1, checkpoint_count)

        # Pack structured pod objects
        our_pods = [
            PodInfo(x0, y0, vx0, vy0, angle0, next_cp0, our_states[0]),
            PodInfo(x1, y1, vx1, vy1, angle1, next_cp1, our_states[1])
        ]
        opp_pods = [
            PodInfo(ox0, oy0, ovx0, ovy0, oangle0, onext_cp0, opp_states[0]),
            PodInfo(ox1, oy1, ovx1, ovy1, oangle1, onext_cp1, opp_states[1])
        ]

        # Calculate race progress scores
        score_our0 = get_progress_score(our_states[0], x0, y0, next_cp0, checkpoints)
        score_our1 = get_progress_score(our_states[1], x1, y1, next_cp1, checkpoints)
        score_opp0 = get_progress_score(opp_states[0], ox0, oy0, onext_cp0, checkpoints)
        score_opp1 = get_progress_score(opp_states[1], ox1, oy1, onext_cp1, checkpoints)

        # Assign Runner and Blocker roles with hysteresis
        ROLE_HYSTERESIS = 2000.0
        if prev_runner_idx is None:
            runner_idx = 0 if score_our0 >= score_our1 else 1
        else:
            runner_idx = prev_runner_idx
            blocker_idx = 1 - runner_idx
            scores = [score_our0, score_our1]
            if scores[blocker_idx] > scores[runner_idx] + ROLE_HYSTERESIS:
                runner_idx = blocker_idx
        prev_runner_idx = runner_idx
        blocker_idx = 1 - runner_idx

        # Identify opponent runner (the leader)
        opp_runner_idx = 0 if score_opp0 >= score_opp1 else 1
        opp_blocker_idx = 1 - opp_runner_idx

        runner = our_pods[runner_idx]
        blocker = our_pods[blocker_idx]
        opp_runner = opp_pods[opp_runner_idx]
        opp_blocker = opp_pods[opp_blocker_idx]

        commands = ["", ""]

        # ── RUNNER LOGIC ──────────────────────────────────────────────────────
        # Racing Line Corner Cutting
        curr_cp = checkpoints[runner.next_cp]
        next_cp_idx = (runner.next_cp + 1) % checkpoint_count
        next_cp = checkpoints[next_cp_idx]
        
        dx = next_cp[0] - curr_cp[0]
        dy = next_cp[1] - curr_cp[1]
        dist_between_cps = math.hypot(dx, dy)
        
        if dist_between_cps > 0:
            # Aim 450 units into the 600 radius of the checkpoint, offset towards the next one
            target_x = curr_cp[0] + (dx / dist_between_cps) * 450.0
            target_y = curr_cp[1] + (dy / dist_between_cps) * 450.0
        else:
            target_x, target_y = curr_cp[0], curr_cp[1]

        # Prevent target overshoot by correcting using target alignment
        next_vx = runner.vx * DRAG
        next_vy = runner.vy * DRAG
        target_x -= next_vx * 2.0
        target_y -= next_vy * 2.0

        # Tactical Bump-Turn: if an opponent is camped near our target checkpoint,
        # adjust our path to bump into them if the bounce will push us towards the next checkpoint,
        # using the collision to brake and realign ourselves.
        is_bump_turn = False
        dist_to_cp = math.hypot(curr_cp[0] - runner.x, curr_cp[1] - runner.y)
        if dist_to_cp < 2500:
            for opp in opp_pods:
                dist_opp_to_cp = math.hypot(opp.x - curr_cp[0], opp.y - curr_cp[1])
                if dist_opp_to_cp < 1000:
                    bounce_dx = runner.x - opp.x
                    bounce_dy = runner.y - opp.y
                    bounce_d = math.hypot(bounce_dx, bounce_dy)
                    
                    next_cp_dx = next_cp[0] - runner.x
                    next_cp_dy = next_cp[1] - runner.y
                    next_cp_d = math.hypot(next_cp_dx, next_cp_dy)
                    
                    if bounce_d > 0 and next_cp_d > 0:
                        dot = (bounce_dx * next_cp_dx + bounce_dy * next_cp_dy) / (bounce_d * next_cp_d)
                        if dot > 0.2:  # Favorable bump direction!
                            target_x = 0.5 * target_x + 0.5 * (opp.x + opp.vx)
                            target_y = 0.5 * target_y + 0.5 * (opp.y + opp.vy)
                            is_bump_turn = True
                            break

        # Calculate angles
        angle_to_target = math.degrees(math.atan2(target_y - runner.y, target_x - runner.x))
        angle_diff = (angle_to_target - runner.angle + 180) % 360 - 180
        angle_to_target_abs = abs(angle_diff)

        # Proportional thrust calculation (continuous, not binary)
        dist = math.hypot(curr_cp[0] - runner.x, curr_cp[1] - runner.y)

        if angle_to_target_abs > 90:
            thrust = 0
        else:
            # Scale thrust smoothly by angle: full at 0°, zero at 90°
            angle_factor = max(0.0, 1.0 - (angle_to_target_abs / 90.0))
            thrust = int(100 * angle_factor)

        # Proportional braking for corner entry
        braking_cp_idx = runner.next_cp
        braking_dist = dist
        
        next_cp_braking_idx = (braking_cp_idx + 1) % checkpoint_count
        prev_cp_braking_idx = (braking_cp_idx - 1) % checkpoint_count

        prev_cp_b = checkpoints[prev_cp_braking_idx]
        curr_cp_b = checkpoints[braking_cp_idx]
        next_cp_b = checkpoints[next_cp_braking_idx]

        turn_angle = turn_angle_between(prev_cp_b, curr_cp_b, next_cp_b)
        target_speed = 220.0 + 580.0 * (1.0 - turn_angle / 180.0)

        brake_dist = get_braking_distance(runner.speed, target_speed)
        if not is_bump_turn and braking_dist > 0:
            # Proportional braking: smoothly reduce thrust as we approach braking zone
            brake_ratio = braking_dist / (brake_dist + 200.0)
            if brake_ratio < 1.0:
                # We're inside the braking zone — scale thrust down proportionally
                brake_thrust = int(100 * max(0.0, brake_ratio))
                thrust = min(thrust, brake_thrust)

        # Boost Logic (widen search to any long straight segments and relax collision checks)
        if boosts_left > 0 and angle_to_target_abs < 15:
            prev_cp_idx = (runner.next_cp - 1) % checkpoint_count
            prev_cp = checkpoints[prev_cp_idx]
            segment_len = math.hypot(curr_cp[0] - prev_cp[0], curr_cp[1] - prev_cp[1])
            
            if (segment_len > 4000 or runner.next_cp == (best_boost_cp_idx + 1) % checkpoint_count) and dist > 3000:
                opp_directly_in_front = False
                for opp in opp_pods:
                    dist_opp = math.hypot(opp.x - runner.x, opp.y - runner.y)
                    if dist_opp < 1200:
                        opp_angle = math.degrees(math.atan2(opp.y - runner.y, opp.x - runner.x))
                        angle_to_opp = (opp_angle - runner.angle + 180) % 360 - 180
                        if abs(angle_to_opp) < 20:
                            opp_directly_in_front = True
                            break
                
                if not opp_directly_in_front:
                    thrust = "BOOST"
                    boosts_left -= 1

        # Runner Shielding
        desired_deg = math.degrees(math.atan2(target_y - runner.y, target_x - runner.x))
        diff = (desired_deg - runner.angle + 180) % 360 - 180
        rotate = max(-MAX_ROTATE, min(MAX_ROTATE, diff))
        new_heading = runner.angle + rotate
        rad = math.radians(new_heading)

        t_val = 650 if thrust == "BOOST" else (0 if thrust == "SHIELD" else thrust)
        next_runner_vx = runner.vx + math.cos(rad) * t_val
        next_runner_vy = runner.vy + math.sin(rad) * t_val

        for opp in opp_pods:
            collide, t_coll = will_collide_this_turn(runner.x, runner.y, next_runner_vx, next_runner_vy, opp.x, opp.y, opp.vx, opp.vy)
            if collide and runner.state.shield_cooldown == 0:
                rel_speed = math.hypot(next_runner_vx - opp.vx, next_runner_vy - opp.vy)
                if rel_speed > 120:
                    our_coll_x = runner.x + next_runner_vx * t_coll
                    our_coll_y = runner.y + next_runner_vy * t_coll
                    opp_coll_x = opp.x + opp.vx * t_coll
                    opp_coll_y = opp.y + opp.vy * t_coll
                    
                    bounce_dx = our_coll_x - opp_coll_x
                    bounce_dy = our_coll_y - opp_coll_y
                    bounce_d = math.hypot(bounce_dx, bounce_dy)
                    
                    # Determine which checkpoint the bounce should guide us towards
                    # If we are close to passing the current one, guide towards the next one
                    dist_to_curr_cp = math.hypot(checkpoints[runner.next_cp][0] - our_coll_x, checkpoints[runner.next_cp][1] - our_coll_y)
                    if dist_to_curr_cp < 700:
                        target_cp_idx = (runner.next_cp + 1) % checkpoint_count
                    else:
                        target_cp_idx = runner.next_cp
                        
                    cx_val, cy_val = checkpoints[target_cp_idx]
                    our_target_dx = cx_val - our_coll_x
                    our_target_dy = cy_val - our_coll_y
                    our_target_d = math.hypot(our_target_dx, our_target_dy)
                    
                    should_shield = True
                    if our_target_d > 0 and bounce_d > 0:
                        dot_product = (bounce_dx * our_target_dx + bounce_dy * our_target_dy) / (bounce_d * our_target_d)
                        if dot_product > 0.3:
                            should_shield = False
                    
                    if should_shield:
                        thrust = "SHIELD"
                        runner.state.shield_cooldown = 4
                        break

        commands[runner_idx] = f"{int(target_x)} {int(target_y)} {thrust}"

        # ── BLOCKER LOGIC ─────────────────────────────────────────────────────
        # Identify the checkpoint to block
        block_cp_idx = get_intercept_checkpoint(blocker, opp_runner, runner, checkpoints)
        block_cp = checkpoints[block_cp_idx]

        # Calculate blocking target
        opp_dx = block_cp[0] - opp_runner.x
        opp_dy = block_cp[1] - opp_runner.y
        opp_dist = math.hypot(opp_dx, opp_dy)

        if opp_dist > 0:
            # Stand 800 units in front of the checkpoint along the opponent's approach vector
            post_x = block_cp[0] - (opp_dx / opp_dist) * 800.0
            post_y = block_cp[1] - (opp_dy / opp_dist) * 800.0
        else:
            post_x, post_y = block_cp[0], block_cp[1]

        dist_to_post = math.hypot(post_x - blocker.x, post_y - blocker.y)

        # Move to position, or launch intercept
        if dist_to_post > 500:
            target_x = post_x
            target_y = post_y
            thrust = 100
        else:
            if opp_dist < 2500:
                # Intercept attack: Ram opponent runner
                target_x = opp_runner.x + opp_runner.vx * 1.5
                target_y = opp_runner.y + opp_runner.vy * 1.5
                thrust = 100
            else:
                # Maintain post: face opponent and slow down
                target_x = opp_runner.x
                target_y = opp_runner.y
                thrust = 0 if blocker.speed > 50 else 50

        # HIGH-PRIORITY PATH CLEARING FOR OUR RUNNER
        clearing_path_target = None

        # 1. Trajectory Threat: Opponent about to collide with our runner
        threat_opp = None
        t_threat = 999
        for opp in opp_pods:
            for t in range(1, 5):
                extrap_r_x = runner.x + runner.vx * t
                extrap_r_y = runner.y + runner.vy * t
                extrap_opp_x = opp.x + opp.vx * t
                extrap_opp_y = opp.y + opp.vy * t
                if math.hypot(extrap_r_x - extrap_opp_x, extrap_r_y - extrap_opp_y) < 950:
                    if t < t_threat:
                        t_threat = t
                        threat_opp = opp
                    break

        if threat_opp is not None:
            clearing_path_target = (
                threat_opp.x + threat_opp.vx * (t_threat - 0.5),
                threat_opp.y + threat_opp.vy * (t_threat - 0.5),
                100
            )

        # 2. Stationed Threat: Opponent camping near runner's checkpoint
        if clearing_path_target is None:
            runner_cp_x, runner_cp_y = checkpoints[runner.next_cp]
            dist_runner_to_cp = math.hypot(runner.x - runner_cp_x, runner.y - runner_cp_y)

            if dist_runner_to_cp < 4000:
                closest_camped_opp = None
                min_dist_to_cp = 99999.0
                for opp in opp_pods:
                    dist_opp_to_cp = math.hypot(opp.x - runner_cp_x, opp.y - runner_cp_y)
                    if dist_opp_to_cp < 1500:
                        if dist_opp_to_cp < min_dist_to_cp:
                            min_dist_to_cp = dist_opp_to_cp
                            closest_camped_opp = opp

                if closest_camped_opp is not None:
                    clearing_path_target = (
                        closest_camped_opp.x + closest_camped_opp.vx * 2.0,
                        closest_camped_opp.y + closest_camped_opp.vy * 2.0,
                        100
                    )

        # Apply path clearing if active, otherwise check approach corridor
        if clearing_path_target is not None:
            target_x, target_y, thrust = clearing_path_target
        else:
            runner_cp_x, runner_cp_y = checkpoints[runner.next_cp]
            dist_blocker_to_runner_cp = math.hypot(blocker.x - runner_cp_x, blocker.y - runner_cp_y)
            dist_runner_to_runner_cp = math.hypot(runner.x - runner_cp_x, runner.y - runner_cp_y)

            if dist_blocker_to_runner_cp < 2500 and dist_blocker_to_runner_cp < dist_runner_to_runner_cp:
                approach_dx = runner_cp_x - runner.x
                approach_dy = runner_cp_y - runner.y
                approach_d = math.hypot(approach_dx, approach_dy)
                if approach_d > 0:
                    perp_x = -approach_dy / approach_d
                    perp_y = approach_dx / approach_d
                    to_blocker_x = blocker.x - runner.x
                    to_blocker_y = blocker.y - runner.y
                    side = perp_x * to_blocker_x + perp_y * to_blocker_y
                    if side < 0:
                        perp_x, perp_y = -perp_x, -perp_y
                    target_x = blocker.x + perp_x * 2000.0
                    target_y = blocker.y + perp_y * 2000.0
                    thrust = 100

        # Avoid collision with our runner
        for t in [1, 2, 3]:
            extrap_blocker_x = blocker.x + blocker.vx * t
            extrap_blocker_y = blocker.y + blocker.vy * t
            extrap_runner_x = runner.x + runner.vx * t
            extrap_runner_y = runner.y + runner.vy * t
            if math.hypot(extrap_blocker_x - extrap_runner_x, extrap_blocker_y - extrap_runner_y) < 950:
                escape_x = extrap_blocker_x - extrap_runner_x
                escape_y = extrap_blocker_y - extrap_runner_y
                escape_d = math.hypot(escape_x, escape_y)
                if escape_d > 0:
                    # Steer hard perpendicular to runner's path
                    target_x = blocker.x + (escape_x / escape_d) * 2000.0
                    target_y = blocker.y + (escape_y / escape_d) * 2000.0
                    # Cut thrust if collision is predicted in 1 or 2 turns to decelerate and pivot faster
                    if t <= 2:
                        thrust = 0
                break

        # Blocker Shielding
        desired_deg = math.degrees(math.atan2(target_y - blocker.y, target_x - blocker.x))
        diff = (desired_deg - blocker.angle + 180) % 360 - 180
        rotate = max(-MAX_ROTATE, min(MAX_ROTATE, diff))
        new_heading = blocker.angle + rotate
        rad = math.radians(new_heading)

        t_val = 0 if thrust == "SHIELD" else thrust
        next_blocker_vx = blocker.vx + math.cos(rad) * t_val
        next_blocker_vy = blocker.vy + math.sin(rad) * t_val

        for opp in opp_pods:
            collide, t_coll = will_collide_this_turn(blocker.x, blocker.y, next_blocker_vx, next_blocker_vy, opp.x, opp.y, opp.vx, opp.vy)
            if collide and blocker.state.shield_cooldown == 0:
                rel_speed = math.hypot(next_blocker_vx - opp.vx, next_blocker_vy - opp.vy)
                if rel_speed > 120:
                    thrust = "SHIELD"
                    blocker.state.shield_cooldown = 4
                    break

        commands[blocker_idx] = f"{int(target_x)} {int(target_y)} {thrust}"

        # ── OUTPUT ACTIONS ────────────────────────────────────────────────────
        print(commands[0])
        print(commands[1])


if __name__ == "__main__":
    main()
