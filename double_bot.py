import math
import sys

# ── Configuration & Constants ──────────────────────────────────────────────────
CHECKPOINT_RADIUS = 600
DRAG = 0.85
MAX_ROTATE = 18.0

# ── Classes & Helper Functions ──────────────────────────────────────────────────

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
    dx = x1 - x2
    dy = y1 - y2
    dvx = vx1 - vx2
    dvy = vy1 - vy2
    
    # If they are already colliding
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
    Select the optimal checkpoint (C, C+1, or C+2) to block the opponent.
    NEVER pick a checkpoint that our runner is heading towards.
    Returns the index of the selected checkpoint.
    """
    num_cp = len(checkpoints)
    curr_opp_cp = target_opp.next_cp
    
    # Checkpoints our runner needs soon — never block these
    runner_cps = set()
    runner_cps.add(runner.next_cp)
    runner_cps.add((runner.next_cp + 1) % num_cp)
    
    for offset in range(3):
        cp_idx = (curr_opp_cp + offset) % num_cp
        
        # Skip checkpoints our runner needs
        if cp_idx in runner_cps:
            continue
            
        cp_x, cp_y = checkpoints[cp_idx]
        
        # Estimate turns for opponent to reach cp_idx along the race track
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
        
        # If blocker can arrive at least 1 turn before opponent, target this checkpoint
        if turns_blocker + 1.0 < turns_opp:
            return cp_idx
            
    # Fallback: target the first opponent checkpoint that isn't our runner's
    for offset in range(1, num_cp):
        cp_idx = (curr_opp_cp + offset) % num_cp
        if cp_idx not in runner_cps:
            return cp_idx
    # Last resort
    return (curr_opp_cp + 1) % num_cp


# ── Initialization ─────────────────────────────────────────────────────────────

try:
    laps = int(input())
    checkpoint_count = int(input())
    checkpoints = []
    for _ in range(checkpoint_count):
        cx, cy = map(int, input().split())
        checkpoints.append((cx, cy))
except Exception as e:
    sys.exit(0)

# Calculate segment lengths to find the longest segment for boost
segment_lengths = []
for i in range(checkpoint_count):
    cp1 = checkpoints[i]
    cp2 = checkpoints[(i + 1) % checkpoint_count]
    segment_lengths.append(math.hypot(cp2[0] - cp1[0], cp2[1] - cp1[1]))
best_boost_cp_idx = segment_lengths.index(max(segment_lengths))

# Initialize persistent states
our_states = [PodState(0), PodState(1)]
opp_states = [PodState(2), PodState(3)]
boosts_left = 1
prev_runner_idx = None  # For role hysteresis

# ── Main Turn Loop ─────────────────────────────────────────────────────────────

while True:
    try:
        # Read our pods
        x0, y0, vx0, vy0, angle0, next_cp0 = map(int, input().split())
        x1, y1, vx1, vy1, angle1, next_cp1 = map(int, input().split())
        # Read opponent pods
        ox0, oy0, ovx0, ovy0, oangle0, onext_cp0 = map(int, input().split())
        ox1, oy1, ovx1, ovy1, oangle1, onext_cp1 = map(int, input().split())
    except Exception as e:
        break

    # Decrement shield cooldowns
    for state in our_states + opp_states:
        if state.shield_cooldown > 0:
            state.shield_cooldown -= 1

    # Update progress metrics
    update_progress(our_states[0], next_cp0, checkpoint_count)
    update_progress(our_states[1], next_cp1, checkpoint_count)
    update_progress(opp_states[0], onext_cp0, checkpoint_count)
    update_progress(opp_states[1], onext_cp1, checkpoint_count)

    # Pack PodInfo
    our_pods = [
        PodInfo(x0, y0, vx0, vy0, angle0, next_cp0, our_states[0]),
        PodInfo(x1, y1, vx1, vy1, angle1, next_cp1, our_states[1])
    ]
    opp_pods = [
        PodInfo(ox0, oy0, ovx0, ovy0, oangle0, onext_cp0, opp_states[0]),
        PodInfo(ox1, oy1, ovx1, ovy1, oangle1, onext_cp1, opp_states[1])
    ]

    # Compute progress scores
    score_our0 = get_progress_score(our_states[0], x0, y0, next_cp0, checkpoints)
    score_our1 = get_progress_score(our_states[1], x1, y1, next_cp1, checkpoints)
    score_opp0 = get_progress_score(opp_states[0], ox0, oy0, onext_cp0, checkpoints)
    score_opp1 = get_progress_score(opp_states[1], ox1, oy1, onext_cp1, checkpoints)

    # Roles allocation with hysteresis to prevent oscillation
    ROLE_HYSTERESIS = 2000.0
    if prev_runner_idx is None:
        # First turn: pick based on raw score
        if score_our0 >= score_our1:
            runner_idx, blocker_idx = 0, 1
        else:
            runner_idx, blocker_idx = 1, 0
    else:
        # Subsequent turns: only swap if the other pod is ahead by a significant margin
        runner_idx = prev_runner_idx
        blocker_idx = 1 - runner_idx
        scores = [score_our0, score_our1]
        if scores[blocker_idx] > scores[runner_idx] + ROLE_HYSTERESIS:
            runner_idx, blocker_idx = blocker_idx, runner_idx
    prev_runner_idx = runner_idx

    if score_opp0 >= score_opp1:
        opp_runner_idx, opp_blocker_idx = 0, 1
    else:
        opp_runner_idx, opp_blocker_idx = 1, 0

    runner = our_pods[runner_idx]
    blocker = our_pods[blocker_idx]
    opp_runner = opp_pods[opp_runner_idx]
    opp_blocker = opp_pods[opp_blocker_idx]

    commands = [None, None]

    # ── RUNNER LOGIC ──────────────────────────────────────────────────────────
    cx, cy = checkpoints[runner.next_cp]
    dist = math.hypot(cx - runner.x, cy - runner.y)
    
    # Corner cutting
    next_cp_idx = (runner.next_cp + 1) % checkpoint_count
    nx, ny = checkpoints[next_cp_idx]
    dx = nx - cx
    dy = ny - cy
    d_seg = math.hypot(dx, dy)
    if d_seg > 0:
        ux, uy = dx / d_seg, dy / d_seg
    else:
        ux, uy = 0.0, 0.0
        
    if dist < CHECKPOINT_RADIUS:
        tx, ty = nx, ny
    else:
        if dist > 2000:
            offset = 0.0
        else:
            ratio = (2000.0 - dist) / (2000.0 - CHECKPOINT_RADIUS)
            offset = ratio * 530.0
        tx = cx + ux * offset
        ty = cy + uy * offset

    # Drift compensation
    target_x = tx - runner.vx * 3.0
    target_y = ty - runner.vy * 3.0

    # Collision prediction & Dodging
    t_collision = None
    for other_pod in opp_pods + [blocker]:
        for t in [1, 2, 3]:
            extrap_our_x = runner.x + runner.vx * t
            extrap_our_y = runner.y + runner.vy * t
            extrap_other_x = other_pod.x + other_pod.vx * t
            extrap_other_y = other_pod.y + other_pod.vy * t
            
            d_t = math.hypot(extrap_our_x - extrap_other_x, extrap_our_y - extrap_other_y)
            if d_t < 950:
                t_collision = (t, other_pod)
                break
        if t_collision is not None:
            break

    if t_collision is not None:
        t, other_pod = t_collision
        extrap_our_x = runner.x + runner.vx * t
        extrap_our_y = runner.y + runner.vy * t
        other_x_t = other_pod.x + other_pod.vx * t
        other_y_t = other_pod.y + other_pod.vy * t
        
        is_own_blocker = (other_pod is blocker)
        # Always dodge our own blocker (up to t=3); dodge opponents only at t<=2
        should_dodge = is_own_blocker or (t <= 2)
        
        if should_dodge:
            escape_x = extrap_our_x - other_x_t
            escape_y = extrap_our_y - other_y_t
            escape_d = math.hypot(escape_x, escape_y)
            if escape_d > 0:
                # Dodge our own blocker more aggressively
                if is_own_blocker:
                    dodge_amt = 800.0
                else:
                    dodge_amt = 550.0 * min(1.0, dist / 1200.0)
                target_x += (escape_x / escape_d) * dodge_amt
                target_y += (escape_y / escape_d) * dodge_amt

    # Angle calculations for steering and thrust
    target_angle_new = math.degrees(math.atan2(target_y - runner.y, target_x - runner.x))
    angle_to_target = (target_angle_new - runner.angle + 180) % 360 - 180
    angle_to_target_abs = abs(angle_to_target)

    # Base thrust
    if angle_to_target_abs > 90:
        thrust = 0
    elif angle_to_target_abs > 18:
        ratio = (90.0 - angle_to_target_abs) / (90.0 - 18.0)
        thrust = int(100 * ratio)
    else:
        thrust = 100

    # Turn-based braking
    braking_cp_idx = runner.next_cp
    braking_dist = dist
    if dist < CHECKPOINT_RADIUS:
        braking_cp_idx = (runner.next_cp + 1) % checkpoint_count
        bcp = checkpoints[braking_cp_idx]
        braking_dist = math.hypot(bcp[0] - runner.x, bcp[1] - runner.y)

    next_cp_braking_idx = (braking_cp_idx + 1) % checkpoint_count
    prev_cp_braking_idx = (braking_cp_idx - 1) % checkpoint_count

    prev_cp_b = checkpoints[prev_cp_braking_idx]
    curr_cp_b = checkpoints[braking_cp_idx]
    next_cp_b = checkpoints[next_cp_braking_idx]

    turn_angle = turn_angle_between(prev_cp_b, curr_cp_b, next_cp_b)
    target_speed = 220.0 + 580.0 * (1.0 - turn_angle / 180.0)

    brake_dist = get_braking_distance(runner.speed, target_speed)
    if braking_dist < brake_dist + 150:
        thrust = min(thrust, 0)

    # Boost logic
    if boosts_left > 0 and angle_to_target_abs < 5:
        if runner.next_cp == (best_boost_cp_idx + 1) % checkpoint_count and dist > 3000:
            opp_too_close = False
            for opp in opp_pods:
                dist_opp = math.hypot(opp.x - runner.x, opp.y - runner.y)
                opp_angle = math.degrees(math.atan2(opp.y - runner.y, opp.x - runner.x))
                angle_to_opp = (opp_angle - runner.angle + 180) % 360 - 180
                if dist_opp < 1500 and abs(angle_to_opp) < 35:
                    opp_too_close = True
                    break
            
            if not opp_too_close:
                thrust = "BOOST"
                boosts_left -= 1

    # Shield logic for runner (using exact continuous-time collision checking)
    desired_deg = math.degrees(math.atan2(target_y - runner.y, target_x - runner.x))
    diff = (desired_deg - runner.angle + 180) % 360 - 180
    rotate = max(-MAX_ROTATE, min(MAX_ROTATE, diff))
    new_heading = runner.angle + rotate
    rad = math.radians(new_heading)

    t_val = 650 if thrust == "BOOST" else (0 if thrust == "SHIELD" else thrust)
    next_our_vx = runner.vx + math.cos(rad) * t_val
    next_our_vy = runner.vy + math.sin(rad) * t_val

    for opp in opp_pods:
        collide, t_coll = will_collide_this_turn(runner.x, runner.y, next_our_vx, next_our_vy, opp.x, opp.y, opp.vx, opp.vy)
        if collide and runner.state.shield_cooldown == 0:
            rel_speed = math.hypot(next_our_vx - opp.vx, next_our_vy - opp.vy)
            if rel_speed > 120:
                # Calculate coordinates at precise moment of collision
                our_coll_x = runner.x + next_our_vx * t_coll
                our_coll_y = runner.y + next_our_vy * t_coll
                opp_coll_x = opp.x + opp.vx * t_coll
                opp_coll_y = opp.y + opp.vy * t_coll
                
                # Check target vector vs bounce direction vector
                bounce_dx = our_coll_x - opp_coll_x
                bounce_dy = our_coll_y - opp_coll_y
                bounce_d = math.hypot(bounce_dx, bounce_dy)
                
                cx_val, cy_val = checkpoints[runner.next_cp]
                our_target_dx = cx_val - our_coll_x
                our_target_dy = cy_val - our_coll_y
                our_target_d = math.hypot(our_target_dx, our_target_dy)
                
                should_shield = True
                if our_target_d > 0 and bounce_d > 0:
                    dot_product = (bounce_dx * our_target_dx + bounce_dy * our_target_dy) / (bounce_d * our_target_d)
                    if dot_product > 0.3:
                        # Collision actually pushes us toward the checkpoint, don't shield!
                        should_shield = False
                
                if should_shield:
                    thrust = "SHIELD"
                    runner.state.shield_cooldown = 4
                    break

    commands[runner_idx] = f"{int(target_x)} {int(target_y)} {thrust}"

    # ── BLOCKER LOGIC ──────────────────────────────────────────────────────────
    block_cp_idx = get_intercept_checkpoint(blocker, opp_runner, runner, checkpoints)
    bcp_x, bcp_y = checkpoints[block_cp_idx]
    
    ox, oy = opp_runner.x, opp_runner.y
    dx = ox - bcp_x
    dy = oy - bcp_y
    d_opp_to_cp = math.hypot(dx, dy)
    if d_opp_to_cp > 0:
        ux, uy = dx / d_opp_to_cp, dy / d_opp_to_cp
    else:
        ux, uy = 1.0, 0.0
        
    # Position outside the checkpoint circle (radius 600) + 500 buffer
    block_x = bcp_x + ux * 1100
    block_y = bcp_y + uy * 1100
    
    dist_to_block_pos = math.hypot(block_x - blocker.x, block_y - blocker.y)
    
    if dist_to_block_pos > 800:
        # Move to block position with drift compensation
        target_x = block_x - blocker.vx * 3.0
        target_y = block_y - blocker.vy * 3.0
        
        target_angle_new = math.degrees(math.atan2(target_y - blocker.y, target_x - blocker.x))
        angle_to_target = (target_angle_new - blocker.angle + 180) % 360 - 180
        angle_to_target_abs = abs(angle_to_target)
        
        if angle_to_target_abs > 90:
            thrust = 0
        elif angle_to_target_abs > 18:
            ratio = (90.0 - angle_to_target_abs) / (90.0 - 18.0)
            thrust = int(100 * ratio)
        else:
            thrust = 100
    else:
        # Positioned correctly: face and intercept opponent runner
        dist_to_opp = math.hypot(ox - blocker.x, oy - blocker.y)
        if dist_to_opp < 1200:
            # Lead target the opponent runner based on their velocity
            target_x = ox + opp_runner.vx * 2.0
            target_y = oy + opp_runner.vy * 2.0
            thrust = 100
        else:
            target_x = ox
            target_y = oy
            thrust = 0

    # High-Priority Blocker Interventions to Clear the Runner's Path
    clearing_path_target = None
    
    # 1. Trajectory Threat Interception: ram any opponent predicted to collide with our runner
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
        
    # 2. Stationed Threat Interception: ram any opponent camping near the runner's next checkpoint
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

    # Apply path clearing target if active, otherwise run the approach corridor check
    if clearing_path_target is not None:
        target_x, target_y, thrust = clearing_path_target
    else:
        # Check if blocker is near the runner's approach corridor to its next checkpoint
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
    
    # Also dodge direct collision with runner (velocity-based prediction)
    for t in [1, 2, 3]:
        extrap_blocker_x = blocker.x + blocker.vx * t
        extrap_blocker_y = blocker.y + blocker.vy * t
        extrap_runner_x = runner.x + runner.vx * t
        extrap_runner_y = runner.y + runner.vy * t
        d_to_runner = math.hypot(extrap_blocker_x - extrap_runner_x, extrap_blocker_y - extrap_runner_y)
        if d_to_runner < 950:
            escape_x = extrap_blocker_x - extrap_runner_x
            escape_y = extrap_blocker_y - extrap_runner_y
            escape_d = math.hypot(escape_x, escape_y)
            if escape_d > 0:
                target_x += (escape_x / escape_d) * 800.0
                target_y += (escape_y / escape_d) * 800.0
            break

    # Shield logic for blocker (using exact continuous-time collision checking)
    desired_deg = math.degrees(math.atan2(target_y - blocker.y, target_x - blocker.x))
    diff = (desired_deg - blocker.angle + 180) % 360 - 180
    rotate = max(-MAX_ROTATE, min(MAX_ROTATE, diff))
    new_heading = blocker.angle + rotate
    rad = math.radians(new_heading)

    t_val = 0 if thrust == "SHIELD" else thrust
    next_our_vx = blocker.vx + math.cos(rad) * t_val
    next_our_vy = blocker.vy + math.sin(rad) * t_val

    for opp in opp_pods:
        collide, t_coll = will_collide_this_turn(blocker.x, blocker.y, next_our_vx, next_our_vy, opp.x, opp.y, opp.vx, opp.vy)
        if collide and blocker.state.shield_cooldown == 0:
            rel_speed = math.hypot(next_our_vx - opp.vx, next_our_vy - opp.vy)
            if rel_speed > 120:
                thrust = "SHIELD"
                blocker.state.shield_cooldown = 4
                break

    commands[blocker_idx] = f"{int(target_x)} {int(target_y)} {thrust}"

    # ── OUTPUT ACTIONS ─────────────────────────────────────────────────────────
    print(commands[0])
    print(commands[1])
