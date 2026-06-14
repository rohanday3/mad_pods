import math
import sys

# ── Configuration & Constants ──────────────────────────────────────────────────
CHECKPOINT_RADIUS = 600
DRAG = 0.85
MAX_ROTATE = 18.0
MAX_BOOSTS = 1  # Parameter for number of boosts per race

# ── State variables ────────────────────────────────────────────────────────────
boosts_left = MAX_BOOSTS
shield_cooldown = 0
best_boost_cp = None

prev_x, prev_y = None, None
prev_opp_x, prev_opp_y = None, None

checkpoint_order = []
checkpoint_set = set()
lap_complete = False

# ── Helper functions ───────────────────────────────────────────────────────────

def snap_checkpoint(cx, cy):
    """Register and snap checkpoint to a unique list."""
    for (ex, ey) in checkpoint_set:
        if math.hypot(cx - ex, cy - ey) < 50:
            return (ex, ey)
    checkpoint_set.add((cx, cy))
    checkpoint_order.append((cx, cy))
    return (cx, cy)


def get_next_checkpoint(current_cp, offset=1):
    """Get checkpoint at a relative offset from current, wrapping if lap complete."""
    if current_cp not in checkpoint_order:
        return None
    idx = checkpoint_order.index(current_cp)
    next_idx = idx + offset
    if next_idx < 0:
        if lap_complete:
            return checkpoint_order[next_idx % len(checkpoint_order)]
        else:
            return None
    if next_idx < len(checkpoint_order):
        return checkpoint_order[next_idx]
    if lap_complete:
        return checkpoint_order[next_idx % len(checkpoint_order)]
    return None


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

# ── Main loop ─────────────────────────────────────────────────────────────────

while True:
    # Read inputs
    try:
        inputs = input().split()
        if not inputs:
            break
        x, y, cx, cy, dist, angle = map(int, inputs)
        opp_x, opp_y = map(int, input().split())
    except Exception as e:
        break

    # 1. Track checkpoint sequence and lap completion
    current_cp = snap_checkpoint(cx, cy)
    if not lap_complete and len(checkpoint_order) >= 2:
        if current_cp == checkpoint_order[0]:
            lap_complete = True
            # Find the longest segment to use boost
            max_segment_dist = -1
            n = len(checkpoint_order)
            for i in range(n):
                cp1 = checkpoint_order[i]
                cp2 = checkpoint_order[(i + 1) % n]
                segment_dist = math.hypot(cp2[0] - cp1[0], cp2[1] - cp1[1])
                if segment_dist > max_segment_dist:
                    max_segment_dist = segment_dist
                    best_boost_cp = cp1

    # 2. Calculate velocities
    vx = x - prev_x if prev_x is not None else 0
    vy = y - prev_y if prev_y is not None else 0
    speed = math.hypot(vx, vy)
    prev_x, prev_y = x, y

    opp_vx = opp_x - prev_opp_x if prev_opp_x is not None else 0
    opp_vy = opp_y - prev_opp_y if prev_opp_y is not None else 0
    prev_opp_x, prev_opp_y = opp_x, opp_y

    # 3. Target calculation (Corner cutting & early switching)
    next_cp = get_next_checkpoint(current_cp, 1)
    
    if dist < CHECKPOINT_RADIUS and next_cp is not None:
        # We are inside the checkpoint circle: switch directly to next checkpoint
        tx, ty = next_cp
    elif next_cp is not None:
        # Corner cutting: aim slightly offset towards the next checkpoint
        nx, ny = next_cp
        cx_val, cy_val = current_cp
        dx = nx - cx_val
        dy = ny - cy_val
        d = math.hypot(dx, dy)
        if d > 0:
            ux, uy = dx / d, dy / d
        else:
            ux, uy = 0.0, 0.0
            
        # Linearly blend offset from 0 (at 2000 units away) to 530 (at 600 units away)
        if dist > 2000:
            offset = 0.0
        else:
            ratio = (2000.0 - dist) / (2000.0 - 600.0)
            offset = ratio * 530.0
            
        tx = cx_val + ux * offset
        ty = cy_val + uy * offset
    else:
        # First lap fallback
        tx, ty = cx, cy

    # 4. Drift compensation
    target_x = tx - vx * 3.0
    target_y = ty - vy * 3.0

    # 4b. Collision avoidance (slight steering adjustment)
    next_our_x = x + vx
    next_our_y = y + vy
    next_opp_x = opp_x + opp_vx
    next_opp_y = opp_y + opp_vy
    dist_next_opp = math.hypot(next_our_x - next_opp_x, next_our_y - next_opp_y)
    
    if dist_next_opp < 800:
        escape_x = next_our_x - next_opp_x
        escape_y = next_our_y - next_opp_y
        escape_d = math.hypot(escape_x, escape_y)
        if escape_d > 0:
            # Shift target slightly away from opponent, scaled by distance to checkpoint
            dodge_amt = 450.0 * min(1.0, dist / 1200.0)
            target_x += (escape_x / escape_d) * dodge_amt
            target_y += (escape_y / escape_d) * dodge_amt

    # 5. Angle calculations for steering and thrust
    # Extract absolute heading orientation of our pod from relative input angle
    target_angle_deg = math.degrees(math.atan2(cy - y, cx - x))
    pod_orientation = target_angle_deg - angle
    
    # Calculate relative angle to our compensated target point
    target_angle_new = math.degrees(math.atan2(target_y - y, target_x - x))
    angle_to_target = (target_angle_new - pod_orientation + 180) % 360 - 180
    angle_to_target_abs = abs(angle_to_target)

    # 6. Base thrust calculation
    if angle_to_target_abs > 90:
        thrust = 0
    elif angle_to_target_abs > 18:
        # Scale thrust down when target is not alignable in a single turn
        ratio = (90.0 - angle_to_target_abs) / (90.0 - 18.0)
        thrust = int(100 * ratio)
    else:
        thrust = 100

    # 7. Turn-based braking (Lookahead speed restriction)
    braking_cp = current_cp
    braking_dist = dist
    if dist < CHECKPOINT_RADIUS and next_cp is not None:
        braking_cp = next_cp
        braking_dist = math.hypot(braking_cp[0] - x, braking_cp[1] - y)
        
    next_cp_braking = get_next_checkpoint(braking_cp, 1)
    prev_cp_braking = get_next_checkpoint(braking_cp, -1)
    
    if next_cp_braking is not None and prev_cp_braking is not None:
        turn_angle = turn_angle_between(prev_cp_braking, braking_cp, next_cp_braking)
        # Target speed: 220 for U-turn (180 deg) up to 800 for straight (0 deg)
        target_speed = 220.0 + 580.0 * (1.0 - turn_angle / 180.0)
        
        brake_dist = get_braking_distance(speed, target_speed)
        if braking_dist < brake_dist + 150:
            thrust = min(thrust, 0)

    # 8. Boost logic
    if boosts_left > 0 and angle_to_target_abs < 5 and lap_complete:
        prev_cp = get_next_checkpoint(current_cp, -1)
        # Only boost on the longest straight segment, and when we have enough runway left
        if prev_cp == best_boost_cp and dist > 3000:
            # Avoid boosting if opponent is close and in front of us
            dist_opp = math.hypot(opp_x - x, opp_y - y)
            opp_angle = math.degrees(math.atan2(opp_y - y, opp_x - x))
            angle_to_opp = (opp_angle - pod_orientation + 180) % 360 - 180
            
            if dist_opp < 1500 and abs(angle_to_opp) < 35:
                pass  # Delay boost to avoid collision
            else:
                thrust = "BOOST"
                boosts_left -= 1

    # 9. Collision detection and Shield activation
    if shield_cooldown > 0:
        shield_cooldown -= 1
        
    # Recalculate predicted position next turn based on the actual target we steer towards
    desired_deg = math.degrees(math.atan2(target_y - y, target_x - x))
    diff = (desired_deg - pod_orientation + 180) % 360 - 180
    rotate = max(-MAX_ROTATE, min(MAX_ROTATE, diff))
    new_heading = pod_orientation + rotate
    rad = math.radians(new_heading)
    
    t_val = 650 if thrust == "BOOST" else thrust
    next_our_vx = vx + math.cos(rad) * t_val
    next_our_vy = vy + math.sin(rad) * t_val
    next_our_x = x + next_our_vx
    next_our_y = y + next_our_vy
    
    dist_next_opp = math.hypot(next_our_x - next_opp_x, next_our_y - next_opp_y)
    
    if dist_next_opp < 800 and shield_cooldown == 0:
        # Only shield if it's a significant collision
        if math.hypot(next_our_vx - opp_vx, next_our_vy - opp_vy) > 120:
            opp_target_dx = cx - next_opp_x
            opp_target_dy = cy - next_opp_y
            opp_target_d = math.hypot(opp_target_dx, opp_target_dy)
            
            bounce_dx = next_opp_x - next_our_x
            bounce_dy = next_opp_y - next_our_y
            bounce_d = math.hypot(bounce_dx, bounce_dy)
            
            should_shield = True
            if opp_target_d > 0 and bounce_d > 0:
                dot_product = (bounce_dx * opp_target_dx + bounce_dy * opp_target_dy) / (bounce_d * opp_target_d)
                # If dot product is > 0.3, the bounce would push them generally towards their target.
                # In that case, we avoid shielding so we don't boost them.
                if dot_product > 0.3:
                    should_shield = False
            
            if should_shield:
                thrust = "SHIELD"
                shield_cooldown = 4  # 1 active turn + 3 cooldown turns

    # 10. Output command
    print(f"{int(target_x)} {int(target_y)} {thrust}")
