import math
import sys
import random
import time

# ── Constants ─────────────────────────────────────────────────────────────────
CHECKPOINT_RADIUS   = 600
POD_RADIUS          = 400.0
DRAG                = 0.85
MAX_ROTATE          = 18.0
MAX_THRUST          = 100
BOOST_THRUST        = 650
MIN_IMPULSE         = 120.0
SHIELD_MASS_MULT    = 10.0
BASE_MASS           = 1.0
MAP_DIAGONAL        = math.hypot(16000, 9000)

# GA hyper-parameters
SIM_TURNS           = 6      # horizon per solution
POPULATION          = 10     # solutions kept alive
TIME_LIMIT_MS       = 70     # leave margin for I/O

# Fitness weights
K_AHEAD             = 2.0    # weight for being ahead of opponent
PROGRESS_C          = MAP_DIAGONAL * 1.5  # checkpoint-pass bonus > max dist


# ── Physics helpers ───────────────────────────────────────────────────────────

def _norm_angle(deg):
    """Normalise degrees to (-180, 180]."""
    return (deg + 180.0) % 360.0 - 180.0


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


# ── Pod simulation state ───────────────────────────────────────────────────────

class SimPod:
    """Mutable pod state used inside the simulator."""
    __slots__ = ('x','y','vx','vy','angle','next_cp','cps_done',
                 'shield_cd','mass','boosted')

    def __init__(self, x, y, vx, vy, angle, next_cp, cps_done,
                 shield_cd=0, boosted=False):
        self.x        = float(x)
        self.y        = float(y)
        self.vx       = float(vx)
        self.vy       = float(vy)
        self.angle    = float(angle)
        self.next_cp  = next_cp
        self.cps_done = cps_done
        self.shield_cd = shield_cd
        self.mass     = SHIELD_MASS_MULT if shield_cd > 0 else BASE_MASS
        self.boosted  = boosted

    def copy(self):
        p = SimPod.__new__(SimPod)
        p.x        = self.x
        p.y        = self.y
        p.vx       = self.vx
        p.vy       = self.vy
        p.angle    = self.angle
        p.next_cp  = self.next_cp
        p.cps_done = self.cps_done
        p.shield_cd = self.shield_cd
        p.mass     = self.mass
        p.boosted  = self.boosted
        return p


# ── Move encoding ──────────────────────────────────────────────────────────────

class Move:
    """One pod's action for one turn."""
    __slots__ = ('rot','thrust','shield','boost')

    def __init__(self, rot=0.0, thrust=100, shield=False, boost=False):
        self.rot    = float(_clamp(rot, -MAX_ROTATE, MAX_ROTATE))
        self.thrust = int(_clamp(thrust, 0, MAX_THRUST))
        self.shield = shield
        self.boost  = boost

    def copy(self):
        return Move(self.rot, self.thrust, self.shield, self.boost)


def random_move():
    """Biased random move matching the guide's mutation distribution."""
    rot_choices    = [-MAX_ROTATE, -MAX_ROTATE, 0.0, 0.0, 0.0, MAX_ROTATE, MAX_ROTATE,
                      random.uniform(-MAX_ROTATE, MAX_ROTATE)]
    thrust_choices = [0, 0, MAX_THRUST, MAX_THRUST, MAX_THRUST, MAX_THRUST,
                      random.randint(0, MAX_THRUST)]
    return Move(
        rot    = random.choice(rot_choices),
        thrust = random.choice(thrust_choices),
        shield = False,
        boost  = False,
    )


def mutate_move(m):
    """Return a copy of m with one field randomised."""
    nm = m.copy()
    # rotation and thrust get double weight
    field = random.choices(
        ['rot','rot','thrust','thrust','shield','boost'],
        k=1
    )[0]
    if field == 'rot':
        nm.rot = float(random.choice([
            -MAX_ROTATE, -MAX_ROTATE, 0.0, 0.0, 0.0, MAX_ROTATE, MAX_ROTATE,
            random.uniform(-MAX_ROTATE, MAX_ROTATE)
        ]))
    elif field == 'thrust':
        nm.thrust = int(random.choice([
            0, 0, MAX_THRUST, MAX_THRUST, MAX_THRUST, random.randint(0, MAX_THRUST)
        ]))
    elif field == 'shield':
        nm.shield = not nm.shield
    else:
        nm.boost = not nm.boost
    return nm


# ── Solution: T moves for each of our 2 pods ──────────────────────────────────

class Solution:
    __slots__ = ('moves',)  # moves[pod_idx][turn_idx] -> Move

    def __init__(self, t=SIM_TURNS):
        self.moves = [[random_move() for _ in range(t)] for _ in range(2)]

    def copy(self):
        s = Solution.__new__(Solution)
        s.moves = [[m.copy() for m in pod] for pod in self.moves]
        return s

    def shift(self):
        """Drop the first turn; append a random move at the end."""
        for pod in self.moves:
            pod.pop(0)
            pod.append(random_move())

    def mutate(self):
        """Return a mutated copy: one random (pod, turn) cell changed."""
        s = self.copy()
        pi = random.randrange(2)
        ti = random.randrange(len(s.moves[pi]))
        s.moves[pi][ti] = mutate_move(s.moves[pi][ti])
        return s


# ── Physics simulation ─────────────────────────────────────────────────────────

def _apply_move(pod, move, is_first_turn, boosts_remaining):
    """
    Apply rotation + acceleration according to the rules.
    Returns updated boosts_remaining.
    """
    # --- Rotation (capped at MAX_ROTATE, except first turn) ---
    if not is_first_turn:
        target_angle = pod.angle + _clamp(move.rot, -MAX_ROTATE, MAX_ROTATE)
    else:
        target_angle = pod.angle + move.rot  # no cap on turn 0

    pod.angle = target_angle % 360.0

    # --- Determine effective thrust ---
    if move.shield:
        pod.shield_cd = 4
        pod.mass = SHIELD_MASS_MULT
        return boosts_remaining  # shield = no thrust this turn

    pod.mass = BASE_MASS
    if pod.shield_cd > 0:
        pod.shield_cd -= 1
        pod.mass = SHIELD_MASS_MULT if pod.shield_cd > 0 else BASE_MASS

    if move.boost and boosts_remaining > 0 and not pod.boosted:
        thrust = BOOST_THRUST
        boosts_remaining -= 1
        pod.boosted = True
    else:
        thrust = move.thrust

    rad = math.radians(pod.angle)
    pod.vx += math.cos(rad) * thrust
    pod.vy += math.sin(rad) * thrust

    return boosts_remaining


def _detect_collision_time(p1, p2):
    """
    Solve for earliest t in (0,1] at which |p2(t)-p1(t)| = 2*POD_RADIUS.
    p_i(t) = p_i.pos + t * p_i.vel  (relative, no friction yet — handled per-segment)
    Returns t, or None if no collision.
    """
    dx  = p2.x  - p1.x
    dy  = p2.y  - p1.y
    dvx = p2.vx - p1.vx
    dvy = p2.vy - p1.vy

    a = dvx*dvx + dvy*dvy
    if a < 1e-9:
        return None  # parallel trajectories

    b = 2.0 * (dx*dvx + dy*dvy)
    c = dx*dx + dy*dy - (2.0*POD_RADIUS)**2

    disc = b*b - 4.0*a*c
    if disc < 0.0:
        return None

    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)

    t = t1 if t1 > 1e-9 else t2
    if t <= 1e-9 or t > 1.0:
        return None
    return t


def _apply_rebound(p1, p2):
    """
    Elastic collision between p1 and p2 (already positioned at collision point).
    Applies minimum impulse rule.
    """
    ux = p2.x - p1.x
    uy = p2.y - p1.y
    dist = math.sqrt(ux*ux + uy*uy)
    if dist < 1e-9:
        return
    ux /= dist
    uy /= dist

    m1, m2 = p1.mass, p2.mass
    m = (m1 * m2) / (m1 + m2)
    k = (p2.vx - p1.vx)*ux + (p2.vy - p1.vy)*uy

    # impulse (negative because we want to push them apart)
    impulse = -2.0 * m * k

    # enforce minimum impulse magnitude
    if abs(impulse) < MIN_IMPULSE:
        impulse = MIN_IMPULSE if impulse >= 0 else -MIN_IMPULSE

    p1.vx += (-1.0/m1) * impulse * ux
    p1.vy += (-1.0/m1) * impulse * uy
    p2.vx += ( 1.0/m2) * impulse * ux
    p2.vy += ( 1.0/m2) * impulse * uy


def _simulate_movement(pods, checkpoints):
    """
    Advance all pods through one turn's movement phase with exact collision detection.
    Mutates pods in-place.  Checkpoints are checked for progress.
    """
    t_remaining = 1.0

    while t_remaining > 1e-9:
        # Find earliest collision among all pairs
        earliest_t = t_remaining
        colliding  = None

        n = len(pods)
        for i in range(n):
            for j in range(i+1, n):
                ct = _detect_collision_time(pods[i], pods[j])
                if ct is not None and ct <= earliest_t:
                    earliest_t  = ct
                    colliding   = (i, j)

        # Move all pods to earliest_t
        for p in pods:
            p.x += p.vx * earliest_t
            p.y += p.vy * earliest_t

        if colliding is not None:
            _apply_rebound(pods[colliding[0]], pods[colliding[1]])

        t_remaining -= earliest_t

    # Friction + truncation
    for p in pods:
        p.vx = int(p.vx * DRAG)
        p.vy = int(p.vy * DRAG)
        p.x  = round(p.x)
        p.y  = round(p.y)

    # Checkpoint progress
    for p in pods:
        cp_x, cp_y = checkpoints[p.next_cp]
        if math.hypot(p.x - cp_x, p.y - cp_y) <= CHECKPOINT_RADIUS:
            p.cps_done += 1
            p.next_cp   = (p.next_cp + 1) % len(checkpoints)


def simulate_solution(our_pods, opp_pods, solution, checkpoints, boosts_left,
                      our_moves_fixed=None):
    """
    Simulate SIM_TURNS turns for our solution against static opponent pods
    (opponent moves are not optimised — we hold them constant at straight-ahead thrust).
    Returns the tuple of final (our_pods, opp_pods) as SimPod lists.
    """
    # Deep-copy so we don't mutate the originals
    my  = [p.copy() for p in our_pods]
    opp = [p.copy() for p in opp_pods]
    bl  = boosts_left

    for t in range(SIM_TURNS):
        # Apply our moves
        for pi in range(2):
            move = solution.moves[pi][t]
            bl = _apply_move(my[pi], move, False, bl)

        # Apply simple straight-ahead for opponents (no model of their intent)
        for op in opp:
            cp_x, cp_y = checkpoints[op.next_cp]
            dx = cp_x - op.x
            dy = cp_y - op.y
            dist = math.sqrt(dx*dx + dy*dy) or 1.0
            target_angle = math.degrees(math.atan2(dy, dx)) % 360.0
            rot = _clamp(_norm_angle(target_angle - op.angle), -MAX_ROTATE, MAX_ROTATE)
            op.angle = (op.angle + rot) % 360.0
            rad = math.radians(op.angle)
            op.vx += math.cos(rad) * MAX_THRUST
            op.vy += math.sin(rad) * MAX_THRUST

        _simulate_movement(my + opp, checkpoints)

    return my, opp


# ── Fitness function ───────────────────────────────────────────────────────────

def _pod_score(pod, checkpoints):
    cp_x, cp_y = checkpoints[pod.next_cp]
    dist = math.hypot(pod.x - cp_x, pod.y - cp_y)
    return pod.cps_done * PROGRESS_C - dist


def fitness(my, opp, checkpoints):
    """
    Score a simulated end-state.
    Higher = better for us.
    """
    # Identify our racer (higher progress)
    s0 = _pod_score(my[0], checkpoints)
    s1 = _pod_score(my[1], checkpoints)
    my_racer_score   = max(s0, s1)
    my_blocker       = my[1] if s0 >= s1 else my[0]

    # Identify opponent racer
    os0 = _pod_score(opp[0], checkpoints)
    os1 = _pod_score(opp[1], checkpoints)
    opp_racer_score  = max(os0, os1)
    opp_racer        = opp[0] if os0 >= os1 else opp[1]

    ahead_score = my_racer_score - opp_racer_score

    # Blocker should be near opponent racer's next checkpoint
    opp_next_cp = checkpoints[opp_racer.next_cp]
    blocker_dist = math.hypot(my_blocker.x - opp_next_cp[0],
                              my_blocker.y - opp_next_cp[1])

    return K_AHEAD * ahead_score - blocker_dist


# ── Genetic algorithm ──────────────────────────────────────────────────────────

def run_ga(our_pods, opp_pods, checkpoints, boosts_left,
           warm_solutions, time_limit_ms):
    """
    Run the genetic algorithm within the time budget.
    warm_solutions: previous turn's best N solutions (already shifted by 1).
    Returns (best_solution, population_for_next_turn).
    """
    start = time.perf_counter()
    limit = time_limit_ms / 1000.0

    # --- Initialise population ---
    if warm_solutions:
        population = warm_solutions[:POPULATION]
        # pad with randoms if needed
        while len(population) < POPULATION:
            population.append(Solution())
    else:
        population = [Solution() for _ in range(POPULATION)]

    # Score initial population
    def score(sol):
        my, opp = simulate_solution(our_pods, opp_pods, sol, checkpoints, boosts_left)
        return fitness(my, opp, checkpoints)

    scored = [(score(s), s) for s in population]
    scored.sort(key=lambda x: x[0], reverse=True)

    generations = 0
    while (time.perf_counter() - start) < limit:
        # Duplicate + mutate
        children = [(sc, s.mutate()) for sc, s in scored]
        # Score children
        children_scored = [(score(s), s) for _, s in children]
        # Merge and keep best N
        combined = scored + children_scored
        combined.sort(key=lambda x: x[0], reverse=True)
        scored = combined[:POPULATION]
        generations += 1

    best_solution = scored[0][1]
    # Prepare warm start: shift every survivor by 1 turn
    next_population = [s.shift() or s for _, s in scored]
    # shift() mutates in place and returns None, so:
    next_population = []
    for _, s in scored:
        sc = s.copy()
        sc.shift()
        next_population.append(sc)

    return best_solution, next_population


# ── Progress tracking ──────────────────────────────────────────────────────────

class PodState:
    def __init__(self):
        self.cps_done   = 0
        self.prev_next  = -1
        self.shield_cd  = 0
        self.boosted    = False

    def update(self, next_cp, checkpoint_count):
        if self.prev_next == -1:
            self.prev_next = next_cp
        elif next_cp != self.prev_next:
            diff = (next_cp - self.prev_next) % checkpoint_count
            self.cps_done += diff
            self.prev_next = next_cp


# ── Output helpers ─────────────────────────────────────────────────────────────

def _move_to_output(pod, move, checkpoints, boosts_left):
    """
    Convert a Move to the (target_x, target_y, action_str) the game expects.
    The target is derived from the pod's angle after applying rotation.
    """
    new_angle = (pod.angle + _clamp(move.rot, -MAX_ROTATE, MAX_ROTATE)) % 360.0
    rad = math.radians(new_angle)
    tx = int(round(pod.x + math.cos(rad) * 10000))
    ty = int(round(pod.y + math.sin(rad) * 10000))

    if move.shield:
        action = "SHIELD"
    elif move.boost and boosts_left > 0 and not pod.boosted:
        action = "BOOST"
    else:
        action = str(move.thrust)

    return tx, ty, action


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    laps             = int(input())
    checkpoint_count = int(input())
    checkpoints      = []
    for _ in range(checkpoint_count):
        cx, cy = map(int, input().split())
        checkpoints.append((cx, cy))

    our_states  = [PodState(), PodState()]
    opp_states  = [PodState(), PodState()]

    boosts_left  = 1
    warm_pop     = []
    turn_number  = 0
    is_first_turn = True

    while True:
        x0,  y0,  vx0,  vy0,  a0,  ncp0  = map(int, input().split())
        x1,  y1,  vx1,  vy1,  a1,  ncp1  = map(int, input().split())
        ox0, oy0, ovx0, ovy0, oa0, oncp0  = map(int, input().split())
        ox1, oy1, ovx1, ovy1, oa1, oncp1  = map(int, input().split())

        turn_number += 1

        # Update checkpoint progress
        for st, ncp in zip(our_states, [ncp0, ncp1]):
            st.update(ncp, checkpoint_count)
        for st, ncp in zip(opp_states, [oncp0, oncp1]):
            st.update(ncp, checkpoint_count)

        # Decrement shield cooldowns
        for st in our_states + opp_states:
            if st.shield_cd > 0:
                st.shield_cd -= 1

        # Build SimPod objects for this turn
        our_pods = [
            SimPod(x0,  y0,  vx0,  vy0,  a0,  ncp0,
                   our_states[0].cps_done, our_states[0].shield_cd, our_states[0].boosted),
            SimPod(x1,  y1,  vx1,  vy1,  a1,  ncp1,
                   our_states[1].cps_done, our_states[1].shield_cd, our_states[1].boosted),
        ]
        opp_pods = [
            SimPod(ox0, oy0, ovx0, ovy0, oa0, oncp0,
                   opp_states[0].cps_done, opp_states[0].shield_cd, opp_states[0].boosted),
            SimPod(ox1, oy1, ovx1, ovy1, oa1, oncp1,
                   opp_states[1].cps_done, opp_states[1].shield_cd, opp_states[1].boosted),
        ]

        # ── Run GA ────────────────────────────────────────────────────────────
        best, warm_pop = run_ga(
            our_pods, opp_pods, checkpoints, boosts_left,
            warm_pop, TIME_LIMIT_MS
        )

        # ── Emit outputs ──────────────────────────────────────────────────────
        for pi in range(2):
            move = best.moves[pi][0]
            pod  = our_pods[pi]
            tx, ty, action = _move_to_output(pod, move, checkpoints, boosts_left)

            # Track boost consumption
            if action == "BOOST":
                boosts_left -= 1
                our_states[pi].boosted = True
            if action == "SHIELD":
                our_states[pi].shield_cd = 4

            print(f"{tx} {ty} {action}")

        is_first_turn = False


if __name__ == "__main__":
    main()