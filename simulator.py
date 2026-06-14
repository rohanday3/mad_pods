"""
simulator.py — Exact Mad Pod Racing physics engine.

Rules (from /home/rohan/max_pod/rules):
  1. Rotation: clamp angle change to ±18°/turn (except turn 1: unlimited).
  2. Acceleration: facing_vector * thrust added to velocity.
  3. Movement: add velocity to position; elastic collision if pods overlap.
  4. Friction: velocity *= 0.85.
  5. Position rounded to nearest int; velocity truncated (int).
  6. Elastic collisions; minimum impulse = 120.
  7. BOOST = thrust 650 (one per team, shared across both pods).
  8. SHIELD: mass *= 10 this turn; pod can't accelerate for next 3 turns.
  9. Checkpoint radius = 600; Pod radius = 400.
  10. Map: 16000 x 9000.

Usage:
    sim = Simulator(checkpoints, laps=3)
    state = sim.reset()
    state, done, info = sim.step(actions_team_a, actions_team_b)

actions_team_x is a list of 2 Action objects, one per pod.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ── Constants ────────────────────────────────────────────────────────────────
CHECKPOINT_RADIUS = 600.0
POD_RADIUS = 400.0
FRICTION = 0.85
MAX_ROTATE = 18.0      # degrees
MIN_IMPULSE = 120.0
BOOST_THRUST = 650.0
SHIELD_MASS_MULT = 10.0
NORMAL_MASS = 1.0
MAP_W = 16000
MAP_H = 9000
MAX_TURNS_WITHOUT_CP = 100  # elimination rule


@dataclass
class Action:
    """One pod's command for this turn."""
    target_x: float
    target_y: float
    thrust: int = 100        # 0-100 for normal; use special flags below
    boost: bool = False
    shield: bool = False


@dataclass
class PodPhysics:
    """Full physics state of a single pod."""
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    angle: float = 0.0        # degrees, 0=East, 90=South
    next_cp: int = 0
    cps_passed: int = 0       # total checkpoints passed

    # shield state
    shield_active: bool = False
    shield_cooldown: int = 0  # turns remaining where thrust is blocked

    # boost tracking (per-team, passed in from outside)
    turns_alive: int = 0
    turns_since_last_cp: int = 0

    @property
    def mass(self) -> float:
        return NORMAL_MASS * (SHIELD_MASS_MULT if self.shield_active else 1.0)

    def pos(self) -> Tuple[float, float]:
        return self.x, self.y

    def vel(self) -> Tuple[float, float]:
        return self.vx, self.vy


def _angle_diff(a: float, b: float) -> float:
    """Signed difference b - a, wrapped to [-180, 180]."""
    d = (b - a) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


def _vec_angle_deg(dx: float, dy: float) -> float:
    """Angle of a vector in degrees, 0=East, 90=South (y-down)."""
    return math.degrees(math.atan2(dy, dx)) % 360.0


def _elastic_collision(p1: PodPhysics, p2: PodPhysics) -> None:
    """
    Apply elastic collision between two pods in-place.
    The minimum impulse is 120. Shield multiplies mass by 10.
    """
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    dist = math.hypot(dx, dy)
    if dist == 0:
        dx, dy, dist = 1.0, 0.0, 1.0  # degenerate: push apart horizontally

    # Unit normal
    nx = dx / dist
    ny = dy / dist

    # Relative velocity along normal
    dvx = p1.vx - p2.vx
    dvy = p1.vy - p2.vy
    dot = dvx * nx + dvy * ny

    m1 = p1.mass
    m2 = p2.mass

    # Elastic impulse magnitude
    # J = 2 * m1 * m2 * dot / (m1 + m2)
    J = 2.0 * m1 * m2 * dot / (m1 + m2)

    # Enforce minimum impulse
    if abs(J) < MIN_IMPULSE:
        J = math.copysign(MIN_IMPULSE, J)

    # Apply impulse
    p1.vx -= (J / m1) * nx
    p1.vy -= (J / m1) * ny
    p2.vx += (J / m2) * nx
    p2.vy += (J / m2) * ny


class Simulator:
    """
    Full 4-pod Mad Pod Racing simulator.

    Team A = pods[0], pods[1]
    Team B = pods[2], pods[3]
    """

    def __init__(self, checkpoints: List[Tuple[int, int]], laps: int = 3):
        self.checkpoints = checkpoints
        self.laps = laps
        self.n_cp = len(checkpoints)
        self.total_cps = self.n_cp * laps
        self.turn = 0
        self.pods: List[PodPhysics] = []
        self.boosts_left = [1, 1]   # [team_a, team_b]
        self.done = False
        self.winner = None   # 0 = team A, 1 = team B
        self._first_turn = True

    def reset(self, pods: Optional[List[PodPhysics]] = None) -> List[PodPhysics]:
        """
        Reset the simulation.
        If pods is None, place all 4 pods at checkpoint 0 with a small offset.
        """
        self.turn = 0
        self.boosts_left = [1, 1]
        self.done = False
        self.winner = None
        self._first_turn = True

        if pods is not None:
            self.pods = pods
        else:
            # Default start: all pods at checkpoint 0, slight spread
            cx, cy = self.checkpoints[0]
            self.pods = [
                PodPhysics(x=cx - 600, y=cy - 300, angle=0.0, next_cp=1 % self.n_cp),
                PodPhysics(x=cx - 600, y=cy + 300, angle=0.0, next_cp=1 % self.n_cp),
                PodPhysics(x=cx + 600, y=cy - 300, angle=180.0, next_cp=1 % self.n_cp),
                PodPhysics(x=cx + 600, y=cy + 300, angle=180.0, next_cp=1 % self.n_cp),
            ]
        return list(self.pods)

    def step(
        self,
        actions_a: List[Action],  # team A: [pod0, pod1]
        actions_b: List[Action],  # team B: [pod2, pod3]
    ) -> Tuple[List[PodPhysics], bool, dict]:
        """
        Advance one game turn. Returns (pods_state, done, info).
        """
        if self.done:
            return list(self.pods), True, {"winner": self.winner}

        self.turn += 1
        all_actions = actions_a + actions_b  # 4 total

        # ── 1. Apply actions (rotation + boost/shield bookkeeping) ────────────
        for i, (pod, action) in enumerate(zip(self.pods, all_actions)):
            team = 0 if i < 2 else 1
            pod.turns_alive += 1
            pod.turns_since_last_cp += 1
            pod.shield_active = False  # reset each turn unless activated below

            # Rotation: clamp to ±18° (first turn: unlimited)
            target_dx = action.target_x - pod.x
            target_dy = action.target_y - pod.y
            if math.hypot(target_dx, target_dy) < 1e-3:
                # Target is on the pod — don't rotate
                pass
            else:
                desired_angle = _vec_angle_deg(target_dx, target_dy)
                if self._first_turn:
                    pod.angle = desired_angle
                else:
                    diff = _angle_diff(pod.angle, desired_angle)
                    diff = max(-MAX_ROTATE, min(MAX_ROTATE, diff))
                    pod.angle = (pod.angle + diff) % 360.0

            # Determine thrust
            if pod.shield_cooldown > 0:
                # Shield cooldown: no thrust
                thrust = 0
                pod.shield_cooldown -= 1
            elif action.shield:
                pod.shield_active = True
                pod.shield_cooldown = 3
                thrust = 0
            elif action.boost and self.boosts_left[team] > 0:
                thrust = BOOST_THRUST
                self.boosts_left[team] -= 1
            else:
                thrust = float(action.thrust)

            # Apply acceleration
            rad = math.radians(pod.angle)
            pod.vx += math.cos(rad) * thrust
            pod.vy += math.sin(rad) * thrust

        self._first_turn = False

        # ── 2. Movement ───────────────────────────────────────────────────────
        for pod in self.pods:
            pod.x += pod.vx
            pod.y += pod.vy

        # ── 3. Elastic Collisions (all pairs) ─────────────────────────────────
        for i in range(len(self.pods)):
            for j in range(i + 1, len(self.pods)):
                p1 = self.pods[i]
                p2 = self.pods[j]
                dist = math.hypot(p1.x - p2.x, p1.y - p2.y)
                if dist <= POD_RADIUS * 2:
                    _elastic_collision(p1, p2)
                    # Push apart so they don't overlap
                    if dist < 1.0:
                        dist = 1.0
                    overlap = (POD_RADIUS * 2 - dist) / 2.0 + 1.0
                    nx = (p2.x - p1.x) / dist
                    ny = (p2.y - p1.y) / dist
                    p1.x -= nx * overlap
                    p1.y -= ny * overlap
                    p2.x += nx * overlap
                    p2.y += ny * overlap

        # ── 4. Friction ───────────────────────────────────────────────────────
        for pod in self.pods:
            pod.vx *= FRICTION
            pod.vy *= FRICTION
            # Truncate velocity (int cast)
            pod.vx = math.trunc(pod.vx)
            pod.vy = math.trunc(pod.vy)
            # Round position
            pod.x = round(pod.x)
            pod.y = round(pod.y)

        # ── 5. Checkpoint detection ───────────────────────────────────────────
        for pod in self.pods:
            cp_x, cp_y = self.checkpoints[pod.next_cp]
            if math.hypot(pod.x - cp_x, pod.y - cp_y) <= CHECKPOINT_RADIUS:
                pod.cps_passed += 1
                pod.turns_since_last_cp = 0
                pod.next_cp = (pod.next_cp + 1) % self.n_cp

        # ── 6. Win / Elimination detection ───────────────────────────────────
        for i, pod in enumerate(self.pods):
            team = 0 if i < 2 else 1
            if pod.cps_passed >= self.total_cps:
                self.done = True
                self.winner = team

        # Elimination: both pods of a team fail to reach CP in time
        for team in range(2):
            idxs = [0, 1] if team == 0 else [2, 3]
            if all(self.pods[i].turns_since_last_cp >= MAX_TURNS_WITHOUT_CP for i in idxs):
                self.done = True
                self.winner = 1 - team  # other team wins
                break

        info = {
            "winner": self.winner,
            "turn": self.turn,
            "cps_a": [self.pods[0].cps_passed, self.pods[1].cps_passed],
            "cps_b": [self.pods[2].cps_passed, self.pods[3].cps_passed],
        }
        return list(self.pods), self.done, info

    # ── Utility ───────────────────────────────────────────────────────────────

    def get_progress_score(self, pod: PodPhysics) -> float:
        """Higher is better for the pod's team. Same as heuristic bot."""
        cp_x, cp_y = self.checkpoints[pod.next_cp]
        dist = math.hypot(cp_x - pod.x, cp_y - pod.y)
        return pod.cps_passed * 20000.0 - dist

    @staticmethod
    def random_checkpoints(n: int = None) -> List[Tuple[int, int]]:
        """Generate a random valid circuit. n checkpoints, 2..8."""
        if n is None:
            n = random.randint(2, 8)
        margin = 2000
        checkpoints = []
        for _ in range(n):
            x = random.randint(margin, MAP_W - margin)
            y = random.randint(margin, MAP_H - margin)
            checkpoints.append((x, y))
        return checkpoints
