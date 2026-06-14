"""
nn_train_ppo.py — Train Mad Pod Racing bots with Proximal Policy Optimization (PPO).

Architecture:
  - Actor-Critic network shared trunk, separate heads.
  - Runner policy: 14 inputs → Hidden(64) → Hidden(64) → [action mean (4), action logstd (4), value (1)]
  - Blocker policy: 12 inputs → Hidden(64) → Hidden(64) → [action mean (3), action logstd (3), value (1)]
  - Continuous Gaussian action space (no discrete thrust bins).
  - Parallel rollout collection via multiprocessing (compensates for pure-Python sim).

Training curriculum (same as ES version but PPO learns far faster per step):
  Phase 1 (0  .. PHASE2_GEN): runner navigation only, no opponent
  Phase 2 (PHASE2_GEN .. PHASE3_GEN): runner vs heuristic opponent
  Phase 3 (PHASE3_GEN .. END): runner frozen, train blocker

Usage:
  python3 nn_train_ppo.py

Outputs:
  nn_weights_ppo.json   — final weights (same format as ES version for compatibility)
  live_state.json        — live dashboard feed (same format, same race_viewer.html works)

Requirements:
  pip install torch numpy
"""

import math
import random
import copy
import json
import os
import sys
import time
import numpy as np
import multiprocessing as mp
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from simulator import Simulator, Action, PodPhysics, MAP_W, MAP_H
from champion_opponent import ChampionOpponent

# ── Running statistics (Welford algorithm) ────────────────────────────────────

class RunningMeanStd:
    """
    Incremental mean/variance tracker using Welford's online algorithm.
    Used for both observation normalisation and return normalisation.
    shape=() for scalars (returns), shape=(n,) for vectors (observations).
    """
    def __init__(self, shape=(), epsilon: float = 1e-4):
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape,  dtype=np.float64)
        self.count = epsilon   # start non-zero so std is valid from step 0

    def update(self, x: np.ndarray):
        """x shape: (batch, *shape) or (*shape,) for a single sample."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == len(self.mean.shape):
            x = x[np.newaxis]          # treat single sample as batch of 1
        batch_mean = x.mean(axis=0)
        batch_var  = x.var(axis=0)
        batch_n    = x.shape[0]

        total = self.count + batch_n
        delta = batch_mean - self.mean
        self.mean  = self.mean + delta * batch_n / total
        self.var   = (self.var * self.count
                      + batch_var * batch_n
                      + delta ** 2 * self.count * batch_n / total) / total
        self.count = total

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        normed = (x - self.mean.astype(np.float32)) / np.sqrt(
            self.var.astype(np.float32) + 1e-8)
        return np.clip(normed, -clip, clip)


# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ── Network config ────────────────────────────────────────────────────────────
RUNNER_IN   = 14
RUNNER_H    = 64
RUNNER_ACT  = 4   # [target_dx, target_dy, thrust, boost_logit]

BLOCKER_IN  = 12
BLOCKER_H   = 64
BLOCKER_ACT = 3   # [target_dx, target_dy, thrust]

# ── PPO Hyperparameters ───────────────────────────────────────────────────────
# These are tuned for CodinGame-style pod racing.
# Lower LR than typical PPO because the sim is noisy.
LR_RUNNER        = 3e-4
LR_BLOCKER       = 3e-4
GAMMA            = 0.99        # discount factor
GAE_LAMBDA       = 0.95        # GAE lambda for advantage estimation
CLIP_EPS         = 0.2         # PPO clip ratio
ENTROPY_COEF     = 0.003        # entropy bonus (encourages exploration)
VALUE_COEF       = 0.5         # value loss weight
MAX_GRAD_NORM    = 0.5         # gradient clipping
PPO_EPOCHS       = 8           # optimisation passes per batch
MAX_KL           = 0.015   # stop epochs early if KL exceeds this
MINIBATCH_SIZE   = 512
ROLLOUT_STEPS    = 2048        # steps collected per update (per worker)
N_WORKERS        = max(1, mp.cpu_count() - 1)   # parallel envs
TOTAL_TIMESTEPS  = 5_000_000   # total env steps

# Curriculum phase boundaries (in timesteps)
PHASE2_STEPS = 500_000    # add opponent at 500k steps
PHASE3_STEPS = 2_000_000  # freeze runner, train blocker at 2M steps

LAPS      = 3
MAX_TURNS = 1200

# ── Reward shaping weights ────────────────────────────────────────────────────
# Runner rewards
W_DIST_CLOSE   = 0.02    # reward for closing distance to next CP each step
W_CP_HIT       = 800.0   # sparse reward per checkpoint passed
W_WIN_BONUS    = 15000.0 # terminal reward for winning/losing
W_SPEED_TO_CP  = 0.3     # velocity projected onto the vector toward next CP
                          # (replaces the old W_SPEED constant that was never used)
W_TIME_PENALTY = -0.2    # small per-step cost to discourage dawdling
W_ALIGN        = 0.1     # reward for facing the next CP while moving
W_LEAD         = 0.25     # reward for being ahead of opponent in progress (phase 2)

# Blocker rewards
W_BLOCKER_DIST  = W_DIST_CLOSE * 2.0  # penalty when opponent closes on its CP
W_PROXIMITY     = 0.8    # bonus for staying close to opponent runner (< 1000 units)
W_PROXIMITY_CAP = 1000.0 # distance threshold for proximity bonus

LOG_INTERVAL  = 10   # update cycles between console prints
SAVE_INTERVAL = 20   # update cycles between live_state.json writes


# ── Actor-Critic Network ──────────────────────────────────────────────────────

def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0):
    """Orthogonal initialisation — standard for PPO."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """
    Shared-trunk actor-critic.
    Actor outputs mean of a Gaussian; log_std is a learned parameter (not
    network output) — this is the standard PPO setup for continuous control.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
        )
        # Actor head: small std init so initial actions are near zero
        self.actor_mean = layer_init(nn.Linear(hidden, act_dim), std=0.01)
        # Critic head: std=1 is standard
        self.critic     = layer_init(nn.Linear(hidden, 1), std=1.0)
        # Log std as a free parameter (not input-dependent)
        self.log_std    = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x: torch.Tensor):
        h     = self.trunk(x)
        mean  = self.actor_mean(h)
        value = self.critic(h).squeeze(-1)
        return mean, self.log_std.expand_as(mean), value

    def get_action(self, x: torch.Tensor, deterministic: bool = False):
        mean, log_std, value = self(x)
        if deterministic:
            return mean.tanh(), None, value
        dist   = Normal(mean, log_std.exp())
        action = dist.sample()
        log_p  = dist.log_prob(action).sum(-1)
        return action.tanh(), log_p, value

    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor):
        """Used during PPO update to get log_prob and entropy for stored actions."""
        mean, log_std, value = self(x)
        # actions were tanh-squashed when sampled; invert tanh to get pre-squash
        raw = torch.atanh(actions.clamp(-1 + 1e-6, 1 - 1e-6))
        dist    = Normal(mean, log_std.exp())
        log_p   = dist.log_prob(raw).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_p, entropy, value


# ── Feature extraction (identical to ES version) ─────────────────────────────

def _norm_pos(x, y):
    return x / (MAP_W * 0.5) - 1.0, y / (MAP_H * 0.5) - 1.0

def _norm_vel(vx, vy):
    return vx / 1000.0, vy / 1000.0

def _norm_angle(a):
    return (a / 180.0) - 1.0

def _dist_norm(d):
    return d / 20000.0


def runner_features(pod: PodPhysics, checkpoints, n_cp) -> np.ndarray:
    cp_x, cp_y   = checkpoints[pod.next_cp]
    ncp_idx      = (pod.next_cp + 1) % n_cp
    ncp_x, ncp_y = checkpoints[ncp_idx]
    px,  py      = _norm_pos(pod.x, pod.y)
    vx,  vy      = _norm_vel(pod.vx, pod.vy)
    angle_n      = _norm_angle(pod.angle)
    dcx          = (cp_x - pod.x) / MAP_W
    dcy          = (cp_y - pod.y) / MAP_H
    dist_cp      = _dist_norm(math.hypot(cp_x - pod.x, cp_y - pod.y))
    dnx          = (ncp_x - pod.x) / MAP_W
    dny          = (ncp_y - pod.y) / MAP_H
    desired      = math.degrees(math.atan2(cp_y - pod.y, cp_x - pod.x)) % 360.0
    ang_diff     = ((desired - pod.angle + 180) % 360) - 180
    ang_diff_n   = ang_diff / 180.0
    speed_n      = math.hypot(pod.vx, pod.vy) / 1000.0
    cps_n        = pod.cps_passed / 30.0
    shield_n     = pod.shield_cooldown / 3.0
    return np.array([
        px, py, vx, vy, angle_n,
        dcx, dcy, dist_cp, ang_diff_n,
        dnx, dny,
        speed_n, cps_n, shield_n
    ], dtype=np.float32)


def blocker_features(blocker: PodPhysics, opp_runner: PodPhysics,
                     checkpoints, n_cp) -> np.ndarray:
    ocp_x, ocp_y = checkpoints[opp_runner.next_cp]
    bpx, bpy     = _norm_pos(blocker.x, blocker.y)
    bvx, bvy     = _norm_vel(blocker.vx, blocker.vy)
    dx_opp       = (opp_runner.x - blocker.x) / MAP_W
    dy_opp       = (opp_runner.y - blocker.y) / MAP_H
    dist_opp     = _dist_norm(math.hypot(opp_runner.x - blocker.x,
                                          opp_runner.y - blocker.y))
    ovx, ovy     = _norm_vel(opp_runner.vx, opp_runner.vy)
    dx_cp        = (ocp_x - blocker.x) / MAP_W
    dy_cp        = (ocp_y - blocker.y) / MAP_H
    angle_n      = _norm_angle(blocker.angle)
    speed_n      = math.hypot(blocker.vx, blocker.vy) / 1000.0
    return np.array([
        bpx, bpy, bvx, bvy,
        dx_opp, dy_opp, dist_opp,
        ovx, ovy,
        dx_cp, dy_cp,
        angle_n
    ], dtype=np.float32)


# ── Action decoding ───────────────────────────────────────────────────────────

def decode_runner_action(raw: np.ndarray, pod: PodPhysics,
                         boosts_left: list) -> Action:
    """
    raw: tanh-squashed 4-vector from actor.
    Decode into simulator Action.
    """
    # raw[0], raw[1] in [-1,1] → target offset from pod position
    tx     = pod.x + raw[0] * 3000.0
    ty     = pod.y + raw[1] * 2000.0
    thrust = int((raw[2] + 1.0) * 50.0)   # [-1,1] → [0,100]
    thrust = max(0, min(100, thrust))
    use_boost = bool(raw[3] > 0.0 and boosts_left[0] > 0)
    if use_boost:
        boosts_left[0] -= 1
    return Action(target_x=tx, target_y=ty, thrust=thrust, boost=use_boost)


def decode_blocker_action(raw: np.ndarray, pod: PodPhysics) -> Action:
    tx     = pod.x + raw[0] * 3000.0
    ty     = pod.y + raw[1] * 2000.0
    thrust = int((raw[2] + 1.0) * 50.0)
    thrust = max(0, min(100, thrust))
    return Action(target_x=tx, target_y=ty, thrust=thrust)


# ── Heuristic (opponent + blocker fallback) ───────────────────────────────────

def heuristic_action(pod: PodPhysics, checkpoints, n_cp,
                     boosts_left: list, turn: int) -> Action:
    cp_x, cp_y   = checkpoints[pod.next_cp]
    ncp_x, ncp_y = checkpoints[(pod.next_cp + 1) % n_cp]
    dist_to_cp   = math.hypot(cp_x - pod.x, cp_y - pod.y)
    entering     = False
    px, py, pvx, pvy = pod.x, pod.y, pod.vx, pod.vy
    for _ in range(6):
        pvx *= 0.85; pvy *= 0.85; px += pvx; py += pvy
        if math.hypot(px - cp_x, py - cp_y) <= 600:
            entering = True; break
    if entering:
        blend = max(0.0, min(1.0, (dist_to_cp - 600.0) / 1400.0))
        tx = blend * cp_x + (1 - blend) * ncp_x
        ty = blend * cp_y + (1 - blend) * ncp_y
    else:
        tx, ty = float(cp_x), float(cp_y)
    desired  = math.degrees(math.atan2(ty - pod.y, tx - pod.x)) % 360.0
    ang_diff = abs(((desired - pod.angle + 180) % 360) - 180)
    thrust   = 0 if ang_diff >= 90 else int(math.ceil(100.0 * math.cos(ang_diff / 180.0)))
    use_boost = boosts_left[0] > 0 and turn > 10 and ang_diff < 5.0 and dist_to_cp > 5000
    if use_boost:
        boosts_left[0] -= 1
    return Action(target_x=tx, target_y=ty, thrust=thrust, boost=use_boost)


# ── Reward shaping ────────────────────────────────────────────────────────────

def _dist_to_cp(pod, checkpoints):
    cp_x, cp_y = checkpoints[pod.next_cp]
    return math.hypot(cp_x - pod.x, cp_y - pod.y)


def _speed_toward_cp(pod: PodPhysics, checkpoints) -> float:
    """
    Dot product of the pod's velocity with the unit vector pointing at its
    next checkpoint.  Positive = moving toward the CP, negative = moving away.
    Range is roughly [-700, +700] for typical in-game speeds.
    """
    cp_x, cp_y = checkpoints[pod.next_cp]
    dx = cp_x - pod.x
    dy = cp_y - pod.y
    dist = math.hypot(dx, dy) + 1e-6
    return (pod.vx * dx + pod.vy * dy) / dist


def _alignment_reward(pod: PodPhysics, checkpoints) -> float:
    """
    How well the pod is facing its next checkpoint.
    Returns +1 when pointing directly at it, 0 when perpendicular,
    and -1 when facing directly away.
    """
    cp_x, cp_y = checkpoints[pod.next_cp]
    desired  = math.degrees(math.atan2(cp_y - pod.y, cp_x - pod.x)) % 360.0
    ang_diff = abs(((desired - pod.angle + 180) % 360) - 180)  # 0..180
    return 1.0 - (ang_diff / 90.0)   # maps [0°→+1, 90°→0, 180°→-1]


def compute_runner_reward(pods, prev_our_dist, prev_opp_dist,
                          prev_our_cps, checkpoints, done, info,
                          with_opponent: bool) -> Tuple[float, float, float, int]:
    """
    Returns (reward, new_our_dist, new_opp_dist, new_our_cps).

    Reward components (applied every step unless noted):
      1. Distance-closing      — how much closer we got to the next CP
      2. Speed toward CP       — velocity projected onto the CP direction vector;
                                 rewards going fast *in the right direction*
      3. Alignment bonus       — facing the CP while moving; penalises spinning
      4. Time penalty          — small per-step cost to discourage dawdling
      5. CP hit                — sparse bonus each time a checkpoint is passed
      6. Relative lead         — continuous race-position signal vs opponent (phase 2)
      7. Opponent progress tax — penalise opponent closing on their CP (phase 2)
      8. Win/loss terminal     — large terminal bonus/penalty (phase 2)
    """
    reward = 0.0

    new_our_dist = _dist_to_cp(pods[0], checkpoints)
    new_opp_dist = _dist_to_cp(pods[2], checkpoints) if with_opponent else prev_opp_dist

    # 1. Distance-closing reward (keep — reliable dense signal)
    reward += (prev_our_dist - new_our_dist) * W_DIST_CLOSE

    # 2. Speed projected toward the next CP
    #    Better than raw speed: going fast *away* from the CP gives no reward.
    reward += _speed_toward_cp(pods[0], checkpoints) * W_SPEED_TO_CP

    # 3. Alignment bonus — only meaningful when actually moving
    speed = math.hypot(pods[0].vx, pods[0].vy)
    if speed > 100.0:
        reward += _alignment_reward(pods[0], checkpoints) * W_ALIGN

    # 4. Time pressure — nudges the agent to finish quickly
    reward += W_TIME_PENALTY

    # 5. Sparse checkpoint hit
    new_our_cps = pods[0].cps_passed
    if new_our_cps > prev_our_cps:
        reward += (new_our_cps - prev_our_cps) * W_CP_HIT

    # 6 & 7. Opponent-relative signals (phase 2 only)
    if with_opponent:
        # 6. Relative race lead: combines CP count and distance-to-next-CP into
        #    a single continuous "who is winning" signal.
        our_progress   = pods[0].cps_passed - new_our_dist / 10000.0
        their_progress = pods[2].cps_passed - new_opp_dist / 10000.0
        reward += (our_progress - their_progress) * W_LEAD

        # 7. Penalise opponent closing their own CP gap
        reward -= (prev_opp_dist - new_opp_dist) * W_DIST_CLOSE * 0.5

    # 8. Terminal win/loss
    if done and with_opponent:
        if info.get("winner") == 0:
            reward += W_WIN_BONUS
        elif info.get("winner") == 1:
            reward -= W_WIN_BONUS

    return reward, new_our_dist, new_opp_dist, new_our_cps


def compute_blocker_reward(pods, prev_opp_dist, checkpoints,
                           done, info) -> Tuple[float, float]:
    """
    Blocker reward: discourage the opponent runner from advancing.

    Components:
      1. Opponent progress tax — penalise opponent closing on its CP
      2. Proximity bonus       — reward staying within collision range of opponent;
                                 encourages active harassment rather than passive shadowing
      3. Win terminal          — large bonus if our team wins
    """
    new_opp_dist = _dist_to_cp(pods[2], checkpoints)

    # 1. Penalise opponent closing on its next CP
    reward = (prev_opp_dist - new_opp_dist) * (-W_BLOCKER_DIST)

    # 2. Proximity bonus: stay close enough to threaten a collision.
    #    Scales linearly from W_PROXIMITY at distance=0 down to 0 at W_PROXIMITY_CAP.
    dist_to_opp = math.hypot(pods[1].x - pods[2].x, pods[1].y - pods[2].y)
    if dist_to_opp < W_PROXIMITY_CAP:
        reward += W_PROXIMITY * (1.0 - dist_to_opp / W_PROXIMITY_CAP)

    # 3. Terminal win bonus
    if done and info.get("winner") == 0:
        reward += W_WIN_BONUS * 0.5

    return reward, new_opp_dist


# ── Rollout buffer ────────────────────────────────────────────────────────────

class RolloutBuffer:
    """Stores transitions for one PPO update cycle."""
    def __init__(self, n_steps: int, obs_dim: int, act_dim: int):
        self.obs      = np.zeros((n_steps, obs_dim), dtype=np.float32)
        self.actions  = np.zeros((n_steps, act_dim), dtype=np.float32)
        self.log_probs= np.zeros(n_steps, dtype=np.float32)
        self.rewards  = np.zeros(n_steps, dtype=np.float32)
        self.values   = np.zeros(n_steps, dtype=np.float32)
        self.dones    = np.zeros(n_steps, dtype=np.float32)
        self.ptr      = 0
        self.n_steps  = n_steps

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr]   = reward
        self.values[self.ptr]    = value
        self.dones[self.ptr]     = done
        self.ptr += 1

    def full(self):
        return self.ptr >= self.n_steps

    def reset(self):
        self.ptr = 0

    def compute_returns_and_advantages(self, last_value: float):
        """GAE-Lambda advantage estimation."""
        advantages = np.zeros(self.n_steps, dtype=np.float32)
        last_gae   = 0.0
        for t in reversed(range(self.n_steps)):
            next_val   = last_value if t == self.n_steps - 1 else self.values[t + 1]
            next_done  = self.dones[t]
            delta      = (self.rewards[t]
                          + GAMMA * next_val * (1 - next_done)
                          - self.values[t])
            last_gae   = delta + GAMMA * GAE_LAMBDA * (1 - next_done) * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values
        return returns, advantages

    def get_tensors(self, last_value: float, device):
        returns, advantages = self.compute_returns_and_advantages(last_value)
        return (
            torch.FloatTensor(self.obs).to(device),
            torch.FloatTensor(self.actions).to(device),
            torch.FloatTensor(self.log_probs).to(device),
            torch.FloatTensor(returns).to(device),
            torch.FloatTensor(advantages).to(device),
        )


# ── PPO Update ────────────────────────────────────────────────────────────────

def ppo_update(policy: ActorCritic, optimizer: optim.Optimizer,
               buffer: RolloutBuffer, last_value: float, device):
    obs, actions, old_log_probs, returns, advantages = \
        buffer.get_tensors(last_value, device)

    # ── Advantage normalisation ────────────────────────────────────────────
    # Rewards are already normalised at collection time so GAE is consistent.
    # We still normalise advantages here for a stable actor gradient scale.
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Old values for value-function clipping (same scale as buffer rewards)
    old_values   = buffer.values   # numpy, shape (n_steps,)
    old_values_t = torch.FloatTensor(old_values).to(device)

    n       = obs.shape[0]
    indices = np.arange(n)
    clip_fracs = []
    approx_kls = []

    for _ in range(PPO_EPOCHS):
        np.random.shuffle(indices)
        for start in range(0, n, MINIBATCH_SIZE):
            mb_idx  = indices[start:start + MINIBATCH_SIZE]
            mb_obs  = obs[mb_idx]
            mb_act  = actions[mb_idx]
            mb_olp  = old_log_probs[mb_idx]
            mb_ret  = returns[mb_idx]
            mb_adv  = advantages[mb_idx]
            mb_oval = old_values_t[mb_idx]

            new_log_probs, entropy, values = policy.evaluate_actions(mb_obs, mb_act)

            log_ratio  = new_log_probs - mb_olp
            ratio      = log_ratio.exp()

            # Approx KL (for diagnostics — alert if > 0.02)
            approx_kl  = ((ratio - 1) - log_ratio).mean().item()
            approx_kls.append(approx_kl)

            if approx_kl > MAX_KL:
                break   # inside the epoch loop

            clip_frac  = ((ratio - 1.0).abs() > CLIP_EPS).float().mean().item()
            clip_fracs.append(clip_frac)

            surr1      = ratio * mb_adv
            surr2      = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
            actor_loss = -torch.min(surr1, surr2).mean()

            # ── Value-function loss with clipping ─────────────────────────
            # Prevents critic from taking huge steps when reward scale jumps
            # between phases (e.g. phase1→2 reward changes from ~300 to ~-10k).
            v_clipped  = mb_oval + (values - mb_oval).clamp(-CLIP_EPS, CLIP_EPS)
            vf_loss1   = (values    - mb_ret).pow(2)
            vf_loss2   = (v_clipped - mb_ret).pow(2)
            value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

            ent_loss   = -entropy.mean()

            loss = actor_loss + VALUE_COEF * value_loss + ENTROPY_COEF * ent_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()

    mean_kl = float(np.mean(approx_kls))
    if mean_kl > 0.02:
        print(f"  [WARNING] approx_kl={mean_kl:.4f} > 0.02 — policy update too large")

    return {
        "actor_loss":  actor_loss.item(),
        "value_loss":  value_loss.item(),
        "entropy":    -ent_loss.item(),
        "clip_frac":   float(np.mean(clip_fracs)),
        "approx_kl":   mean_kl,
    }


# ── Single-process rollout collection ────────────────────────────────────────
# (multiprocessing with PyTorch requires careful handling of shared memory;
# we use a simpler approach: collect N_WORKERS episodes sequentially per
# update, or spawn workers that return numpy arrays.)

def collect_rollout_runner(policy: ActorCritic, n_steps: int,
                           phase: int, device,
                           frozen_blocker_weights=None,
                           obs_rms: RunningMeanStd = None,
                           reward_rms: RunningMeanStd = None) -> Tuple[RolloutBuffer, dict]:
    """
    Collect runner rollout steps.
    phase 1: no opponent
    phase 2+: vs ChampionOpponent
    obs_rms:    if provided, observations are normalised using Welford stats.
    reward_rms: if provided, rewards are normalised by running std before
                buffering, keeping GAE consistent with critic output scale.
    Returns buffer + info dict.
    """
    buffer      = RolloutBuffer(n_steps, RUNNER_IN, RUNNER_ACT)
    ep_rewards  = []
    ep_lengths  = []
    ep_wins     = []

    with_opponent = phase >= 2

    # Run episodes until buffer is full
    while not buffer.full():
        checkpoints = Simulator.random_checkpoints(random.randint(3, 6))
        sim         = Simulator(checkpoints, laps=LAPS)
        sim.reset()
        our_boosts  = [1]
        opp         = ChampionOpponent(checkpoints, laps=LAPS) if with_opponent else None

        prev_our_dist = _dist_to_cp(sim.pods[0], checkpoints)
        prev_opp_dist = _dist_to_cp(sim.pods[2], checkpoints)
        prev_our_cps  = 0
        ep_reward     = 0.0
        ep_len        = 0

        for turn in range(MAX_TURNS):
            if buffer.full():
                break
            pods = sim.pods

            # Runner observation + action
            obs_np  = runner_features(pods[0], checkpoints, sim.n_cp)
            if obs_rms is not None:
                obs_rms.update(obs_np)
                obs_np_norm = obs_rms.normalize(obs_np)
            else:
                obs_np_norm = obs_np
            obs_t   = torch.FloatTensor(obs_np_norm).unsqueeze(0).to(device)

            with torch.no_grad():
                action_t, log_p_t, value_t = policy.get_action(obs_t)

            action_np = action_t.squeeze(0).cpu().numpy()
            log_p     = log_p_t.item()
            value     = value_t.item()
            r_act     = decode_runner_action(action_np, pods[0], our_boosts)

            # Blocker: use frozen NN if available, else heuristic
            if frozen_blocker_weights is not None:
                b_feat  = blocker_features(pods[1], pods[2], checkpoints, sim.n_cp)
                b_obs_t = torch.FloatTensor(b_feat).unsqueeze(0).to(device)
                with torch.no_grad():
                    b_act_t, _, _ = frozen_blocker_weights.get_action(b_obs_t,
                                                                       deterministic=True)
                b_act = decode_blocker_action(b_act_t.squeeze(0).cpu().numpy(), pods[1])
            else:
                b_act = heuristic_action(pods[1], checkpoints, sim.n_cp, [0], turn)

            # Opponent
            if with_opponent:
                o_act1, o_act2 = opp.get_actions(pods)
            else:
                o_act1 = Action(target_x=pods[2].x, target_y=pods[2].y, thrust=0)
                o_act2 = Action(target_x=pods[3].x, target_y=pods[3].y, thrust=0)

            _, done, info = sim.step([r_act, b_act], [o_act1, o_act2])

            reward, prev_our_dist, prev_opp_dist, prev_our_cps = \
                compute_runner_reward(pods, prev_our_dist, prev_opp_dist,
                                      prev_our_cps, checkpoints, done, info,
                                      with_opponent)

            # Normalise reward by running std so GAE stays consistent with
            # critic output scale across phases. Update rms with raw reward,
            # then divide — don't subtract mean to preserve reward sign.
            reward_buf = reward
            if reward_rms is not None:
                reward_rms.update(np.array(reward, dtype=np.float64))
                reward_buf = float(reward / (np.sqrt(reward_rms.var) + 1e-8))

            buffer.add(obs_np_norm, action_np, log_p, reward_buf, value, float(done))
            ep_reward += reward   # track raw reward for logging
            ep_len    += 1

            if done:
                ep_rewards.append(ep_reward)
                ep_lengths.append(ep_len)
                ep_wins.append(1 if info.get("winner") == 0 else 0)
                break

    # Bootstrap last value
    pods    = sim.pods
    obs_np  = runner_features(pods[0], checkpoints, sim.n_cp)
    if obs_rms is not None:
        obs_np = obs_rms.normalize(obs_np)
    obs_t   = torch.FloatTensor(obs_np).unsqueeze(0).to(device)
    with torch.no_grad():
        _, _, last_val = policy.get_action(obs_t)
    last_value = last_val.item()

    info_out = {
        "mean_ep_reward": np.mean(ep_rewards) if ep_rewards else 0.0,
        "mean_ep_len":    np.mean(ep_lengths) if ep_lengths else 0.0,
        "win_rate":       np.mean(ep_wins)    if ep_wins    else 0.0,
        "n_episodes":     len(ep_rewards),
    }
    return buffer, last_value, info_out


def collect_rollout_blocker(policy: ActorCritic, runner_policy: ActorCritic,
                            n_steps: int, device,
                            obs_rms_runner: RunningMeanStd = None,
                            obs_rms_blocker: RunningMeanStd = None,
                            reward_rms: RunningMeanStd = None) -> Tuple[RolloutBuffer, dict]:
    """Collect blocker rollout steps. Runner is frozen."""
    buffer     = RolloutBuffer(n_steps, BLOCKER_IN, BLOCKER_ACT)
    ep_rewards = []
    ep_lengths = []

    while not buffer.full():
        checkpoints = Simulator.random_checkpoints(random.randint(3, 6))
        sim         = Simulator(checkpoints, laps=LAPS)
        sim.reset()
        our_boosts  = [1]
        opp         = ChampionOpponent(checkpoints, laps=LAPS)

        prev_opp_dist = _dist_to_cp(sim.pods[2], checkpoints)
        ep_reward     = 0.0
        ep_len        = 0

        for turn in range(MAX_TURNS):
            if buffer.full():
                break
            pods = sim.pods

            # Runner: frozen policy, deterministic
            r_feat  = runner_features(pods[0], checkpoints, sim.n_cp)
            if obs_rms_runner is not None:
                r_feat = obs_rms_runner.normalize(r_feat)
            r_obs_t = torch.FloatTensor(r_feat).unsqueeze(0).to(device)
            with torch.no_grad():
                r_act_t, _, _ = runner_policy.get_action(r_obs_t, deterministic=True)
            r_act = decode_runner_action(r_act_t.squeeze(0).cpu().numpy(),
                                         pods[0], our_boosts)

            # Blocker: learning policy
            obs_np  = blocker_features(pods[1], pods[2], checkpoints, sim.n_cp)
            if obs_rms_blocker is not None:
                obs_rms_blocker.update(obs_np)
                obs_np = obs_rms_blocker.normalize(obs_np)
            obs_t   = torch.FloatTensor(obs_np).unsqueeze(0).to(device)
            with torch.no_grad():
                b_act_t, log_p_t, value_t = policy.get_action(obs_t)
            action_np = b_act_t.squeeze(0).cpu().numpy()
            log_p     = log_p_t.item()
            value     = value_t.item()
            b_act     = decode_blocker_action(action_np, pods[1])

            o_act1, o_act2 = opp.get_actions(pods)
            _, done, info  = sim.step([r_act, b_act], [o_act1, o_act2])

            reward, prev_opp_dist = compute_blocker_reward(
                pods, prev_opp_dist, checkpoints, done, info)

            reward_buf = reward
            if reward_rms is not None:
                reward_rms.update(np.array(reward, dtype=np.float64))
                reward_buf = float(reward / (np.sqrt(reward_rms.var) + 1e-8))

            buffer.add(obs_np, action_np, log_p, reward_buf, value, float(done))
            ep_reward += reward   # track raw reward for logging
            ep_len    += 1

            if done:
                ep_rewards.append(ep_reward)
                ep_lengths.append(ep_len)
                break

    # Bootstrap
    pods    = sim.pods
    obs_np  = blocker_features(pods[1], pods[2], checkpoints, sim.n_cp)
    if obs_rms_blocker is not None:
        obs_np = obs_rms_blocker.normalize(obs_np)
    obs_t   = torch.FloatTensor(obs_np).unsqueeze(0).to(device)
    with torch.no_grad():
        _, _, last_val = policy.get_action(obs_t)
    last_value = last_val.item()

    info_out = {
        "mean_ep_reward": np.mean(ep_rewards) if ep_rewards else 0.0,
        "mean_ep_len":    np.mean(ep_lengths) if ep_lengths else 0.0,
        "n_episodes":     len(ep_rewards),
    }
    return buffer, last_value, info_out


# ── Live state writer ─────────────────────────────────────────────────────────

def runner_policy_to_flat(policy: ActorCritic) -> List[float]:
    """Export weights in same flat format as ES version for compatibility."""
    sd = policy.state_dict()
    parts = [
        sd['trunk.0.weight'].cpu().numpy(),   # W1
        sd['trunk.0.bias'].cpu().numpy(),      # b1
        sd['trunk.2.weight'].cpu().numpy(),    # W2
        sd['trunk.2.bias'].cpu().numpy(),      # b2
        sd['actor_mean.weight'].cpu().numpy(), # W3
        sd['actor_mean.bias'].cpu().numpy(),   # b3
    ]
    return np.concatenate([p.flatten() for p in parts]).tolist()


def record_live_race(runner_policy: ActorCritic, blocker_policy: ActorCritic,
                     device, seed: int = 0):
    random.seed(seed)
    checkpoints = Simulator.random_checkpoints(random.randint(3, 5))
    sim         = Simulator(checkpoints, laps=LAPS)
    sim.reset()
    our_boosts  = [1]
    opp         = ChampionOpponent(checkpoints, laps=LAPS)
    frames      = []

    for turn in range(MAX_TURNS):
        pods = sim.pods
        frames.append({"t": turn, "pods": [
            {"x": p.x, "y": p.y, "vx": p.vx, "vy": p.vy,
             "angle": p.angle, "next_cp": p.next_cp,
             "cps": p.cps_passed, "shield": p.shield_active}
            for p in pods
        ]})
        r_feat  = runner_features(pods[0], checkpoints, sim.n_cp)
        r_obs_t = torch.FloatTensor(r_feat).unsqueeze(0).to(device)
        with torch.no_grad():
            r_act_t, _, _ = runner_policy.get_action(r_obs_t, deterministic=True)
        r_act = decode_runner_action(r_act_t.squeeze(0).cpu().numpy(), pods[0], our_boosts)

        b_feat  = blocker_features(pods[1], pods[2], checkpoints, sim.n_cp)
        b_obs_t = torch.FloatTensor(b_feat).unsqueeze(0).to(device)
        with torch.no_grad():
            b_act_t, _, _ = blocker_policy.get_action(b_obs_t, deterministic=True)
        b_act = decode_blocker_action(b_act_t.squeeze(0).cpu().numpy(), pods[1])

        o_act1, o_act2 = opp.get_actions(pods)
        _, done, info  = sim.step([r_act, b_act], [o_act1, o_act2])
        if done:
            break

    return checkpoints, frames, sim.winner


def write_live_state(update_idx, timestep, total_timesteps, phase,
                     runner_policy, blocker_policy, device,
                     history, ppo_info):
    try:
        checkpoints, frames, winner = record_live_race(
            runner_policy, blocker_policy, device, seed=update_idx % 20)
        state = {
            "gen":          update_idx,
            "phase":        phase,
            "gen_fitness":  float(ppo_info.get("mean_ep_reward", 0)),
            "best_fitness": float(max((h[2] for h in history), default=0)),
            "sigma":        float(ppo_info.get("entropy", 0)),
            "elapsed":      float(ppo_info.get("elapsed", 0)),
            "history":      history,
            "checkpoints":  checkpoints,
            "laps":         LAPS,
            "total_cps":    len(checkpoints) * LAPS,
            "map_w":        MAP_W,
            "map_h":        MAP_H,
            "frames":       frames,
            "winner":       winner,
            "team_names":   ["NN Bot", "Heuristic"],
            "max_gens":     total_timesteps // ROLLOUT_STEPS,
            "runner_weights":  runner_policy_to_flat(runner_policy),
            "blocker_weights": runner_policy_to_flat(blocker_policy),
            # PPO-specific extras (dashboard ignores unknown keys)
            "win_rate":     float(ppo_info.get("win_rate", 0)),
            "clip_frac":    float(ppo_info.get("clip_frac", 0)),
            "timestep":     timestep,
        }
        tmp = "live_state.tmp.json"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, "live_state.json")
    except Exception as e:
        print(f"[live_state write error] {e}")


# ── Weight export (for embedding in submission bot) ───────────────────────────

def export_weights(runner_policy: ActorCritic, blocker_policy: ActorCritic):
    def policy_dict(p: ActorCritic, obs_dim, act_dim, hidden):
        sd = p.state_dict()
        return {
            "trunk_w0": sd['trunk.0.weight'].cpu().tolist(),
            "trunk_b0": sd['trunk.0.bias'].cpu().tolist(),
            "trunk_w2": sd['trunk.2.weight'].cpu().tolist(),
            "trunk_b2": sd['trunk.2.bias'].cpu().tolist(),
            "actor_w":  sd['actor_mean.weight'].cpu().tolist(),
            "actor_b":  sd['actor_mean.bias'].cpu().tolist(),
            "obs_dim":  obs_dim,
            "act_dim":  act_dim,
            "hidden":   hidden,
        }
    weights = {
        "runner":  policy_dict(runner_policy,  RUNNER_IN,  RUNNER_ACT,  RUNNER_H),
        "blocker": policy_dict(blocker_policy, BLOCKER_IN, BLOCKER_ACT, BLOCKER_H),
    }
    with open("nn_weights_ppo.json", "w") as f:
        json.dump(weights, f)
    print("Weights saved to nn_weights_ppo.json")


# ── Main training loop ────────────────────────────────────────────────────────

def get_phase(timestep: int) -> int:
    if timestep < PHASE2_STEPS: return 1
    if timestep < PHASE3_STEPS: return 2
    return 3


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Workers: {N_WORKERS}  |  Rollout steps per update: {ROLLOUT_STEPS}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Curriculum: phase1=navigation (0-{PHASE2_STEPS:,}), "
          f"phase2=vs-opponent ({PHASE2_STEPS:,}-{PHASE3_STEPS:,}), "
          f"phase3=blocker ({PHASE3_STEPS:,}-{TOTAL_TIMESTEPS:,})\n")

    runner_policy  = ActorCritic(RUNNER_IN,  RUNNER_ACT,  RUNNER_H).to(device)
    blocker_policy = ActorCritic(BLOCKER_IN, BLOCKER_ACT, BLOCKER_H).to(device)

    runner_opt  = optim.Adam(runner_policy.parameters(),  lr=LR_RUNNER,  eps=1e-5)
    blocker_opt = optim.Adam(blocker_policy.parameters(), lr=LR_BLOCKER, eps=1e-5)

    # Running stats for obs and reward normalisation
    obs_rms_runner   = RunningMeanStd(shape=(RUNNER_IN,))
    obs_rms_blocker  = RunningMeanStd(shape=(BLOCKER_IN,))
    # Reward normalised by running std at collection time — keeps GAE
    # consistent with critic output scale across all phases automatically.
    reward_rms_runner  = RunningMeanStd(shape=())
    reward_rms_blocker = RunningMeanStd(shape=())

    # LR annealing schedulers
    def lr_lambda(update):
        frac = 1.0 - update / (TOTAL_TIMESTEPS // ROLLOUT_STEPS)
        return max(frac, 0.01)

    runner_sched  = optim.lr_scheduler.LambdaLR(runner_opt,  lr_lambda)
    blocker_sched = optim.lr_scheduler.LambdaLR(blocker_opt, lr_lambda)

    timestep    = 0
    update_idx  = 0
    history     = []   # [update_idx, mean_ep_reward, best_mean_reward]
    best_reward = -1e18
    t0          = time.time()
    prev_phase  = -1

    while timestep < TOTAL_TIMESTEPS:
        phase = get_phase(timestep)

        if phase != prev_phase:
            prev_phase  = phase
            best_reward = -1e18
            labels = {1: "NAVIGATION (no opponent)",
                      2: "RACING vs heuristic",
                      3: "BLOCKER training (runner frozen)"}
            print(f"\n>>> Phase {phase}: {labels[phase]} <<<\n")

        # ── Collect rollout ───────────────────────────────────────────────
        if phase <= 2:
            runner_policy.train()
            buffer, last_val, ep_info = collect_rollout_runner(
                runner_policy, ROLLOUT_STEPS, phase, device,
                frozen_blocker_weights=None,
                obs_rms=obs_rms_runner,
                reward_rms=reward_rms_runner)
            ppo_info = ppo_update(runner_policy, runner_opt, buffer, last_val, device)
            runner_sched.step()
        else:
            blocker_policy.train()
            runner_policy.eval()
            buffer, last_val, ep_info = collect_rollout_blocker(
                blocker_policy, runner_policy, ROLLOUT_STEPS, device,
                obs_rms_runner=obs_rms_runner,
                obs_rms_blocker=obs_rms_blocker,
                reward_rms=reward_rms_blocker)
            ppo_info = ppo_update(blocker_policy, blocker_opt, buffer, last_val, device)
            blocker_sched.step()

        timestep   += ROLLOUT_STEPS
        update_idx += 1
        elapsed     = time.time() - t0

        mean_reward = ep_info["mean_ep_reward"]
        if mean_reward > best_reward:
            best_reward = mean_reward
            mark = " *** NEW BEST ***"
            export_weights(runner_policy, blocker_policy)
            # also save a frozen copy
            torch.save(runner_policy.state_dict(),  "runner_best.pt")
            torch.save(blocker_policy.state_dict(), "blocker_best.pt")
        else:
            mark = ""

        history.append([update_idx, float(mean_reward), float(best_reward)])
        if len(history) > 500:
            history = history[-500:]

        ppo_info["elapsed"]    = elapsed
        ppo_info["win_rate"]   = ep_info.get("win_rate", 0.0)

        if update_idx % SAVE_INTERVAL == 0 or mark:
            write_live_state(update_idx, timestep, TOTAL_TIMESTEPS, phase,
                             runner_policy, blocker_policy, device,
                             history, ppo_info)

        if update_idx % LOG_INTERVAL == 0 or mark:
            steps_per_sec = timestep / elapsed if elapsed > 0 else 0
            print(
                f"[{timestep:>8,}] update={update_idx:4d} ph={phase} "
                f"| rew={mean_reward:8.1f} best={best_reward:8.1f} "
                f"| win={ep_info.get('win_rate',0):.2f} "
                f"| ent={ppo_info['entropy']:.3f} "
                f"| clip={ppo_info['clip_frac']:.3f} "
                f"| kl={ppo_info['approx_kl']:.4f} "
                f"| {steps_per_sec:.0f} sps "
                f"| {elapsed:.0f}s{mark}"
            )

        sys.stdout.flush()

    print("\n=== Training complete ===")
    export_weights(runner_policy, blocker_policy)


if __name__ == "__main__":
    train()
