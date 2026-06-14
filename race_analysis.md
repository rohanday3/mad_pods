# Race Log Analysis: daemon_slayer vs DarthBoss

## Race Overview

- **131 turns**, 3 laps
- **Result**: DarthBoss wins (rank 1 from turn 10 onwards, never relinquished)
- **We led turns 1–9**, then lost the lead at turn 10 and **never recovered**

---

## Timeline of Key Events

| Turn | Event |
|------|-------|
| 1 | Both start. We boost Pod 2 on turn 2. DarthBoss boosts Pod 1 on turn 2. |
| 6–9 | DarthBoss Pod 1 uses **thrust 0** to brake before a checkpoint. Our Pod 2 is also at thrust 0 (blocker idling). |
| **10** | **We lose the lead permanently.** DarthBoss overtakes us. |
| 11 | DarthBoss Pod 1 uses **thrust 29** — precision braking for a tight turn |
| 16–17 | DarthBoss Pod 2 uses **thrust 40**, then **SHIELD** — tactical bump on our Pod 1 |
| 17–22 | Our Pod 1 stuck at **thrust 0** for 6 consecutive turns (blocker doing nothing useful) |
| 57 | Our Pod 1 uses its only SHIELD of the entire game |
| 68 | DarthBoss Pod 1 uses SHIELD — aggressive blocker ramming us |
| 80 | DarthBoss Pod 1 uses SHIELD again |
| 84 | DarthBoss Pod 1 uses SHIELD again (3rd time!) |
| 90 | DarthBoss Pod 1 uses SHIELD (4th time) |
| 122–131 | DarthBoss Pod 1 finishes race using **precision thrust 27–64** for final checkpoints |

---

## 5 Critical Weaknesses Identified

### 1. 🔴 We Only Use Binary Thrust (0 or 100) — Enemy Uses Continuous Thrust

**The biggest gap.** DarthBoss uses finely tuned thrust values like 27, 28, 29, 30, 31, 34, 40, 41, 49, 51, 56, 59, 60, 64, 65, 70, 77, 99. These are **not random** — they're precisely calculated to:

- **Brake smoothly into turns** without overshooting (turns 11, 122–131)
- **Maintain optimal speed through curves** (thrust 56–77 range)
- **Creep through checkpoints** at exactly the right speed for instant turn-around

Our bot only outputs `0` or `100` (or BOOST/SHIELD). This means we either stop dead or go full blast. On tight corners we overshoot, lose time correcting, and waste turns.

> **Fix**: Implement continuous thrust calculation. Instead of `thrust = 0` when braking, calculate `thrust = max(0, int(target_speed - current_speed))` proportional to the angular deviation and distance remaining.

---

### 2. 🔴 Our Blocker Wastes Massive Time at Thrust 0

Between turns 6–11 and again turns 17–22, our **Pod 2 blocker** sits at power 0 for long stretches (6+ consecutive turns!). Meanwhile DarthBoss's blocker is **always doing something useful** — either racing to position, intercepting, or ramming.

Our blocker spent **at least 20 turns** at thrust 0 across the race. That's 20 turns of a pod doing literally nothing.

> **Fix**: The blocker should never idle. If it has no blocking task, it should race as a secondary runner. Even a slow secondary runner is better than a stationary pod.

---

### 3. 🔴 Enemy SHIELD Usage is Far More Aggressive (8 vs 1)

- **DarthBoss used SHIELD 8 times** (turns 17, 19, 58, 68, 70, 80, 84, 90)
- **We used SHIELD 1 time** (turn 57)

DarthBoss's blocker uses SHIELD **offensively** to ram our runner with 10x mass, sending us flying off course. Our shield threshold (`rel_speed > 120`) is too conservative — we almost never trigger it.

> **Fix**: Lower the shield trigger threshold. The blocker should SHIELD whenever it's about to collide with an opponent, period. The runner should SHIELD whenever a collision would deflect it more than ~30° off its heading.

---

### 4. 🟡 We Never Regain the Lead After Losing It

Once DarthBoss overtook us at turn 10, we stayed rank 2 for the remaining **121 turns**. This suggests:
- Our runner isn't fast enough on straights (no variable thrust optimization)
- Our blocker isn't disrupting the enemy runner effectively
- When the enemy's blocker bumps us, we lose too much time recovering

> **Fix**: The blocker needs to be more aggressively targeting the enemy runner with rams + SHIELD combos. When we're behind, the blocker should switch from "positional blocking" to "active pursuit ramming."

---

### 5. 🟡 Corner Speed Management is Poor

DarthBoss approaches checkpoints with carefully modulated thrust (27–34 on the final approach) to arrive at exactly the right speed for the next turn. We either coast at 0 or blaze in at 100 and overshoot.

Looking at turns 122–131, DarthBoss's runner uses thrust 28→28→27→27→28→30→34→40→49→64, gradually accelerating out of a tight series of checkpoints. This is **optimal corner exit technique** — similar to racing games.

> **Fix**: Implement proportional thrust: `thrust = int(100 * (1.0 - angle_error/90.0) * speed_factor)` where `speed_factor` accounts for braking distance.

---

## Priority Implementation Order

1. **Continuous/proportional thrust** — highest impact, affects every single turn
2. **Blocker never idles** — use it as secondary runner when not blocking
3. **More aggressive SHIELD usage** — especially blocker offense
4. **Proportional corner braking** — smooth speed management
5. **Behind-mode pursuit ramming** — when losing, go aggressive
