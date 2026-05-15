# Iteration Log

Tracks all manual training iterations for the Visual Eureka project. Each entry records what changed, why, the reward function used, training config, results, and lessons learned.

Use this log to avoid repeating failed approaches and to understand the progression of reward design decisions.

---

## Go2 Quadruped — Initial Debugging Session (2026-05-15)

### Environment Bugs Fixed Before Training

These issues affected ALL training runs and had to be fixed first.

| Bug | Root Cause | Impact | Fix |
|---|---|---|---|
| 4-second episodes | No frame-skipping: single `mj_step` per action (dt=0.002s) | 2000 steps = 4 seconds of sim time, not enough for locomotion | Added `n_substeps=5` for quadruped, effective dt=0.01s per action |
| Robot in free-fall | No ground plane geom in `go2.xml` | Robot falls through empty space, foot contacts never fire | Added `<geom name="floor" type="plane" .../>` to worldbody |
| Intractable torque control | Policy outputs mapped directly to raw torques | Robot can't discover gravity-compensating torques from scratch | Added PD position control layer: policy outputs joint offsets from home pose, env converts to torques with kp=20, kd=0.5 |
| Catastrophic action penalty | `data.ctrl` used for action energy (ctrl range [-45, +45]) | `-0.005 * sum(ctrl^2)` = -21.5/step, robot learns to do nothing | Changed to `info["action"]` which is normalized [-1, 1] |
| Phantom survival bonus | LLM prompt said env adds +1.0 survival, but env never did | LLM skipped survival in reward, robot had zero alive incentive | Actually added +1.0 survival bonus in `step()` |
| No observation normalization | PPO trained on raw obs but evaluated with raw obs (no VecNormalize) | Inconsistent observation scales across components | Added VecNormalize wrapper to training, propagated stats to eval and renderer |
| Weak PPO hyperparams | Small batch, high gamma, no LR schedule | Slow convergence, wasted training steps | batch_size=512, n_steps=4096, gamma=0.995, linear LR decay from 3e-4 |

---

### Iteration 0 — Baseline (LLM-generated reward, pre-fix)

- **Script**: `test_go2_iteration.py`
- **Training steps**: 300K
- **Model**: `state/models/test_go2_model.zip`
- **Status**: Pre-ground-plane, pre-PD-control. Only frame-skipping + VecNormalize + survival bonus fixes applied.

**Results**:
| Metric | Value |
|---|---|
| Mean reward | 17.40 |
| Mean ep length | 15.85 steps |
| Forward velocity | 0.10 (per-step component) |
| Success rate | 0% |

**Behavior**: Robot falls within 16 steps (free-fall from 0.27m to 0.15m termination). No meaningful locomotion possible.

**Diagnosis**: `env.data.ncon = 0` after step — no contacts anywhere. Missing ground plane. Also, raw torque control means action=0 maps to ctrl=0, but standing requires ctrl=[0, 0.9, -1.8] per leg.

---

### Iteration 0.5 — Improved Reward, Still Pre-Ground-Plane

- **Training steps**: 1M
- **Model**: `state/models/go2_1m_improved.zip`
- **Reward function**: Hand-crafted with `5.0 * forward_vel`, height bonus, orientation penalty, action energy via `info["action"]`

**Results**:
| Metric | Value |
|---|---|
| Mean reward | 49.24 |
| Mean ep length | 15.7 steps |
| Forward velocity | 0.52 (per-step component) |

**Behavior**: Same free-fall as iteration 0. Training curve rises (4 to 42) but robot still dies at step 16. The improved reward is better shaped but cannot overcome the missing ground plane.

---

### Iteration 1 (v1) — Ground Plane + PD Control

- **Script**: Background task (inline)
- **Training steps**: 1M
- **Model**: `state/models/go2_pd_1m.zip`
- **Fixes applied**: Ground plane in go2.xml, PD position control, lowered height termination to 0.08m

**Reward function**:
```python
forward_reward = 5.0 * data.qvel[0]
lateral_pen = -1.0 * data.qvel[1]**2
orientation = -5.0 * (roll**2 + pitch**2)
height_bonus = 2.0 * exp(-40 * (height - 0.27)**2)
action_pen = -0.01 * sum(info["action"]**2)
ang_vel_pen = -0.5 * sum(data.qvel[3:6]**2)
# + env survival bonus +1.0
```

**Results**:
| Metric | Value |
|---|---|
| Mean reward | 2,312 |
| Mean ep length | 2,000 (full) |
| Forward velocity | 0.095 (per-step) = 0.019 m/s actual |
| Orientation penalty | -1.169/step |
| Height bonus | 1.449/step |
| Training time | 282s (4.7 min) |

**Behavior**: Robot lies flat on its belly/side. Survives full episodes because height ~0.10m > 0.08m threshold. Collects survival + partial height rewards without standing. Training curve unstable (rose to 5400, crashed to 3700, recovered to 6200).

**Lesson**: Height termination at 0.08m is too low — the robot lying flat at ~0.10m doesn't trigger it. Need to raise threshold so lying down = death.

---

### Iteration 2 (v2) — Tighter Termination

- **Script**: `train_go2_v2.py`
- **Training steps**: 1M
- **Model**: `state/models/go2_v2_1m.zip`
- **Changes**: Height termination raised from 0.08m to 0.18m. Orientation limits tightened to roll>0.7 rad, pitch>0.5 rad. Forward velocity kept at 5.0x. Added foot contact bonus (0.5 * n_contacts/4).

**Reward function**:
```python
forward_reward = 5.0 * data.qvel[0]
lateral_pen = -2.0 * data.qvel[1]**2
orientation_pen = -10.0 * (roll**2 + pitch**2)
height_reward = 3.0 * exp(-30 * (height - 0.27)**2)
action_pen = -0.02 * sum(info["action"]**2)
ang_vel_pen = -0.5 * sum(data.qvel[3:6]**2)
foot_bonus = 0.5 * min(n_foot_contacts, 4) / 4.0
# + env survival bonus +1.0
```

**Results**:
| Metric | Value |
|---|---|
| Mean reward | 8,089 |
| Mean ep length | 2,000 (full) |
| Forward velocity | 0.157 (per-step) = 0.031 m/s actual |
| Orientation penalty | -0.101/step |
| Height | 2.636/3.0 |
| Foot contact | 0.498/0.5 |
| Training time | 303s (5.0 min) |

**Behavior**: Robot stands upright on all four feet at correct height. Smooth, monotonically rising training curve (627 to 8087). But barely moves forward (0.031 m/s). Standing rewards (height 2.636 + survival 1.0 + foot 0.498 = 4.13/step) completely dominate velocity reward (0.157/step).

**Lesson**: Standing rewards are too generous relative to velocity. The robot has no incentive to walk when it gets 4.13/step just for standing still vs 0.157/step for moving forward.

---

### Iteration 3 (v3) — Strong Velocity Reward

- **Script**: `train_go2_v3.py`
- **Training steps**: 2M
- **Model**: `state/models/go2_v3_2m.zip`
- **Changes**: Forward velocity weight increased from 5.0 to 15.0. Height reward reduced from 3.0 to 1.0. Removed foot contact bonus.

**Reward function**:
```python
forward_reward = 15.0 * data.qvel[0]
lateral_pen = -2.0 * data.qvel[1]**2
orientation_pen = -10.0 * (roll**2 + pitch**2)
height_reward = 1.0 * exp(-30 * (height - 0.27)**2)
action_pen = -0.01 * sum(info["action"]**2)
ang_vel_pen = -0.3 * sum(data.qvel[3:6]**2)
# + env survival bonus +1.0
```

**Results**:
| Metric | Value |
|---|---|
| Mean reward | 481 |
| Mean ep length | 42.6 steps |
| Forward velocity | 9.77 (per-step) = 0.65 m/s actual |
| Orientation penalty | -0.176/step |
| Training time | 465s (7.8 min) |

**Behavior**: Robot lunges forward aggressively and falls over after ~42 steps. Thumbnail shows mid-fall posture with front legs in the air. Training curve volatile but rising (132 to 12,000 with periodic dips).

**Lesson**: High velocity weight alone creates a "sprint and crash" policy. The robot is incentivized to maximize instantaneous forward speed without maintaining balance. Need to couple velocity reward with stability.

---

### Iteration 4 (v4) — Stability-Gated Velocity (BEST)

- **Script**: `train_go2_v4.py`
- **Training steps**: 2M
- **Model**: `state/models/go2_v4_2m.zip`
- **Changes**: Velocity reward multiplied by stability factor `exp(-8 * (roll^2 + pitch^2))`. Robot only gets velocity credit when upright. Orientation penalty increased to -15.0. Angular velocity penalty increased to -0.8.

**Reward function**:
```python
stability = exp(-8.0 * (roll**2 + pitch**2))
forward_reward = 10.0 * data.qvel[0] * stability
lateral_pen = -2.0 * data.qvel[1]**2
orientation_pen = -15.0 * (roll**2 + pitch**2)
height_reward = 2.0 * exp(-30 * (height - 0.27)**2)
action_pen = -0.02 * sum(info["action"]**2)
ang_vel_pen = -0.8 * sum(data.qvel[3:6]**2)
# + env survival bonus +1.0
```

**Results**:
| Metric | Value |
|---|---|
| Mean reward | 22,978 |
| Mean ep length | 2,000 (full) |
| Forward velocity | 9.33 (per-step) = 0.97 m/s actual |
| Stability factor | 0.963 (nearly perfect) |
| Orientation penalty | -0.071/step |
| Height | 1.995/2.0 |
| Success rate | 100% |
| Std reward | 6.27 (extremely consistent) |
| Training time | ~480s (8 min) |

**Behavior**: Robot walks forward at ~1 m/s with stable upright posture, all four feet on the ground, proper standing height. Consistent across all 10 eval episodes.

**Lesson**: The stability-gated velocity pattern `vel * exp(-k * tilt^2)` is the key insight. It naturally couples speed with balance: the robot can only maximize velocity reward by staying upright. This avoids both the "stand still" trap (no velocity = no reward) and the "sprint and crash" trap (tilting kills the velocity multiplier).

---

## Summary Table

| Version | Steps | Ep Length | Velocity (m/s) | Reward | Behavior |
|---|---|---|---|---|---|
| iter 0 | 300K | 16 | 0 | 17 | Falls immediately (no ground) |
| iter 0.5 | 1M | 16 | 0 | 49 | Falls immediately (no ground) |
| v1 | 1M | 2,000 | 0.02 | 2,312 | Lies flat on belly |
| v2 | 1M | 2,000 | 0.03 | 8,089 | Stands still |
| v3 | 2M | 43 | 0.65 | 481 | Sprints and crashes |
| **v4** | **2M** | **2,000** | **0.97** | **22,978** | **Walks forward stably** |

---

## Key Takeaways for Future Reward Design

1. **Fix the physics first**: No reward function can overcome a missing ground plane or broken control scheme. Verify `data.ncon > 0` and that the robot is stable at zero action before tuning rewards.

2. **Termination conditions are reward shaping**: Loose termination (height < 0.08m) lets the robot exploit degenerate policies. Tight termination (height < 0.18m, orientation limits) forces the robot to stay in valid configurations.

3. **Gate velocity on stability**: `reward = velocity * exp(-k * tilt^2)` naturally couples speed with balance. The robot can only maximize reward by moving forward while upright.

4. **Standing rewards must be small relative to velocity**: If the robot gets 4x more reward for standing than walking, it will stand still. Forward velocity should be the dominant signal (5-10x standing rewards at target speed).

5. **Use normalized actions for energy penalties**: `info["action"]` is in [-1, 1]. `data.ctrl` can be in [-45, +45]. Using ctrl for energy penalties creates a catastrophic penalty that trains the robot to be paralyzed.

6. **2M steps minimum for Go2**: 1M shows the trend but gaits don't converge. 2M gives robust walking policies. Training takes ~8 min on M2 Air.

7. **VecNormalize is non-negotiable**: PPO with unnormalized observations/rewards is extremely brittle. Always use VecNormalize and propagate stats to eval/render.
