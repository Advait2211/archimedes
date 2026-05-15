# Visual Eureka

Evolutionary reward design system for MuJoCo locomotion RL, inspired by the Eureka paper (arXiv 2310.12931). Iteratively generates reward function candidates via LLM, trains with PPO (SB3), evaluates via rendered GIF rollouts, reflects using VLM + statistical analysis, and loops. Streamlit GUI for experiment management.

## Stack

| Component | Package |
|---|---|
| Simulation | `mujoco` (native Python, >=3.0.0) + `gymnasium` |
| RL | `stable-baselines3`, PPO, `SubprocVecEnv`/`DummyVecEnv` |
| LLM/VLM | NVIDIA NIM via `integrate.api.nvidia.com` (OpenAI-compatible client) |
| GUI | Streamlit (dark theme, no emojis anywhere) |
| Database | SQLite via `sqlite3` (stdlib) |
| Rendering | `mujoco.Renderer` offscreen + `imageio` for GIF, `PIL` for thumbnails |
| Language | Python 3.10+ (tested on 3.11) |

## Project Structure

```
visual_eureka/
  app.py                          # Streamlit GUI — 3 pages (Setup, Current Iteration, History)
  requirements.txt
  README.md
  config/
    env_schema.yaml               # Documented obs_config schema
    examples/
      quadruped_obs.yaml          # 12-DOF quadruped example
      biped_obs.yaml              # 17-DOF biped example
  envs/
    __init__.py                   # re-exports EurekaEnv
    base_env.py                   # EurekaEnv(gymnasium.Env) — injectable reward via exec()
    uploads/                      # Built-in test assets only
      test_quadruped.xml          # Mesh-free test robot (12 DOF, 4 legs)
      test_obs_config.yaml        # Matching obs config for test robot
  llm/
    __init__.py                   # re-exports NIMClient, generate_candidates, critiques
    client.py                     # NIMClient — text (meta/llama-3.3-70b-instruct) + vision (meta/llama-3.2-90b-vision-instruct)
    reward_generator.py           # generate_candidates(), extract_xml_summary()
    visual_reflection.py          # generate_visual_critique() — single middle frame, 1 image limit on NIM
    text_reflection.py            # generate_text_critique() — stats-based analysis
  training/
    __init__.py
    trainer.py                    # run_filter_phase(), run_full_training() — accept status_dict for live progress
    evaluator.py                  # evaluate_policy() — deterministic rollout stats
    renderer.py                   # render_eval_gifs() — 3 rollouts + thumbnail; render_eval_gif() wraps it
  state/
    __init__.py
    iteration_store.py            # IterationStore — SQLite CRUD for experiment iterations
    models/                       # Saved SB3 .zip, eval GIFs, thumbnails per iteration
assets/                           # Go2 mesh .obj files (copied from eklavya project)
go2.xml                           # Unitree Go2 MuJoCo XML (meshdir="assets", loads fine)
go2_obs_config.yaml               # obs config for Go2 (34-dim: joint_pos/vel, quat, lin/ang vel)
go2_env.py                        # Reference implementation only — not used by visual_eureka
seed_iteration.py                 # Seeds DB with a quick iteration for testing the loop
test_go2_iteration.py             # Headless full pipeline test: env check → LLM candidates → filter → train → render → critique
train_go2_v2.py                   # Manual training script: tighter termination (v2)
train_go2_v3.py                   # Manual training script: strong velocity reward (v3)
train_go2_v4.py                   # Manual training script: stability-gated velocity (v4, best)
uploaded_robot.xml                # Where the Streamlit uploader saves the user's XML
ITERATION_LOG.md                  # Tracks all manual training iterations, reward functions, fixes, and results
```

## Key Architecture Decisions

### Reward Injection
`EurekaEnv.set_reward_fn(code_str)` uses `exec()` with an isolated namespace `{"np": numpy, "math": math, "mujoco": mujoco}`. Never `exec(code, globals())`. The reward function signature the LLM must produce:
```python
def compute_reward(obs, info, data, model) -> tuple[float, dict]:
    # obs: dict of named observation slices (from obs_config)
    # info: {"action": np.ndarray} — normalized action in [-1, 1]
    # data: mujoco.MjData
    # model: mujoco.MjModel
    return total_reward, {"component_name": float_value, ...}
```

### Frame-Skipping
`EurekaEnv` calls `mj_step` multiple times per action (n_substeps=5 for quadruped, 4 for biped). Effective dt = model.opt.timestep * n_substeps (0.01s for Go2). Without this, episodes are too short for meaningful locomotion.

### PD Position Control
`EurekaEnv` uses PD position control when a "home" keyframe is present in the XML. The policy outputs target joint position offsets from the home pose, and the env converts these to torques: `torque = kp * (target - current_pos) - kd * current_vel`. Default gains: kp=20, kd=0.5, action_scale=0.5 rad. This is critical for torque-actuated robots (like Go2) — without it, the RL policy must discover gravity-compensating torques from scratch, which is intractable.

### Ground Plane
Robot XMLs MUST include a ground plane geom in the worldbody for locomotion. The Go2 XML includes one. Without a floor, the robot falls through empty space regardless of torque control. This was a critical bug — the original Go2 XML had no floor, causing free-fall regardless of what the policy did.

### Termination Conditions
`EurekaEnv.step()` terminates episodes when:
- Height < 0.18m or > 1.0m (prevents lying-flat exploitation)
- Roll > 0.7 rad (~40 deg) or pitch > 0.5 rad (~29 deg)

These thresholds were tuned empirically. Too loose (e.g. height < 0.08m) allows the robot to collect survival rewards while lying flat. Too tight kills exploration. The Go2 at zero action stabilizes at ~0.196m, giving ~0.016m margin above the 0.18m floor.

### Survival Bonus
The environment adds +1.0 survival bonus per step automatically unless the reward function returns a "survival" key in its components dict. This keeps robots alive long enough to learn.

### SubprocVecEnv Compatibility
Reward code is passed as a string through the `make_env` factory to the EurekaEnv constructor. Each subprocess calls `set_reward_fn()` in its own `__init__`. On macOS, uses `fork` start method; on Linux, `forkserver`. Falls back to `DummyVecEnv` on failure. Always uses `DummyVecEnv` when `n_envs=1`.

### XML Upload and Mesh Resolution
Uploaded XMLs are saved to `uploaded_robot.xml` at the project root (not `envs/uploads/`). This ensures `meshdir="assets"` in Go2's XML resolves correctly to `assets/` at the project root. Never save user XMLs to subdirectories — relative mesh paths will break.

### obs_config.yaml Format
```yaml
task_description: "string"
robot_type: quadruped | biped
obs_vector_dim: int
obs_components:
  component_name:
    start: int    # inclusive index into flat obs vector
    end: int      # exclusive
    description: "string"
success_metric: null | "Python expression"
episode_length: int
```
Validation in `envs/base_env.py:validate_obs_config()` raises `ValueError` with specific field names.

### Observation Construction
`_get_obs()` in `base_env.py` maps obs_config components to MuJoCo data by matching keywords in the description field: "joint"+"pos" -> `qpos[7:]`, "joint"+"vel" -> `qvel[6:]`, "quat" -> `qpos[3:7]`, "lin"+"vel" -> `qvel[0:3]`, "ang"+"vel" -> `qvel[3:6]`, etc. This is heuristic-based — if a new component type is needed, update `_get_obs()`.

### LLM Models
- Text (reward gen + stats reflection): `meta/llama-3.3-70b-instruct`
- Vision (visual reflection): `meta/llama-3.2-90b-vision-instruct` — accepts at most 1 image per request. Always send exactly 1 frame (middle frame of middle rollout GIF).
- API key: loaded from `.env` at project root via `python-dotenv` in `llm/client.py`. Never hardcoded. No UI field for the key.

### Reward Function Error Logging
When a reward function throws an exception during `step()`, the error is deduplicated per `EurekaEnv` instance — only the first occurrence of each unique error is logged (to console via `logger.warning` and appended to `state/reward_errors.log`). Subsequent identical errors are silently swallowed (reward = 0.0). This prevents log spam when a broken reward function errors on every step. The log file accumulates across runs and is never cleared automatically.

### Reward Design Principles
The most important lesson from Go2 tuning: **gate velocity reward on stability**. The pattern `forward_reward = weight * velocity * exp(-k * (roll^2 + pitch^2))` ensures the robot only gets velocity credit when upright. Without this gating:
- High velocity weight alone → robot lunges forward and falls (fast but short episodes)
- High standing rewards alone → robot stands still (long episodes but no locomotion)
- Stability-gated velocity → robot learns smooth forward gait (fast AND stable)

Typical working reward magnitudes for Go2 (per step):
- Forward velocity: 10.0 * qvel[0] * stability_gate (dominant signal)
- Height maintenance: 2.0 * exp(-30 * (h - 0.27)^2)
- Orientation penalty: -15.0 * (roll^2 + pitch^2)
- Action energy: -0.02 * sum(info["action"]^2)
- Angular velocity: -0.8 * sum(qvel[3:6]^2)
- Lateral drift: -2.0 * qvel[1]^2
- Survival bonus: +1.0 (auto from env)

See `ITERATION_LOG.md` for full history of reward tuning iterations and what each change produced.

### Training Defaults
- Filter phase: 100K steps, 4 envs (recommended 80K for Go2)
- Full training: 500K steps, 4 envs (recommended 2M for Go2)
- PPO: lr=3e-4 (linear decay), n_steps=4096, batch_size=512, n_epochs=10, gamma=0.995, gae_lambda=0.95, MlpPolicy
- VecNormalize: obs normalization (clip=10) + reward normalization (clip=10) during training; eval uses obs normalization only
- VecNormalize stats saved alongside model as `*_vecnorm.pkl`; renderer and evaluator load them automatically
- Device: auto-detect (CUDA if available, CPU fallback)
- Failed candidates: score = -inf, logged, continue to next
- `_StatusCallback` in trainer.py updates `status_dict` every 2048 steps for live sidebar progress
- Action energy penalty in reward functions must use `info["action"]` (normalized [-1,1]), NOT `data.ctrl` (robot-specific magnitudes)

### Rendering
`render_eval_gifs()` in `renderer.py` renders `n_rollouts=3` GIFs per iteration with different seeds, saves as `eval_{run_id}_{iteration}_0.gif`, `_1.gif`, `_2.gif`. Also saves a 320×240 thumbnail PNG (`_thumb.png`) from the middle frame of the middle rollout. Visual critique uses the middle GIF. History page shows thumbnails for fast loading.

### Database
SQLite at `state/eureka_runs.db`. Table `iterations` with columns: id, run_id, iteration, reward_code, candidate_codes (JSON), filter_stats (JSON), training_stats (JSON), eval_gif_path (primary/middle GIF), eval_gif_paths (JSON list of all 3), thumb_path, visual_critique, text_critique, user_critique, mean_reward, success_rate, model_path, timestamp, notes.

Migration: `ALTER TABLE` in `_create_tables()` adds `eval_gif_paths` and `thumb_path` to existing DBs — safe to run on old schema.

### UI Freeze During Training
When training starts, `FREEZE_CSS` injects `pointer-events: none` on `.stApp *` to prevent any clicks from interrupting the run. `st.empty()` placeholders update in real time via websocket even while the main thread is blocked. Sidebar shows phase label + step progress bar.

### Auto-run Loop
`_run_auto_loop()` in `app.py` runs N iterations back-to-back using model-generated critiques as reflection input — no human input between iterations. Available on both Setup page (set N before starting) and Current Iteration page (continue from current iteration). Stops cleanly on error.

### Config Overrides
Current Iteration page has a collapsed "Override training config for next iteration" expander. Allows changing K (candidates), full training steps, and eval steps mid-experiment. Filter steps are intentionally excluded — changing them mid-experiment makes candidate comparison inconsistent.

## Running

```bash
cd /Users/advaitdesai/Programming/archimedes
caffeinate -i streamlit run visual_eureka/app.py
```

API key is read from `.env` at project root — no need to export manually.

## Go2 Setup

Meshes are at `assets/` (copied from `eklavya/9_PPO_go2/optimised_ppo/unitree_go2/assets/`). `go2.xml` loads fine. Upload `go2.xml` + `go2_obs_config.yaml` on the Setup page.

Recommended hyperparameters for overnight Go2 runs:
- Filter steps: 50K–80K
- Full training: 2M (1M is borderline — gait may not fully converge)
- Candidates (K): 3
- Total iterations: 5 (approx 8–10 hours on M2 Air)

Best known Go2 model: `state/models/go2_v4_2m.zip` — 0.97 m/s forward, 100% success, 2000-step episodes.

## Testing the Loop Without Full Training

```bash
source /Users/advaitdesai/Programming/eklavya/venv/bin/activate
python seed_iteration.py
```

Seeds the DB with a 10K-step trained model and pre-written critiques. Then: Setup page → upload test XML + obs_config (don't click Start) → History → select the seeded run → Current Iteration → Start Auto-run.

## Iteration Tracking

All manual training iterations, reward function experiments, and environment fixes are logged in `ITERATION_LOG.md` at the project root. Update this file after every training run with: what changed, the reward function, training config, results table, observed behavior, and lessons learned. This avoids repeating failed approaches across sessions.

## Hard Rules

- No emojis anywhere — UI labels, buttons, headers, comments, strings, log messages
- No TODOs or placeholder comments
- `exec()` always uses isolated namespace
- No Isaac Gym, Isaac Lab, or NVIDIA-specific sim dependencies
- NVIDIA NIM API only — no other LLM/VLM providers
- Every LLM call wrapped in try/except — errors surface in GUI, not silent crashes
- obs_config validation fails loudly with specific field name
- Vision model: always exactly 1 image per request (NIM hard limit)
- Uploaded XMLs always saved to project root to preserve relative mesh paths
