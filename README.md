# Visual Eureka

An evolutionary reward design system for MuJoCo locomotion RL, inspired by the Eureka paper (arXiv 2310.12931). Visual Eureka iteratively generates reward function candidates using an LLM, trains them with PPO via Stable Baselines3, evaluates the resulting policies through rendered rollout GIFs, and reflects on performance using both a vision-language model and statistical analysis. A Streamlit GUI provides experiment management, visualization, and human-in-the-loop feedback.

## Setup

```bash
pip install -r visual_eureka/requirements.txt
```

Create a `.env` file at the project root with your NVIDIA NIM API key:

```
NVIDIA_API_KEY=nvapi-your-key-here
```

## Running

```bash
cd /path/to/archimedes
caffeinate -i streamlit run visual_eureka/app.py
```

`caffeinate -i` prevents macOS idle sleep during long overnight training runs.

## Using Your Robot

Upload your MuJoCo XML and `obs_config.yaml` on the Setup page.

**Important:** if your XML references mesh files via a relative `meshdir`, keep the XML at the project root and the meshes in the directory specified by `meshdir`. The uploader saves your XML to `uploaded_robot.xml` at the project root, preserving relative mesh paths.

## Observation Config

Each robot needs an `obs_config.yaml` describing its observation space and task. See `visual_eureka/config/examples/` for templates.

```yaml
task_description: "Quadruped forward locomotion at maximum speed"
robot_type: quadruped          # quadruped | biped
obs_vector_dim: 34
obs_components:
  joint_pos:
    start: 0
    end: 12
    description: "joint positions in radians"
  joint_vel:
    start: 12
    end: 24
    description: "joint velocities in rad/s"
  base_quat:
    start: 24
    end: 28
    description: "base body orientation quaternion (w, x, y, z)"
  base_lin_vel:
    start: 28
    end: 31
    description: "base linear velocity xyz in m/s"
  base_ang_vel:
    start: 31
    end: 34
    description: "base angular velocity xyz in rad/s"
success_metric: "obs['base_lin_vel'][0] > 0.8"   # optional Python expression
episode_length: 1000
```

## Reward Function Signature

The LLM generates functions with this exact signature:

```python
def compute_reward(obs: dict, info: dict, data, model) -> tuple[float, dict]:
    # obs: named slices from obs_config (e.g. obs["joint_pos"], obs["base_lin_vel"])
    # info: gymnasium info dict
    # data: mujoco.MjData
    # model: mujoco.MjModel
    return total_reward, {"forward_vel": fv_reward, "stability": stab_reward}
```

Available in scope: `numpy as np`, `math`, `mujoco`.

Valid `MjData` attributes:

| Attribute | Shape | Description |
|---|---|---|
| `data.qpos` | `(nq,)` | Joint positions; `[0:3]` = base xyz, `[3:7]` = base quat, `[7:]` = joint angles |
| `data.qvel` | `(nv,)` | Joint velocities; `[0:3]` = base lin vel, `[3:6]` = base ang vel, `[6:]` = joint vel |
| `data.ctrl` | `(nu,)` | Actuator control signals |
| `data.actuator_force` | `(nu,)` | Actuator forces |
| `data.cfrc_ext` | `(nbody, 6)` | External forces on each body |
| `data.xpos` | `(nbody, 3)` | Body positions in world frame |
| `data.xquat` | `(nbody, 4)` | Body orientations in world frame |

All arrays are indexed with integers or slices only — never strings. For named lookups: `data.xpos[model.body("trunk").id]`.

## Iteration Loop

1. **Generate** — LLM produces K distinct reward function candidates informed by previous critiques
2. **Filter** — Each candidate trains with PPO for a short phase; best is selected by mean reward
3. **Train** — Best candidate trains for a full run; model is saved
4. **Evaluate** — Policy runs 3 deterministic rollouts; GIFs and a thumbnail are saved
5. **Reflect** — VLM analyzes behavior from the middle rollout GIF; text model analyzes training statistics
6. **Iterate** — Critiques feed back into the next generation of reward candidates

## Modes

**Manual:** Review GIF and critiques after each iteration, optionally edit them, click "Generate Next Iteration".

**Auto-run:** Set the number of iterations on the Setup page (before starting) or the Current Iteration page (to continue from the current result). The loop runs unattended using model-generated critiques as feedback.

## Hyperparameter Overrides

The Current Iteration page has a collapsed "Override training config for next iteration" section. K (candidates), full training steps, and eval steps can be changed mid-experiment. Filter steps are fixed for the experiment lifetime to keep candidate comparisons consistent.

## Recommended Settings for Go2 Overnight Runs

| Parameter | Value |
|---|---|
| Filter steps | 50,000 |
| Full training steps | 1,000,000 – 1,500,000 |
| Candidates (K) | 3–4 |
| Total iterations | 5 |
| Estimated time (M2 Air) | 8–10 hours |

## Reward Error Logging

If a reward function throws an exception during a training step, the error is logged once (to console and to `state/reward_errors.log`) and silently suppressed for all subsequent steps. This prevents log spam when a broken reward function errors on every step of an episode. The log file is never cleared automatically — check it to diagnose LLM-generated reward functions that silently zeroed out the reward signal.

## Recovery After Page Refresh

Session state is lost on browser refresh but all data persists in SQLite (`state/eureka_runs.db`) and on disk. After refreshing:

1. Go to **History** — loads runs from DB automatically
2. Select your run
3. Go to **Setup** — re-upload XML and obs_config (restores session state; do not click Start)
4. Go to **Current Iteration** — auto-run and generate next iteration will work normally

## Testing the Loop

To test the iteration loop without waiting through a full training run:

```bash
source /path/to/venv/bin/activate
python seed_iteration.py
```

This trains for 10K steps, seeds the DB with a completed iteration and pre-written critiques, and prints the run ID to select in History.
