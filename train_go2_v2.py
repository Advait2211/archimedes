"""Quick 1M-step Go2 training with tighter termination + improved reward."""
import os, sys, json, time, yaml, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visual_eureka.training.trainer import run_full_training
from visual_eureka.training.renderer import render_eval_gifs

XML_PATH = "go2.xml"
OBS_CONFIG_PATH = "go2_obs_config.yaml"

REWARD_CODE = '''
import numpy as np
import math

def compute_reward(obs, info, data, model):
    forward_vel = data.qvel[0]
    forward_reward = 5.0 * forward_vel

    lateral_pen = -2.0 * data.qvel[1]**2

    w, x, y, z = obs["base_quat"]
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    pitch = np.arctan2(2*(w*y - z*x), 1 - 2*(y**2 + z**2))
    orientation_pen = -10.0 * (roll**2 + pitch**2)

    height = data.qpos[2]
    height_reward = 3.0 * np.exp(-30.0 * (height - 0.27)**2)

    action = info["action"]
    action_pen = -0.02 * np.sum(action**2)

    ang_vel_pen = -0.5 * np.sum(data.qvel[3:6]**2)

    n_foot_contacts = 0
    for i in range(data.ncon):
        g1 = model.geom(data.contact[i].geom1).name or ""
        g2 = model.geom(data.contact[i].geom2).name or ""
        if "FL" in g1 or "FR" in g1 or "RL" in g1 or "RR" in g1:
            n_foot_contacts += 1
        if "FL" in g2 or "FR" in g2 or "RL" in g2 or "RR" in g2:
            n_foot_contacts += 1
    foot_bonus = 0.5 * min(n_foot_contacts, 4) / 4.0

    total = forward_reward + lateral_pen + orientation_pen + height_reward + action_pen + ang_vel_pen + foot_bonus

    return float(total), {
        "forward_vel": float(forward_reward),
        "lateral_pen": float(lateral_pen),
        "orientation": float(orientation_pen),
        "height": float(height_reward),
        "action_pen": float(action_pen),
        "ang_vel_pen": float(ang_vel_pen),
        "foot_contact": float(foot_bonus),
    }
'''

def main():
    with open(OBS_CONFIG_PATH) as f:
        obs_config = yaml.safe_load(f)

    save_path = "state/models/go2_v2_1m.zip"
    print("Training 1M steps with tighter termination...")
    t0 = time.time()

    stats = run_full_training(
        xml_path=XML_PATH,
        obs_config=obs_config,
        reward_code=REWARD_CODE,
        full_steps=1_000_000,
        n_envs=4,
        save_path=save_path,
    )

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Stats: {json.dumps(stats, indent=2, default=str)}")

    gif_paths, thumb_path = render_eval_gifs(
        xml_path=XML_PATH,
        obs_config=obs_config,
        reward_code=REWARD_CODE,
        model_path=save_path,
        base_output_path="state/models/go2_v2_eval.gif",
        n_steps=500,
        n_rollouts=3,
    )
    print(f"GIFs: {gif_paths}")
    print(f"Thumb: {thumb_path}")

if __name__ == "__main__":
    main()
