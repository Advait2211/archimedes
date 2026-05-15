"""Go2 v3: strong velocity incentive, reduced standing rewards, 2M steps."""
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
    forward_reward = 15.0 * forward_vel

    lateral_pen = -2.0 * data.qvel[1]**2

    w, x, y, z = obs["base_quat"]
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    pitch = np.arctan2(2*(w*y - z*x), 1 - 2*(y**2 + z**2))
    orientation_pen = -10.0 * (roll**2 + pitch**2)

    height = data.qpos[2]
    height_reward = 1.0 * np.exp(-30.0 * (height - 0.27)**2)

    action = info["action"]
    action_pen = -0.01 * np.sum(action**2)

    ang_vel_pen = -0.3 * np.sum(data.qvel[3:6]**2)

    total = forward_reward + lateral_pen + orientation_pen + height_reward + action_pen + ang_vel_pen

    return float(total), {
        "forward_vel": float(forward_reward),
        "lateral_pen": float(lateral_pen),
        "orientation": float(orientation_pen),
        "height": float(height_reward),
        "action_pen": float(action_pen),
        "ang_vel_pen": float(ang_vel_pen),
    }
'''

def main():
    with open(OBS_CONFIG_PATH) as f:
        obs_config = yaml.safe_load(f)

    save_path = "state/models/go2_v3_2m.zip"
    print("Training 2M steps — strong velocity reward...")
    t0 = time.time()

    stats = run_full_training(
        xml_path=XML_PATH,
        obs_config=obs_config,
        reward_code=REWARD_CODE,
        full_steps=2_000_000,
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
        base_output_path="state/models/go2_v3_eval.gif",
        n_steps=500,
        n_rollouts=3,
    )
    print(f"GIFs: {gif_paths}")
    print(f"Thumb: {thumb_path}")

if __name__ == "__main__":
    main()
