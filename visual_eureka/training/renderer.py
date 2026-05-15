import logging
import os

import numpy as np
import mujoco
import imageio
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from visual_eureka.envs.base_env import EurekaEnv

logger = logging.getLogger(__name__)


def _single_rollout_frames(env, model, n_steps: int, seed: int) -> list:
    frames = []
    obs, _ = env.reset(seed=seed)
    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        frames.append(None)  # placeholder — filled below
        if terminated or truncated:
            obs, _ = env.reset()
    return frames


def render_eval_gifs(
    xml_path: str,
    obs_config: dict,
    reward_code: str,
    model_path: str,
    base_output_path: str,
    n_steps: int = 1000,
    n_rollouts: int = 3,
    fps: int = 30,
    height: int = 480,
    width: int = 640,
) -> tuple[list[str], str]:
    """Render n_rollouts GIFs and one thumbnail PNG. Returns (gif_paths, thumb_path)."""
    env = EurekaEnv(xml_path, obs_config, reward_code=reward_code)
    model = PPO.load(model_path)
    renderer = mujoco.Renderer(env.model, height=height, width=width)

    vecnorm_path = model_path.replace(".zip", "_vecnorm.pkl")
    obs_normalizer = None
    if os.path.exists(vecnorm_path):
        dummy_vec = DummyVecEnv([lambda: Monitor(EurekaEnv(xml_path, obs_config, reward_code=reward_code))])
        obs_normalizer = VecNormalize.load(vecnorm_path, dummy_vec)
        obs_normalizer.training = False
        logger.info("Loaded VecNormalize stats for rendering")

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    base_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    cam.trackbodyid = base_id if base_id >= 0 else 1
    cam.distance = 2.0
    cam.elevation = -20.0
    cam.azimuth = 135.0

    os.makedirs(os.path.dirname(base_output_path) if os.path.dirname(base_output_path) else ".", exist_ok=True)

    stem, ext = os.path.splitext(base_output_path)
    gif_paths = []

    for r in range(n_rollouts):
        frames = []
        obs, _ = env.reset(seed=r * 42)
        for _ in range(n_steps):
            pred_obs = obs
            if obs_normalizer is not None:
                pred_obs = obs_normalizer.normalize_obs(obs)
            action, _ = model.predict(pred_obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            renderer.update_scene(env.data, camera=cam)
            frames.append(renderer.render().copy())
            if terminated or truncated:
                obs, _ = env.reset(seed=r * 42 + 1)

        out_path = f"{stem}_{r}{ext}"
        imageio.mimsave(out_path, frames, fps=fps, loop=0)
        logger.info(f"Saved eval GIF {r+1}/{n_rollouts} to {out_path} ({len(frames)} frames)")
        gif_paths.append(out_path)

    renderer.close()
    env.close()
    if obs_normalizer is not None:
        obs_normalizer.close()

    # Thumbnail from the middle frame of the middle rollout
    mid_gif_path = gif_paths[len(gif_paths) // 2]
    mid_frames = imageio.mimread(mid_gif_path, memtest=False)
    thumb_frame = mid_frames[len(mid_frames) // 2]
    thumb = Image.fromarray(thumb_frame).resize((320, 240), Image.LANCZOS)
    thumb_path = f"{stem}_thumb.png"
    thumb.save(thumb_path)
    logger.info(f"Saved thumbnail to {thumb_path}")

    return gif_paths, thumb_path


def render_eval_gif(
    xml_path: str,
    obs_config: dict,
    reward_code: str,
    model_path: str,
    output_path: str,
    n_steps: int = 1000,
    fps: int = 30,
    height: int = 480,
    width: int = 640,
) -> str:
    gif_paths, _ = render_eval_gifs(
        xml_path=xml_path,
        obs_config=obs_config,
        reward_code=reward_code,
        model_path=model_path,
        base_output_path=output_path,
        n_steps=n_steps,
        n_rollouts=1,
        fps=fps,
        height=height,
        width=width,
    )
    return gif_paths[0]
