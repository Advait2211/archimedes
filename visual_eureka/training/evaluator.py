import logging

import numpy as np
from stable_baselines3 import PPO

from visual_eureka.envs.base_env import EurekaEnv

logger = logging.getLogger(__name__)


def evaluate_policy(
    model_path: str,
    xml_path: str,
    obs_config: dict,
    reward_code: str,
    n_episodes: int = 20,
) -> dict:
    env = EurekaEnv(xml_path, obs_config, reward_code=reward_code)
    model = PPO.load(model_path)

    rewards = []
    lengths = []
    component_sums = {}
    component_count = 0
    early_terminations = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_len = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1
            done = terminated or truncated

            rc = info.get("reward_components", {})
            if rc:
                component_count += 1
                for k, v in rc.items():
                    component_sums[k] = component_sums.get(k, 0.0) + float(v)

        rewards.append(ep_reward)
        lengths.append(ep_len)
        if terminated:
            early_terminations += 1

    env.close()

    component_means = {}
    if component_count > 0:
        component_means = {k: v / component_count for k, v in component_sums.items()}

    success_rate = 1.0 - (early_terminations / n_episodes) if n_episodes > 0 else 0.0

    success_metric = obs_config.get("success_metric")
    if success_metric:
        eval_env = EurekaEnv(xml_path, obs_config, reward_code=reward_code)
        successes = 0
        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
            obs_dict = eval_env._make_obs_dict(obs)
            try:
                ns = {"obs": obs_dict, "np": np}
                if eval(success_metric, {"__builtins__": {}}, ns):
                    successes += 1
            except Exception:
                pass
        eval_env.close()
        success_rate = successes / n_episodes

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_ep_length": float(np.mean(lengths)),
        "reward_component_means": component_means,
        "success_rate": success_rate,
    }
