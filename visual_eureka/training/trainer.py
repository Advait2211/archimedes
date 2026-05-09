import logging
import os
import platform

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from visual_eureka.envs.base_env import EurekaEnv

logger = logging.getLogger(__name__)


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_env(xml_path: str, obs_config: dict, reward_code: str, seed: int):
    def _init():
        env = EurekaEnv(xml_path, obs_config, reward_code=reward_code)
        env.reset(seed=seed)
        env = Monitor(env)
        return env
    return _init


def _create_vec_env(xml_path, obs_config, reward_code, n_envs, base_seed=0):
    env_fns = [_make_env(xml_path, obs_config, reward_code, seed=base_seed + j) for j in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(env_fns)
    try:
        start_method = "fork" if platform.system() == "Darwin" else "forkserver"
        return SubprocVecEnv(env_fns, start_method=start_method)
    except Exception as e:
        logger.warning(f"SubprocVecEnv failed ({e}), falling back to DummyVecEnv")
        return DummyVecEnv(env_fns)


class _TrainingCurveCallback(BaseCallback):
    def __init__(self, target_points: int = 50):
        super().__init__()
        self.target_points = target_points
        self.curve = []
        self._episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])
        return True

    def _on_rollout_end(self) -> None:
        if self._episode_rewards:
            mean_r = float(np.mean(self._episode_rewards))
            self.curve.append((self.num_timesteps, mean_r))
            self._episode_rewards = []


class _RewardComponentCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.component_sums = {}
        self.component_counts = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            rc = info.get("reward_components", {})
            if rc:
                self.component_counts += 1
                for key, val in rc.items():
                    if key not in self.component_sums:
                        self.component_sums[key] = 0.0
                    self.component_sums[key] += float(val)
        return True

    def get_means(self) -> dict:
        if self.component_counts == 0:
            return {}
        return {k: v / self.component_counts for k, v in self.component_sums.items()}


class _StatusCallback(BaseCallback):
    def __init__(self, status_dict: dict, phase: str, total_steps: int, update_every: int = 2048):
        super().__init__()
        self._status = status_dict
        self._phase = phase
        self._total = total_steps
        self._update_every = update_every
        self._last_update = -update_every

    def _on_step(self) -> bool:
        self._status["step"] = self.num_timesteps
        if self.num_timesteps - self._last_update >= self._update_every:
            self._last_update = self.num_timesteps
            pct = min(self.num_timesteps / self._total, 1.0) if self._total > 0 else 0.0
            label = f"{self._phase} — {self.num_timesteps:,} / {self._total:,} steps"
            phase_ph = self._status.get("_phase_ph")
            sidebar_ph = self._status.get("_sidebar_ph")
            if phase_ph is not None:
                phase_ph.info(label)
            if sidebar_ph is not None:
                sidebar_ph.progress(pct, text=f"{self.num_timesteps:,} / {self._total:,} steps")
        return True


def _evaluate_after_training(
    model: PPO,
    xml_path: str,
    obs_config: dict,
    reward_code: str,
    n_episodes: int = 10,
) -> dict:
    env = EurekaEnv(xml_path, obs_config, reward_code=reward_code)
    rewards = []
    lengths = []
    component_sums = {}
    component_count = 0

    for _ in range(n_episodes):
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

    env.close()

    component_means = {}
    if component_count > 0:
        component_means = {k: v / component_count for k, v in component_sums.items()}

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_ep_length": float(np.mean(lengths)),
        "reward_component_means": component_means,
    }


def run_filter_phase(
    xml_path: str,
    obs_config: dict,
    candidate_codes: list[str],
    filter_steps: int = 100_000,
    n_envs: int = 4,
    status_dict: dict = None,
) -> tuple[str, dict, list[dict]]:
    device = _get_device()
    all_stats = []
    best_score = -np.inf
    best_idx = 0

    for i, code in enumerate(candidate_codes):
        logger.info(f"Filter phase: candidate {i+1}/{len(candidate_codes)}")
        stats = {
            "candidate_index": i,
            "mean_reward": -np.inf,
            "std_reward": 0.0,
            "mean_ep_length": 0.0,
            "reward_component_means": {},
            "error": None,
        }

        try:
            test_env = EurekaEnv(xml_path, obs_config, reward_code=code)
            test_env.close()
        except Exception as e:
            logger.warning(f"Candidate {i} failed reward injection: {e}")
            stats["error"] = str(e)
            all_stats.append(stats)
            continue

        try:
            vec_env = _create_vec_env(xml_path, obs_config, code, n_envs, base_seed=i * 100)

            comp_cb = _RewardComponentCallback()
            phase_label = f"Filter: candidate {i+1}/{len(candidate_codes)}"

            if status_dict is not None:
                status_dict["phase"] = phase_label
                status_dict["step"] = 0
                status_dict["total_steps"] = filter_steps
                phase_ph = status_dict.get("_phase_ph")
                if phase_ph is not None:
                    phase_ph.info(f"{phase_label} — 0 / {filter_steps:,} steps")

            callbacks = [comp_cb]
            if status_dict is not None:
                callbacks.append(_StatusCallback(status_dict, phase_label, filter_steps))

            model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                device=device,
                verbose=0,
            )

            model.learn(total_timesteps=filter_steps, callback=callbacks)

            eval_stats = _evaluate_after_training(
                model, xml_path, obs_config, code, n_episodes=10
            )

            stats["mean_reward"] = eval_stats["mean_reward"]
            stats["std_reward"] = eval_stats["std_reward"]
            stats["mean_ep_length"] = eval_stats["mean_ep_length"]
            stats["reward_component_means"] = eval_stats["reward_component_means"]

            vec_env.close()
            del model

        except Exception as e:
            logger.warning(f"Candidate {i} training failed: {e}")
            stats["error"] = str(e)
            try:
                vec_env.close()
            except Exception:
                pass

        all_stats.append(stats)

        if stats["mean_reward"] > best_score:
            best_score = stats["mean_reward"]
            best_idx = i

    best_code = candidate_codes[best_idx]
    best_stats = all_stats[best_idx]

    logger.info(
        f"Filter phase complete. Best candidate: {best_idx} "
        f"(mean_reward={best_stats['mean_reward']:.4f})"
    )

    return best_code, best_stats, all_stats


def run_full_training(
    xml_path: str,
    obs_config: dict,
    reward_code: str,
    full_steps: int = 500_000,
    n_envs: int = 4,
    save_path: str = None,
    status_dict: dict = None,
) -> dict:
    device = _get_device()

    vec_env = _create_vec_env(xml_path, obs_config, reward_code, n_envs)

    curve_cb = _TrainingCurveCallback(target_points=50)
    comp_cb = _RewardComponentCallback()

    phase_label = f"Full training"
    if status_dict is not None:
        status_dict["phase"] = phase_label
        status_dict["step"] = 0
        status_dict["total_steps"] = full_steps
        phase_ph = status_dict.get("_phase_ph")
        if phase_ph is not None:
            phase_ph.info(f"{phase_label} — 0 / {full_steps:,} steps")

    callbacks = [curve_cb, comp_cb]
    if status_dict is not None:
        callbacks.append(_StatusCallback(status_dict, phase_label, full_steps))

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=device,
        verbose=0,
    )

    model.learn(total_timesteps=full_steps, callback=callbacks)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)

    eval_stats = _evaluate_after_training(
        model, xml_path, obs_config, reward_code, n_episodes=20
    )

    success_rate = None
    success_metric = obs_config.get("success_metric")
    if success_metric:
        env = EurekaEnv(xml_path, obs_config, reward_code=reward_code)
        successes = 0
        n_eval = 20
        for _ in range(n_eval):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            obs_dict = env._make_obs_dict(obs)
            try:
                ns = {"obs": obs_dict, "np": np}
                if eval(success_metric, {"__builtins__": {}}, ns):
                    successes += 1
            except Exception:
                pass
        env.close()
        success_rate = successes / n_eval

    training_curve = curve_cb.curve
    if len(training_curve) > 50:
        step = len(training_curve) // 50
        training_curve = training_curve[::step][:50]

    vec_env.close()

    return {
        "mean_reward": eval_stats["mean_reward"],
        "std_reward": eval_stats["std_reward"],
        "mean_ep_length": eval_stats["mean_ep_length"],
        "reward_component_means": eval_stats["reward_component_means"],
        "success_rate": success_rate,
        "training_curve": training_curve,
        "model_path": save_path,
    }
