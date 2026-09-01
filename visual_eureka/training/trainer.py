import logging
import os
import platform

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from visual_eureka.envs.base_env import EurekaEnv

logger = logging.getLogger(__name__)


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_ppo_batch_size(n_steps: int, n_envs: int) -> int:
    """Pick batch size based on available hardware. On GPU use larger batches."""
    rollout_size = n_steps * n_envs
    if torch.cuda.is_available():
        batch = 1024
    else:
        batch = 256
    # batch must divide rollout_size evenly
    while rollout_size % batch != 0 and batch > 64:
        batch //= 2
    return batch


def _make_env(xml_path: str, obs_config: dict, reward_code: str, seed: int):
    def _init():
        env = EurekaEnv(xml_path, obs_config, reward_code=reward_code)
        env.reset(seed=seed)
        env = Monitor(env)
        return env
    return _init


def _create_vec_env(xml_path, obs_config, reward_code, n_envs, base_seed=0, normalize=True):
    env_fns = [_make_env(xml_path, obs_config, reward_code, seed=base_seed + j) for j in range(n_envs)]
    if n_envs == 1:
        vec_env = DummyVecEnv(env_fns)
    else:
        try:
            start_method = "fork" if platform.system() == "Darwin" else "forkserver"
            vec_env = SubprocVecEnv(env_fns, start_method=start_method)
        except Exception as e:
            logger.warning(f"SubprocVecEnv failed ({e}), falling back to DummyVecEnv")
            vec_env = DummyVecEnv(env_fns)
    if normalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    return vec_env


def _linear_schedule(initial_value: float):
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return schedule


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


class _CheckpointCallback(BaseCallback):
    """Saves model + vecnorm every checkpoint_freq steps during full training."""
    def __init__(self, save_path: str, checkpoint_freq: int = 100_000):
        super().__init__()
        self._save_path = save_path
        self._freq = checkpoint_freq
        self._last_save = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save >= self._freq:
            self._last_save = self.num_timesteps
            ckpt_path = self._save_path.replace(".zip", f"_ckpt{self.num_timesteps}")
            self.model.save(ckpt_path)
            vn = self.training_env
            if isinstance(vn, VecNormalize):
                vn.save(ckpt_path + "_vecnorm.pkl")
            logger.info(f"Checkpoint saved at {self.num_timesteps} steps: {ckpt_path}.zip")
        return True


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
    vec_normalize: VecNormalize = None,
) -> dict:
    eval_vec = _create_vec_env(xml_path, obs_config, reward_code, n_envs=1, normalize=False)
    if vec_normalize is not None:
        eval_vec = VecNormalize(eval_vec, norm_obs=True, norm_reward=False, clip_obs=10.0)
        eval_vec.obs_rms = vec_normalize.obs_rms
        eval_vec.ret_rms = vec_normalize.ret_rms
        eval_vec.training = False

    rewards = []
    lengths = []
    component_sums = {}
    component_count = 0

    for _ in range(n_episodes):
        obs = eval_vec.reset()
        ep_reward = 0.0
        ep_len = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, done_arr, infos = eval_vec.step(action)
            ep_reward += float(rew[0])
            ep_len += 1
            done = done_arr[0]

            rc = infos[0].get("reward_components", {})
            if rc:
                component_count += 1
                for k, v in rc.items():
                    component_sums[k] = component_sums.get(k, 0.0) + float(v)

        rewards.append(ep_reward)
        lengths.append(ep_len)

    eval_vec.close()

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

            n_steps = 4096
            batch_size = _get_ppo_batch_size(n_steps, n_envs)
            logger.info(f"Filter phase PPO: device={device}, n_steps={n_steps}, batch_size={batch_size}")

            model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=_linear_schedule(3e-4),
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=10,
                gamma=0.995,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                device=device,
                verbose=0,
            )

            model.learn(total_timesteps=filter_steps, callback=callbacks)

            vn = vec_env if isinstance(vec_env, VecNormalize) else None
            eval_stats = _evaluate_after_training(
                model, xml_path, obs_config, code, n_episodes=10, vec_normalize=vn
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

    n_steps = 4096
    batch_size = _get_ppo_batch_size(n_steps, n_envs)
    logger.info(f"Full training PPO: device={device}, n_steps={n_steps}, batch_size={batch_size}")

    callbacks = [curve_cb, comp_cb]
    if status_dict is not None:
        callbacks.append(_StatusCallback(status_dict, phase_label, full_steps))
    if save_path:
        checkpoint_freq = max(100_000, full_steps // 5)
        callbacks.append(_CheckpointCallback(save_path, checkpoint_freq=checkpoint_freq))

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=_linear_schedule(3e-4),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.995,
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
        if isinstance(vec_env, VecNormalize):
            vec_env.save(save_path.replace(".zip", "_vecnorm.pkl"))

    vn = vec_env if isinstance(vec_env, VecNormalize) else None
    eval_stats = _evaluate_after_training(
        model, xml_path, obs_config, reward_code, n_episodes=20, vec_normalize=vn
    )

    success_rate = None
    success_metric = obs_config.get("success_metric")
    if success_metric:
        eval_vec = _create_vec_env(xml_path, obs_config, reward_code, n_envs=1, normalize=False)
        if vn is not None:
            eval_vec = VecNormalize(eval_vec, norm_obs=True, norm_reward=False, clip_obs=10.0)
            eval_vec.obs_rms = vn.obs_rms
            eval_vec.ret_rms = vn.ret_rms
            eval_vec.training = False
        successes = 0
        n_eval = 20
        for _ in range(n_eval):
            obs = eval_vec.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done_arr, infos = eval_vec.step(action)
                done = done_arr[0]
            raw_env = eval_vec.venv.envs[0] if hasattr(eval_vec, 'venv') else eval_vec.envs[0]
            while hasattr(raw_env, 'env'):
                raw_env = raw_env.env
            final_obs_vec = raw_env._get_obs()
            obs_dict = raw_env._make_obs_dict(final_obs_vec)
            try:
                ns = {"obs": obs_dict, "np": np}
                if eval(success_metric, {"__builtins__": {}}, ns):
                    successes += 1
            except Exception:
                pass
        eval_vec.close()
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
