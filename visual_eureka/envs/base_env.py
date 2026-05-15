import math
import logging
import os
import traceback
from datetime import datetime, timezone

import gymnasium
import numpy as np
import mujoco
from gymnasium import spaces

logger = logging.getLogger(__name__)

_REWARD_ERROR_LOG = "state/reward_errors.log"


def _log_reward_error_once(err_key: str, tb: str) -> None:
    os.makedirs("state", exist_ok=True)
    with open(_REWARD_ERROR_LOG, "a") as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"{datetime.now(timezone.utc).isoformat()}\n")
        f.write(tb)

REQUIRED_OBS_FIELDS = [
    "task_description", "robot_type", "obs_vector_dim",
    "obs_components", "episode_length",
]


def validate_obs_config(obs_config: dict) -> None:
    for field in REQUIRED_OBS_FIELDS:
        if field not in obs_config:
            raise ValueError(f"obs_config missing required field: '{field}'")

    if not isinstance(obs_config["obs_components"], dict):
        raise ValueError("obs_config field 'obs_components' must be a dict")

    if not obs_config["obs_components"]:
        raise ValueError("obs_config field 'obs_components' must not be empty")

    dim = obs_config["obs_vector_dim"]
    for name, comp in obs_config["obs_components"].items():
        for key in ("start", "end", "description"):
            if key not in comp:
                raise ValueError(
                    f"obs_config component '{name}' missing field: '{key}'"
                )
        if comp["start"] < 0:
            raise ValueError(
                f"obs_config component '{name}': start must be >= 0, got {comp['start']}"
            )
        if comp["end"] <= comp["start"]:
            raise ValueError(
                f"obs_config component '{name}': end ({comp['end']}) must be > start ({comp['start']})"
            )
        if comp["end"] > dim:
            raise ValueError(
                f"obs_config component '{name}': end ({comp['end']}) exceeds obs_vector_dim ({dim})"
            )

    if obs_config["robot_type"] not in ("quadruped", "biped"):
        raise ValueError(
            f"obs_config field 'robot_type' must be 'quadruped' or 'biped', got '{obs_config['robot_type']}'"
        )


class EurekaEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    DEFAULT_SUBSTEPS = {
        "quadruped": 5,
        "biped": 4,
    }

    def __init__(
        self,
        xml_path: str,
        obs_config: dict,
        reward_code: str = None,
        render_mode: str = None,
    ):
        super().__init__()
        validate_obs_config(obs_config)

        self.xml_path = xml_path
        self.obs_config = obs_config
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        robot_type = obs_config.get("robot_type", "quadruped")
        self.n_substeps = self.DEFAULT_SUBSTEPS.get(robot_type, 5)
        self.dt = self.model.opt.timestep * self.n_substeps

        obs_dim = obs_config["obs_vector_dim"]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        n_actuators = self.model.nu
        ctrl_range = self.model.actuator_ctrlrange
        self.action_low = ctrl_range[:, 0].copy()
        self.action_high = ctrl_range[:, 1].copy()

        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0 and self.model.key_qpos is not None:
            self._home_qpos = self.model.key_qpos[key_id][7:].copy()
        else:
            self._home_qpos = np.zeros(n_actuators)

        self._use_pd = True
        self._kp = np.full(n_actuators, 20.0)
        self._kd = np.full(n_actuators, 0.5)
        self._action_scale_pos = np.full(n_actuators, 0.5)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_actuators,), dtype=np.float32
        )

        self.max_episode_steps = obs_config.get("episode_length", 1000)
        self._current_step = 0
        self.reward_fn = None
        self._logged_reward_errors: set[str] = set()

        if reward_code is not None:
            self.set_reward_fn(reward_code)

    def set_reward_fn(self, code: str) -> None:
        ns = {"np": np, "math": math, "mujoco": mujoco}
        try:
            exec(code, ns)
        except Exception:
            raise ValueError(
                f"Failed to exec reward code:\n{traceback.format_exc()}"
            )
        if "compute_reward" not in ns or not callable(ns["compute_reward"]):
            raise ValueError(
                "Reward code must define a callable 'compute_reward' function"
            )
        self.reward_fn = ns["compute_reward"]

    def _make_obs_dict(self, obs_vec: np.ndarray) -> dict:
        obs_dict = {}
        for name, comp in self.obs_config["obs_components"].items():
            obs_dict[name] = obs_vec[comp["start"]:comp["end"]].copy()
        return obs_dict

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos
        qvel = self.data.qvel

        components = []
        for name, comp in self.obs_config["obs_components"].items():
            start = comp["start"]
            end = comp["end"]
            size = end - start

            desc = comp["description"].lower()
            if "joint" in desc and "pos" in desc:
                components.append(qpos[7:7 + size])
            elif "joint" in desc and "vel" in desc:
                components.append(qvel[6:6 + size])
            elif "quat" in desc or "orientation" in desc:
                components.append(qpos[3:7])
            elif "lin" in desc and "vel" in desc:
                components.append(qvel[0:3])
            elif "ang" in desc and "vel" in desc:
                components.append(qvel[3:6])
            elif "height" in desc:
                components.append(np.array([qpos[2]]))
            elif "gravity" in desc or "projected" in desc:
                quat = qpos[3:7]
                grav_world = np.array([0.0, 0.0, -1.0])
                grav_body = self._rotate_vector_by_quat_inv(grav_world, quat)
                components.append(grav_body)
            elif "contact" in desc or "foot" in desc:
                n_contacts = size
                contacts = np.zeros(n_contacts)
                for i in range(self.data.ncon):
                    con = self.data.contact[i]
                    geom1_name = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1
                    )
                    geom2_name = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2
                    )
                    if geom1_name and "foot" in geom1_name.lower():
                        idx = self._foot_geom_to_index(geom1_name, n_contacts)
                        if idx is not None:
                            contacts[idx] = 1.0
                    if geom2_name and "foot" in geom2_name.lower():
                        idx = self._foot_geom_to_index(geom2_name, n_contacts)
                        if idx is not None:
                            contacts[idx] = 1.0
                components.append(contacts)
            else:
                components.append(np.zeros(size))

        obs = np.concatenate(components).astype(np.float32)

        expected_dim = self.obs_config["obs_vector_dim"]
        if obs.shape[0] != expected_dim:
            if obs.shape[0] < expected_dim:
                obs = np.concatenate([obs, np.zeros(expected_dim - obs.shape[0], dtype=np.float32)])
            else:
                obs = obs[:expected_dim]

        return obs

    def _rotate_vector_by_quat_inv(self, vec: np.ndarray, quat: np.ndarray) -> np.ndarray:
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        rot = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y + w*z),     2*(x*z - w*y)],
            [2*(x*y - w*z),     1 - 2*(x*x + z*z), 2*(y*z + w*x)],
            [2*(x*z + w*y),     2*(y*z - w*x),     1 - 2*(x*x + y*y)],
        ])
        return rot.T @ vec

    def _foot_geom_to_index(self, geom_name: str, n_contacts: int) -> int | None:
        name_upper = geom_name.upper()
        foot_order = ["FL", "FR", "RL", "RR"]
        for i, prefix in enumerate(foot_order):
            if prefix in name_upper and i < n_contacts:
                return i
        return None

    def step(self, action: np.ndarray):
        if self._use_pd:
            target_pos = self._home_qpos + action * self._action_scale_pos
            current_pos = self.data.qpos[7:7 + self.model.nu]
            current_vel = self.data.qvel[6:6 + self.model.nu]
            ctrl = self._kp * (target_pos - current_pos) - self._kd * current_vel
            ctrl = np.clip(ctrl, self.action_low, self.action_high)
        else:
            ctrl = self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)
        self.data.ctrl[:] = ctrl
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self._current_step += 1

        obs = self._get_obs()
        obs_dict = self._make_obs_dict(obs)
        info = {"action": action.copy()}

        reward = 0.0
        reward_components = {}

        if self.reward_fn is not None:
            try:
                reward, reward_components = self.reward_fn(
                    obs_dict, info, self.data, self.model
                )
                if not np.isfinite(reward):
                    reward = 0.0
            except Exception as e:
                err_key = f"{type(e).__name__}: {e}"
                if err_key not in self._logged_reward_errors:
                    self._logged_reward_errors.add(err_key)
                    logger.warning(f"Reward function error: {e}\n{traceback.format_exc()}")
                    _log_reward_error_once(err_key, traceback.format_exc())
                reward = 0.0
                reward_components = {}

        if "survival" not in reward_components:
            reward += 1.0
            reward_components["survival"] = 1.0

        info["reward_components"] = reward_components

        base_pos = self.data.qpos[:3]
        base_quat = self.data.qpos[3:7]
        terminated = False

        height = base_pos[2]
        if height < 0.18 or height > 1.0:
            terminated = True

        w, x, y, z = base_quat[0], base_quat[1], base_quat[2], base_quat[3]
        sinr = 2.0 * (w * x + y * z)
        cosr = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr, cosr)
        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = math.asin(sinp)

        if abs(roll) > 0.7 or abs(pitch) > 0.5:
            terminated = True

        truncated = self._current_step >= self.max_episode_steps

        return obs, float(reward), terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            self.data.qpos[:] = self.model.key_qpos[key_id]

        n_joints = self.model.nq - 7
        self.data.qpos[7:] += self.np_random.uniform(-0.02, 0.02, size=n_joints)
        self.data.qvel[:] += self.np_random.uniform(-0.01, 0.01, size=self.model.nv)

        mujoco.mj_forward(self.model, self.data)
        self._current_step = 0

        obs = self._get_obs()
        return obs, {}

    def close(self):
        pass
