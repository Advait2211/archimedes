"""
Quick script to seed the DB with one iteration so you can test the
"Generate Next Iteration" loop without waiting through a full training run.

Run from the project root:
    source /Users/advaitdesai/Programming/eklavya/venv/bin/activate
    python seed_iteration.py

Then in the Streamlit app:
  1. Setup page — upload the XML + obs_config (sets session state; do NOT click Start)
  2. History page — select the run_id printed below
  3. Current Iteration page — click "Generate Next Iteration"
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uuid
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from visual_eureka.envs.base_env import EurekaEnv
from visual_eureka.state.iteration_store import IterationStore

XML_PATH = "visual_eureka/envs/uploads/test_quadruped.xml"
OBS_CONFIG_PATH = "visual_eureka/envs/uploads/test_obs_config.yaml"

REWARD_CODE = """
def compute_reward(obs, info, data, model):
    forward_vel = obs['base_lin_vel'][0]
    height = data.qpos[2]
    height_ok = float(np.exp(-((height - 0.35) ** 2) / 0.02))
    forward = float(np.clip(forward_vel, 0.0, 2.0))
    total = 0.7 * forward + 0.3 * height_ok
    return total, {"forward_vel": forward, "height_ok": height_ok}
"""

TRAIN_STEPS = 10_000

if __name__ == "__main__":
    import yaml

    obs_config = yaml.safe_load(open(OBS_CONFIG_PATH))

    def make_env():
        env = EurekaEnv(XML_PATH, obs_config, reward_code=REWARD_CODE)
        return Monitor(env)

    vec_env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
    )

    print(f"Training for {TRAIN_STEPS} steps...")
    model.learn(total_timesteps=TRAIN_STEPS)

    run_id = str(uuid.uuid4())[:8]
    os.makedirs("state/models", exist_ok=True)
    model_path = f"state/models/iteration_{run_id}_0.zip"
    model.save(model_path)
    print(f"Model saved: {model_path}")

    env = EurekaEnv(XML_PATH, obs_config, reward_code=REWARD_CODE)
    rewards = []
    for _ in range(5):
        obs, _ = env.reset()
        ep_r = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ep_r += r
            done = term or trunc
        rewards.append(ep_r)
    env.close()
    vec_env.close()

    mean_reward = float(np.mean(rewards))

    training_stats = {
        "mean_reward": mean_reward,
        "std_reward": float(np.std(rewards)),
        "mean_ep_length": 500.0,
        "reward_component_means": {"forward_vel": 0.3, "height_ok": 0.7},
        "success_rate": None,
        "training_curve": [[TRAIN_STEPS, mean_reward]],
        "model_path": model_path,
    }

    store = IterationStore()
    store.save_iteration(
        run_id=run_id,
        iteration=0,
        reward_code=REWARD_CODE,
        candidate_codes=[REWARD_CODE],
        filter_stats=[{"candidate_index": 0, "mean_reward": mean_reward}],
        training_stats=training_stats,
        eval_gif_path=None,
        visual_critique="The robot is not yet moving forward efficiently. The forward velocity component should be weighted more heavily, and a stability term for roll/pitch would help prevent tipping.",
        text_critique=f"After {TRAIN_STEPS} steps, mean episode reward is {mean_reward:.3f}. The reward signal is simple — consider adding a survival bonus and penalizing lateral drift to encourage straighter locomotion.",
        mean_reward=mean_reward,
        success_rate=None,
        model_path=model_path,
    )
    store.close()

    print(f"\nSeeded run_id: {run_id}")
    print(f"Mean reward:   {mean_reward:.4f}")
    print("\nNext steps:")
    print("  1. streamlit run visual_eureka/app.py")
    print(f"  2. Setup page — upload {XML_PATH} + {OBS_CONFIG_PATH} (do NOT click Start)")
    print(f"  3. History page — select run '{run_id}'")
    print("  4. Current Iteration page — click 'Generate Next Iteration'")
