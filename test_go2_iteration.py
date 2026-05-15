"""
Headless test: run 1 full Go2 iteration with API call logging.
Logs all prompts and responses to test_run_log.txt.
Uses reduced steps for faster turnaround while still being meaningful.
"""

import os
import sys
import json
import time
import logging
import traceback
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visual_eureka.envs.base_env import EurekaEnv
from visual_eureka.llm.client import NIMClient
from visual_eureka.llm.reward_generator import generate_candidates, extract_xml_summary, SYSTEM_PROMPT as REWARD_SYSTEM_PROMPT, _build_user_prompt
from visual_eureka.llm.text_reflection import generate_text_critique
from visual_eureka.training.trainer import run_filter_phase, run_full_training
from visual_eureka.training.renderer import render_eval_gifs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_go2")

LOG_FILE = "test_run_log.txt"
XML_PATH = "go2.xml"
OBS_CONFIG_PATH = "go2_obs_config.yaml"

K = 3
FILTER_STEPS = 80_000
FULL_STEPS = 300_000
EVAL_STEPS = 500


def log_section(f, title, content):
    f.write(f"\n{'='*80}\n")
    f.write(f"  {title}\n")
    f.write(f"{'='*80}\n\n")
    f.write(str(content))
    f.write("\n\n")


def main():
    start_time = time.time()

    with open(OBS_CONFIG_PATH) as f:
        obs_config = yaml.safe_load(f)

    xml_summary = extract_xml_summary(XML_PATH)

    with open(LOG_FILE, "w") as log:
        log_section(log, "GO2 TEST RUN", f"Started: {datetime.now().isoformat()}")
        log_section(log, "OBS CONFIG", yaml.dump(obs_config, default_flow_style=False))
        log_section(log, "XML SUMMARY", xml_summary)

        # --- Sanity check: can the environment be created? ---
        logger.info("Sanity check: creating EurekaEnv...")
        try:
            test_env = EurekaEnv(XML_PATH, obs_config)
            obs, _ = test_env.reset()
            logger.info(f"Env created. obs shape: {obs.shape}, n_substeps: {test_env.n_substeps}, dt: {test_env.dt:.4f}s")
            logger.info(f"Action space: {test_env.action_space.shape}, ctrl range: [{test_env.action_low[0]:.2f}, {test_env.action_high[0]:.2f}]")
            logger.info(f"Initial qpos[2] (height): {test_env.data.qpos[2]:.4f}")

            for i in range(10):
                action = test_env.action_space.sample()
                obs, reward, terminated, truncated, info = test_env.step(action)
                if i == 0:
                    logger.info(f"Step 0: reward={reward:.4f}, components={info.get('reward_components', {})}")
            test_env.close()
            log_section(log, "ENV SANITY CHECK", f"OK. obs={obs.shape}, n_substeps={test_env.n_substeps}, dt={test_env.dt:.4f}s, height={test_env.data.qpos[2]:.4f}")
        except Exception as e:
            logger.error(f"Env creation failed: {e}")
            log_section(log, "ENV SANITY CHECK", f"FAILED: {traceback.format_exc()}")
            return

        # --- Generate reward candidates ---
        logger.info(f"Generating {K} reward candidates...")
        client = NIMClient()

        user_prompt = _build_user_prompt(obs_config, xml_summary, K)
        log_section(log, "REWARD GENERATION: SYSTEM PROMPT", REWARD_SYSTEM_PROMPT)
        log_section(log, "REWARD GENERATION: USER PROMPT", user_prompt)

        try:
            t0 = time.time()
            candidates = generate_candidates(
                obs_config=obs_config,
                xml_summary=xml_summary,
                k=K,
                client=client,
            )
            gen_time = time.time() - t0
            logger.info(f"Got {len(candidates)} candidates in {gen_time:.1f}s")

            for i, code in enumerate(candidates):
                log_section(log, f"CANDIDATE {i}", code)
                logger.info(f"\n--- Candidate {i} ---\n{code[:200]}...")

                try:
                    test_env = EurekaEnv(XML_PATH, obs_config, reward_code=code)
                    obs, _ = test_env.reset()
                    total_r = 0.0
                    for step in range(20):
                        action = test_env.action_space.sample()
                        obs, reward, terminated, truncated, info = test_env.step(action)
                        total_r += reward
                        if terminated or truncated:
                            break
                    test_env.close()
                    logger.info(f"  Candidate {i} dry run: total_reward={total_r:.4f} over {step+1} steps, components={info.get('reward_components', {})}")
                    log_section(log, f"CANDIDATE {i} DRY RUN", f"total_reward={total_r:.4f}, steps={step+1}, components={info.get('reward_components', {})}")
                except Exception as e:
                    logger.warning(f"  Candidate {i} failed dry run: {e}")
                    log_section(log, f"CANDIDATE {i} DRY RUN", f"FAILED: {e}")

        except Exception as e:
            logger.error(f"Reward generation failed: {e}")
            log_section(log, "REWARD GENERATION FAILED", traceback.format_exc())
            return

        # --- Filter phase ---
        logger.info(f"Starting filter phase: {FILTER_STEPS} steps per candidate, {K} candidates...")
        t0 = time.time()
        try:
            best_code, best_stats, all_stats = run_filter_phase(
                xml_path=XML_PATH,
                obs_config=obs_config,
                candidate_codes=candidates,
                filter_steps=FILTER_STEPS,
                n_envs=4,
            )
            filter_time = time.time() - t0
            logger.info(f"Filter phase done in {filter_time:.1f}s")
            logger.info(f"Best candidate stats: {json.dumps(best_stats, indent=2, default=str)}")

            for i, stats in enumerate(all_stats):
                marker = " <-- BEST" if candidates[i] == best_code else ""
                log_section(log, f"FILTER STATS: CANDIDATE {i}{marker}", json.dumps(stats, indent=2, default=str))

            log_section(log, "BEST REWARD CODE (from filter)", best_code)

        except Exception as e:
            logger.error(f"Filter phase failed: {e}")
            log_section(log, "FILTER PHASE FAILED", traceback.format_exc())
            return

        # --- Full training ---
        model_save_path = "state/models/test_go2_model.zip"
        logger.info(f"Starting full training: {FULL_STEPS} steps...")
        t0 = time.time()
        try:
            training_stats = run_full_training(
                xml_path=XML_PATH,
                obs_config=obs_config,
                reward_code=best_code,
                full_steps=FULL_STEPS,
                n_envs=4,
                save_path=model_save_path,
            )
            train_time = time.time() - t0
            logger.info(f"Full training done in {train_time:.1f}s")
            logger.info(f"Training stats: {json.dumps(training_stats, indent=2, default=str)}")
            log_section(log, "FULL TRAINING STATS", json.dumps(training_stats, indent=2, default=str))

        except Exception as e:
            logger.error(f"Full training failed: {e}")
            log_section(log, "FULL TRAINING FAILED", traceback.format_exc())
            return

        # --- Render eval GIFs ---
        logger.info("Rendering eval GIFs...")
        try:
            gif_paths, thumb_path = render_eval_gifs(
                xml_path=XML_PATH,
                obs_config=obs_config,
                reward_code=best_code,
                model_path=model_save_path,
                base_output_path="state/models/test_go2_eval.gif",
                n_steps=EVAL_STEPS,
                n_rollouts=3,
            )
            logger.info(f"GIFs saved: {gif_paths}")
            log_section(log, "EVAL GIFS", f"Paths: {gif_paths}\nThumb: {thumb_path}")
        except Exception as e:
            logger.error(f"Rendering failed: {e}")
            log_section(log, "RENDERING FAILED", traceback.format_exc())

        # --- Text critique ---
        logger.info("Generating statistical critique...")
        try:
            text_critique = generate_text_critique(
                training_stats=training_stats,
                task_description=obs_config["task_description"],
                current_reward_code=best_code,
                client=client,
            )
            logger.info(f"Text critique:\n{text_critique}")
            log_section(log, "TEXT CRITIQUE", text_critique)
        except Exception as e:
            logger.error(f"Text critique failed: {e}")
            log_section(log, "TEXT CRITIQUE FAILED", traceback.format_exc())

        # --- Summary ---
        total_time = time.time() - start_time
        summary = f"""
TOTAL TIME: {total_time:.0f}s ({total_time/60:.1f} min)
CANDIDATES GENERATED: {len(candidates)}
BEST CANDIDATE FILTER REWARD: {best_stats.get('mean_reward', 'N/A')}
FULL TRAINING MEAN REWARD: {training_stats.get('mean_reward', 'N/A')}
FULL TRAINING STD REWARD: {training_stats.get('std_reward', 'N/A')}
MEAN EPISODE LENGTH: {training_stats.get('mean_ep_length', 'N/A')}
SUCCESS RATE: {training_stats.get('success_rate', 'N/A')}
REWARD COMPONENTS: {json.dumps(training_stats.get('reward_component_means', {}), indent=2)}
MODEL SAVED: {model_save_path}
"""
        log_section(log, "SUMMARY", summary)
        logger.info(summary)

    logger.info(f"Full log written to {LOG_FILE}")


if __name__ == "__main__":
    main()
