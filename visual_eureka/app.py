import os
import sys
import uuid
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import pandas as pd
import streamlit as st

from visual_eureka.envs.base_env import EurekaEnv, validate_obs_config
from visual_eureka.llm.client import NIMClient
from visual_eureka.llm.reward_generator import generate_candidates, extract_xml_summary
from visual_eureka.llm.visual_reflection import generate_visual_critique
from visual_eureka.llm.text_reflection import generate_text_critique
from visual_eureka.training.trainer import run_filter_phase, run_full_training
from visual_eureka.training.renderer import render_eval_gif, render_eval_gifs
from visual_eureka.state.iteration_store import IterationStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Visual Eureka", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface-raised: #21253a;
    --accent: #4f6ef7;
    --accent-secondary: #2ecc8f;
    --warning: #e07b39;
    --text-primary: #e8eaf0;
    --text-secondary: #8b90a0;
    --border: #2a2d3e;
    --danger: #c0392b;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

.stApp {
    background-color: var(--bg);
}

h1 {
    font-size: 24px !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}

h2, h3 {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}

.subtitle {
    font-size: 14px;
    color: var(--text-secondary);
    margin-top: -10px;
}

.label-text {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: var(--accent);
}

.metric-label {
    font-size: 12px;
    font-weight: 500;
    color: var(--text-secondary);
}

.caption-text {
    font-size: 12px;
    color: var(--text-secondary);
}

code, .stCodeBlock {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
}

.stButton > button {
    background-color: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 500 !important;
}

.stButton > button:hover {
    opacity: 0.9;
}

.danger-btn > button {
    background-color: var(--danger) !important;
}

.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
    background-color: var(--surface-raised) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
}

.stSlider > div > div > div {
    background-color: var(--surface-raised) !important;
}

div[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}

div[data-testid="stMetricDelta"] svg {
    display: none;
}

.stTabs [data-baseweb="tab-list"] {
    border-bottom: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    font-size: 14px;
    font-weight: 600;
}

div[data-testid="stExpander"] details summary p {
    font-size: 14px !important;
    font-weight: 600 !important;
}

.stSidebar {
    background-color: var(--surface) !important;
}

.stSidebar .stRadio label {
    color: var(--text-primary) !important;
}

div[data-testid="stFileUploader"] {
    background-color: var(--surface-raised);
    border: 1px dashed var(--border);
    border-radius: 4px;
}
</style>
"""

FREEZE_CSS = """
<style>
.stApp, .stApp * {
    pointer-events: none !important;
    cursor: wait !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _init_session_state():
    defaults = {
        "run_id": None,
        "obs_config": None,
        "xml_path": None,
        "current_iteration": 0,
        "store": None,
        "training_status": {"running": False, "phase": "", "step": 0, "total_steps": 0},
        "current_reward_code": None,
        "nim_client": None,
        "xml_summary": None,
        "config_k": 4,
        "config_filter_steps": 100_000,
        "config_full_steps": 500_000,
        "config_eval_steps": 500,
        "reverted_code": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session_state()


def _save_uploaded_file(uploaded_file, dest_path: str) -> str:
    parent = os.path.dirname(dest_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest_path


def _load_obs_config(config_bytes: bytes) -> dict:
    config = yaml.safe_load(config_bytes)
    validate_obs_config(config)
    return config


def _run_iteration(
    run_id: str,
    iteration: int,
    xml_path: str,
    obs_config: dict,
    xml_summary: str,
    client: NIMClient,
    store: IterationStore,
    k: int,
    filter_steps: int,
    full_steps: int,
    eval_steps: int,
    current_reward_code: str = None,
    reflection: str = None,
    phase_ph=None,
    sidebar_ph=None,
):
    status = st.session_state.training_status
    status["_phase_ph"] = phase_ph
    status["_sidebar_ph"] = sidebar_ph

    def _set_phase(msg):
        status["phase"] = msg
        status["step"] = 0
        if phase_ph is not None:
            phase_ph.info(msg)

    _set_phase(f"Generating {k} reward candidates...")
    candidates = generate_candidates(
        obs_config=obs_config,
        xml_summary=xml_summary,
        k=k,
        current_reward_code=current_reward_code,
        reflection=reflection,
        client=client,
    )

    best_code, best_filter_stats, all_filter_stats = run_filter_phase(
        xml_path=xml_path,
        obs_config=obs_config,
        candidate_codes=candidates,
        filter_steps=filter_steps,
        n_envs=4,
        status_dict=status,
    )

    model_save_path = os.path.join("state", "models", f"iteration_{run_id}_{iteration}.zip")

    training_stats = run_full_training(
        xml_path=xml_path,
        obs_config=obs_config,
        reward_code=best_code,
        full_steps=full_steps,
        n_envs=4,
        save_path=model_save_path,
        status_dict=status,
    )

    base_gif_path = os.path.join("state", "models", f"eval_{run_id}_{iteration}.gif")
    _set_phase("Rendering evaluation GIFs (3 rollouts)...")
    gif_paths, thumb_path = render_eval_gifs(
        xml_path=xml_path,
        obs_config=obs_config,
        reward_code=best_code,
        model_path=model_save_path,
        base_output_path=base_gif_path,
        n_steps=eval_steps,
        n_rollouts=3,
    )
    primary_gif = gif_paths[len(gif_paths) // 2]

    _set_phase("Generating visual critique...")
    visual_critique = ""
    try:
        visual_critique = generate_visual_critique(
            gif_path=primary_gif,
            task_description=obs_config["task_description"],
            robot_type=obs_config["robot_type"],
            current_reward_code=best_code,
            client=client,
        )
    except Exception as e:
        visual_critique = f"Visual critique failed: {e}"
        logger.error(f"Visual critique error: {e}")

    _set_phase("Generating statistical critique...")
    text_critique = ""
    try:
        text_critique = generate_text_critique(
            training_stats=training_stats,
            task_description=obs_config["task_description"],
            current_reward_code=best_code,
            client=client,
        )
    except Exception as e:
        text_critique = f"Text critique failed: {e}"
        logger.error(f"Text critique error: {e}")

    store.save_iteration(
        run_id=run_id,
        iteration=iteration,
        reward_code=best_code,
        candidate_codes=candidates,
        filter_stats=all_filter_stats,
        training_stats=training_stats,
        eval_gif_path=primary_gif,
        eval_gif_paths=gif_paths,
        thumb_path=thumb_path,
        visual_critique=visual_critique,
        text_critique=text_critique,
        mean_reward=training_stats["mean_reward"],
        success_rate=training_stats.get("success_rate"),
        model_path=model_save_path,
    )

    st.session_state.current_reward_code = best_code
    st.session_state.current_iteration = iteration


def _run_auto_loop(
    n_iterations: int,
    starting_iteration: int,
    run_id: str,
    xml_path: str,
    obs_config: dict,
    xml_summary: str,
    client: NIMClient,
    store: IterationStore,
    k: int,
    filter_steps: int,
    full_steps: int,
    eval_steps: int,
    phase_ph,
    sidebar_ph,
):
    current_code = None
    current_reflection = None

    seed_data = store.get_iteration(run_id, starting_iteration)
    if seed_data:
        current_code = seed_data.get("reward_code")
        vc = seed_data.get("visual_critique", "") or ""
        tc = seed_data.get("text_critique", "") or ""
        current_reflection = f"{vc}\n\n{tc}".strip()

    last_completed = starting_iteration
    for i in range(n_iterations):
        next_iter = starting_iteration + 1 + i
        status = st.session_state.training_status
        status["phase"] = f"Auto-run: iteration {next_iter} of {starting_iteration + n_iterations}"
        if phase_ph is not None:
            phase_ph.info(status["phase"])

        try:
            _run_iteration(
                run_id=run_id,
                iteration=next_iter,
                xml_path=xml_path,
                obs_config=obs_config,
                xml_summary=xml_summary,
                client=client,
                store=store,
                k=k,
                filter_steps=filter_steps,
                full_steps=full_steps,
                eval_steps=eval_steps,
                current_reward_code=current_code,
                reflection=current_reflection,
                phase_ph=phase_ph,
                sidebar_ph=sidebar_ph,
            )
            last_completed = next_iter

            iter_data = store.get_iteration(run_id, next_iter)
            if iter_data:
                current_code = iter_data.get("reward_code")
                vc = iter_data.get("visual_critique", "") or ""
                tc = iter_data.get("text_critique", "") or ""
                current_reflection = f"{vc}\n\n{tc}".strip()

        except Exception as e:
            logger.error(f"Auto-run stopped at iteration {next_iter}: {e}")
            break

    st.session_state.current_iteration = last_completed


def page_setup():
    st.title("Visual Eureka")
    st.markdown(
        '<p class="subtitle">Evolutionary Reward Design for MuJoCo Locomotion</p>',
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns(2)

    with left_col:
        xml_file = st.file_uploader("MuJoCo XML", type=["xml"])
        config_file = st.file_uploader("Observation Config (YAML)", type=["yaml", "yml"])

        if xml_file is not None:
            xml_dest = "uploaded_robot.xml"
            _save_uploaded_file(xml_file, xml_dest)
            st.session_state.xml_path = xml_dest

            try:
                st.session_state.xml_summary = extract_xml_summary(xml_dest)
                st.caption("XML loaded successfully")
            except Exception as e:
                st.error(f"Failed to parse XML: {e}")

        if config_file is not None:
            config_dest = os.path.join("envs", "uploads", "obs_config.yaml")
            _save_uploaded_file(config_file, config_dest)

            try:
                obs_config = _load_obs_config(config_file.getvalue())
                st.session_state.obs_config = obs_config

                rows = []
                for name, comp in obs_config["obs_components"].items():
                    rows.append({
                        "Component": name,
                        "Shape": f"({comp['end'] - comp['start']},)",
                        "Description": comp["description"],
                    })
                st.table(pd.DataFrame(rows))
            except Exception as e:
                st.error(f"Invalid obs_config: {e}")

        if st.session_state.obs_config:
            task_desc = st.text_area(
                "Task Description",
                value=st.session_state.obs_config.get("task_description", ""),
                key="task_desc_input",
            )
            if task_desc:
                st.session_state.obs_config["task_description"] = task_desc

    with right_col:
        k = st.slider("Candidates per iteration (K)", 2, 8, 4)
        filter_steps = st.slider(
            "Filter phase steps", 50_000, 500_000, 100_000, step=50_000
        )
        full_steps = st.slider(
            "Full training steps", 500_000, 3_000_000, 500_000, step=100_000
        )
        eval_steps = st.slider("Eval rollout steps", 200, 1000, 500, step=100)

        st.session_state.config_k = k
        st.session_state.config_filter_steps = filter_steps
        st.session_state.config_full_steps = full_steps
        st.session_state.config_eval_steps = eval_steps

        n_auto_setup = st.slider("Total iterations to run (1 = manual review after first)", 1, 20, 1, key="n_auto_setup")

        if st.button("Start Experiment" if n_auto_setup == 1 else f"Start + Auto-run {n_auto_setup} iterations"):
            if st.session_state.xml_path is None:
                st.error("Upload a MuJoCo XML file first.")
                return
            if st.session_state.obs_config is None:
                st.error("Upload an observation config YAML first.")
                return

            try:
                client = NIMClient()
                st.session_state.nim_client = client
            except Exception as e:
                st.error(f"Failed to initialize NIM client: {e}")
                return

            store = IterationStore()
            st.session_state.store = store

            run_id = str(uuid.uuid4())[:8]
            st.session_state.run_id = run_id

            st.session_state.training_status = {
                "running": True, "phase": "Starting...", "step": 0, "total_steps": 0,
            }
            st.markdown(FREEZE_CSS, unsafe_allow_html=True)
            phase_ph = st.empty()
            sidebar_ph = st.sidebar.empty()

            try:
                _run_iteration(
                    run_id=run_id,
                    iteration=0,
                    xml_path=st.session_state.xml_path,
                    obs_config=st.session_state.obs_config,
                    xml_summary=st.session_state.xml_summary,
                    client=client,
                    store=store,
                    k=k,
                    filter_steps=filter_steps,
                    full_steps=full_steps,
                    eval_steps=eval_steps,
                    phase_ph=phase_ph,
                    sidebar_ph=sidebar_ph,
                )
            except Exception as e:
                st.session_state.training_status["running"] = False
                st.error(f"Experiment failed: {e}")
                logger.exception("Experiment error")
                return

            if n_auto_setup > 1:
                _run_auto_loop(
                    n_iterations=n_auto_setup - 1,
                    starting_iteration=0,
                    run_id=run_id,
                    xml_path=st.session_state.xml_path,
                    obs_config=st.session_state.obs_config,
                    xml_summary=st.session_state.xml_summary,
                    client=client,
                    store=store,
                    k=k,
                    filter_steps=filter_steps,
                    full_steps=full_steps,
                    eval_steps=eval_steps,
                    phase_ph=phase_ph,
                    sidebar_ph=sidebar_ph,
                )

            st.session_state.training_status["running"] = False
            st.rerun()


def page_current_iteration():
    store = st.session_state.store
    if store is None:
        store = IterationStore()
        st.session_state.store = store

    if st.session_state.run_id is None:
        runs = store.list_runs()
        if not runs:
            st.info("Start an experiment on the Setup page.")
            return
        st.session_state.run_id = runs[0]
        iters = store.get_all_iterations(runs[0])
        if iters:
            st.session_state.current_iteration = iters[-1]["iteration"]

    run_id = st.session_state.run_id
    iteration = st.session_state.current_iteration
    iter_data = store.get_iteration(run_id, iteration)

    if iter_data is None:
        st.info("No iteration data found. Run an experiment first.")
        return

    prev_data = store.get_iteration(run_id, iteration - 1) if iteration > 0 else None

    header_left, header_right = st.columns([3, 1])
    with header_left:
        st.subheader(f"Iteration {iteration}")
    with header_right:
        mean_r = iter_data.get("mean_reward") or 0.0
        delta = None
        if prev_data and prev_data.get("mean_reward") is not None:
            prev_r = prev_data.get("mean_reward") or 0.0
            delta = f"{mean_r - prev_r:+.4f} vs previous"
        st.metric("Mean Reward", f"{mean_r:.4f}", delta=delta)

    col1, col2, col3 = st.columns(3)

    with col1:
        gif_paths = iter_data.get("eval_gif_paths") or []
        if isinstance(gif_paths, str):
            gif_paths = [gif_paths]
        if not gif_paths:
            fallback = iter_data.get("eval_gif_path")
            if fallback:
                gif_paths = [fallback]
        existing_gifs = [p for p in gif_paths if p and os.path.exists(p)]
        if existing_gifs:
            for idx, gp in enumerate(existing_gifs):
                st.image(gp, caption=f"Rollout {idx + 1}")
        else:
            st.caption("No evaluation GIFs available")

    with col2:
        stats = iter_data.get("training_stats", {})
        curve = stats.get("training_curve", [])
        if curve:
            df = pd.DataFrame(curve, columns=["step", "mean_reward"])
            st.line_chart(df, x="step", y="mean_reward")
        else:
            st.caption("No training curve data")

    with col3:
        comp_means = stats.get("reward_component_means", {})
        if comp_means:
            df = pd.DataFrame(
                list(comp_means.items()), columns=["component", "value"]
            )
            st.bar_chart(df, x="component", y="value")
        else:
            st.caption("No reward component data")

    reward_code = iter_data.get("reward_code", "")
    visual_critique = iter_data.get("visual_critique", "")
    text_critique = iter_data.get("text_critique", "")

    tab_visual, tab_stats = st.tabs(["Visual Critique", "Stats Critique"])

    with tab_visual:
        with st.expander("Current Reward Code"):
            st.code(reward_code, language="python")
        st.text_area(
            "Critique -- pre-filled by model, edit before generating next iteration",
            value=visual_critique,
            height=220,
            key="critique_visual",
        )

    with tab_stats:
        with st.expander("Current Reward Code"):
            st.code(reward_code, language="python")
        st.text_area(
            "Critique -- pre-filled by model, edit before generating next iteration",
            value=text_critique,
            height=220,
            key="critique_stats",
        )

    with st.expander("Override training config for next iteration"):
        st.caption("Filter steps are fixed per experiment. Only these three affect next iterations.")
        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            st.session_state.config_k = st.slider(
                "Candidates (K)", 2, 8,
                st.session_state.config_k,
                key="override_k",
            )
        with oc2:
            st.session_state.config_full_steps = st.slider(
                "Full training steps", 100_000, 3_000_000,
                st.session_state.config_full_steps,
                step=100_000,
                key="override_full_steps",
            )
        with oc3:
            st.session_state.config_eval_steps = st.slider(
                "Eval rollout steps", 200, 1000,
                st.session_state.config_eval_steps,
                step=100,
                key="override_eval_steps",
            )

    bottom_left, bottom_mid, bottom_right = st.columns(3)

    with bottom_left:
        notes = st.text_input("Iteration notes (optional)", key="iter_notes")
        if st.button("Generate Next Iteration"):
            visual_edited = st.session_state.get("critique_visual", "")
            stats_edited = st.session_state.get("critique_stats", "")
            combined_critique = f"{visual_edited}\n\n{stats_edited}".strip()

            if notes:
                store.save_iteration(
                    run_id=run_id,
                    iteration=iteration,
                    user_critique=combined_critique,
                    notes=notes,
                )

            base_code = st.session_state.reverted_code or reward_code
            st.session_state.reverted_code = None
            next_iter = iteration + 1

            client = st.session_state.nim_client
            if client is None:
                try:
                    client = NIMClient()
                    st.session_state.nim_client = client
                except Exception as e:
                    st.error(f"Failed to initialize NIM client: {e}")
                    return

            st.session_state.training_status = {
                "running": True, "phase": "Starting...", "step": 0, "total_steps": 0,
            }
            st.markdown(FREEZE_CSS, unsafe_allow_html=True)
            phase_ph = st.empty()
            sidebar_ph = st.sidebar.empty()

            try:
                _run_iteration(
                    run_id=run_id,
                    iteration=next_iter,
                    xml_path=st.session_state.xml_path,
                    obs_config=st.session_state.obs_config,
                    xml_summary=st.session_state.xml_summary,
                    client=client,
                    store=store,
                    k=st.session_state.config_k,
                    filter_steps=st.session_state.config_filter_steps,
                    full_steps=st.session_state.config_full_steps,
                    eval_steps=st.session_state.config_eval_steps,
                    current_reward_code=base_code,
                    reflection=combined_critique,
                    phase_ph=phase_ph,
                    sidebar_ph=sidebar_ph,
                )
            except Exception as e:
                st.session_state.training_status["running"] = False
                st.error(f"Next iteration failed: {e}")
                logger.exception("Iteration error")
                return

            st.session_state.training_status["running"] = False
            st.rerun()

    with bottom_mid:
        st.download_button(
            "Download Reward (.py)",
            data=reward_code,
            file_name=f"reward_iter_{iteration}.py",
        )

    with bottom_right:
        if st.button("Re-render Eval GIF"):
            model_path = iter_data.get("model_path")
            if model_path and os.path.exists(model_path):
                gif_out = os.path.join(
                    "state", "models", f"eval_{run_id}_{iteration}.gif"
                )
                with st.spinner("Re-rendering..."):
                    render_eval_gif(
                        xml_path=st.session_state.xml_path,
                        obs_config=st.session_state.obs_config,
                        reward_code=reward_code,
                        model_path=model_path,
                        output_path=gif_out,
                        n_steps=st.session_state.config_eval_steps,
                    )
                store.save_iteration(
                    run_id=run_id, iteration=iteration, eval_gif_path=gif_out
                )
                st.rerun()
            else:
                st.error("Model file not found for re-rendering.")

    st.divider()
    st.subheader("Auto-run")
    st.caption("Runs N iterations automatically using model-generated critiques as feedback — no human input between iterations.")

    auto_col_left, auto_col_right = st.columns(2)
    with auto_col_left:
        n_auto = st.slider("Iterations to run", 1, 20, 3, key="n_auto_iterations")
    with auto_col_right:
        st.write("")
        st.write("")
        if st.button("Start Auto-run", type="primary"):
            client = st.session_state.nim_client
            if client is None:
                try:
                    client = NIMClient()
                    st.session_state.nim_client = client
                except Exception as e:
                    st.error(f"Failed to initialize NIM client: {e}")
                    return

            if st.session_state.xml_path is None or st.session_state.obs_config is None:
                st.error("XML and obs config must be loaded. Go to Setup and upload the files first.")
                return

            st.session_state.training_status = {
                "running": True, "phase": "Starting auto-run...", "step": 0, "total_steps": 0,
            }
            st.markdown(FREEZE_CSS, unsafe_allow_html=True)
            phase_ph = st.empty()
            sidebar_ph = st.sidebar.empty()

            _run_auto_loop(
                n_iterations=n_auto,
                starting_iteration=iteration,
                run_id=run_id,
                xml_path=st.session_state.xml_path,
                obs_config=st.session_state.obs_config,
                xml_summary=st.session_state.xml_summary,
                client=client,
                store=store,
                k=st.session_state.config_k,
                filter_steps=st.session_state.config_filter_steps,
                full_steps=st.session_state.config_full_steps,
                eval_steps=st.session_state.config_eval_steps,
                phase_ph=phase_ph,
                sidebar_ph=sidebar_ph,
            )

            st.session_state.training_status["running"] = False
            st.rerun()


def page_history():
    store = st.session_state.store
    if store is None:
        store = IterationStore()
        st.session_state.store = store

    runs = store.list_runs()
    if not runs:
        st.info("No experiment runs found.")
        return

    current_run = st.session_state.run_id
    selected_run = st.selectbox(
        "Active run",
        runs,
        index=runs.index(current_run) if current_run in runs else 0,
    )

    if selected_run != st.session_state.run_id:
        st.session_state.run_id = selected_run
        iters = store.get_all_iterations(selected_run)
        if iters:
            st.session_state.current_iteration = iters[-1]["iteration"]

    iterations = store.get_all_iterations(selected_run)

    if iterations:
        chart_data = pd.DataFrame([
            {"iteration": it["iteration"], "mean_reward": it.get("mean_reward") or 0.0}
            for it in iterations
        ])
        st.line_chart(chart_data, x="iteration", y="mean_reward")

    for it in reversed(iterations):
        n = it["iteration"]
        r = it.get("mean_reward") or 0.0
        with st.expander(f"Iteration {n} -- Mean Reward: {r:.4f}"):
            left, right = st.columns(2)

            with left:
                thumb = it.get("thumb_path")
                if thumb and os.path.exists(thumb):
                    st.image(thumb, width=280)
                else:
                    gif_path = it.get("eval_gif_path")
                    if gif_path and os.path.exists(gif_path):
                        st.image(gif_path, width=280)

            with right:
                code = it.get("reward_code", "")
                st.code(code, language="python")

                vc = it.get("visual_critique", "")
                if vc:
                    display_vc = vc[:300] + "..." if len(vc) > 300 else vc
                    st.caption(f"Visual critique: {display_vc}")

                uc = it.get("user_critique", "")
                if uc:
                    display_uc = uc[:300] + "..." if len(uc) > 300 else uc
                    st.caption(f"User critique: {display_uc}")

                st.metric("Mean Reward", f"{r:.4f}")

                if st.button(f"Revert to Iteration {n}", key=f"revert_{n}", type="primary"):
                    st.session_state.reverted_code = code
                    st.session_state.current_reward_code = code
                    st.info(f"Reverted to iteration {n}. Go to Current Iteration to generate the next one.")


_ts = st.session_state.training_status
if _ts.get("running"):
    st.markdown(FREEZE_CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("**Training in progress — UI locked**")
        st.caption(_ts.get("phase", ""))
        step = _ts.get("step", 0)
        total = _ts.get("total_steps", 0)
        if total > 0:
            st.progress(min(step / total, 1.0), text=f"{step:,} / {total:,} steps")

page = st.sidebar.radio("Navigation", ["Setup", "Current Iteration", "History"])

if page == "Setup":
    page_setup()
elif page == "Current Iteration":
    page_current_iteration()
elif page == "History":
    page_history()
