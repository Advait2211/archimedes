import re
import logging
import xml.etree.ElementTree as ET

from .client import NIMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert reward function designer for MuJoCo locomotion RL.
You write Python reward functions for quadruped and biped robots.
Return ONLY the function code. No markdown fences, no explanation.
Each function must match this exact signature:
  def compute_reward(obs, info, data, model) -> tuple[float, dict]
Available in scope: numpy as np, math, mujoco.
MjData API: data.qpos, data.qvel, data.cfrc_ext, data.body_xpos[model.body('name').id], data.contact, etc.
The dict return must map component name strings to float values."""


def extract_xml_summary(xml_path: str) -> str:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    bodies = []
    joints = []
    geoms = []
    actuators = []

    for body in root.iter("body"):
        name = body.get("name", "unnamed")
        pos = body.get("pos", "0 0 0")
        bodies.append(f"  {name} (pos={pos})")

    for joint in root.iter("joint"):
        name = joint.get("name")
        if name:
            axis = joint.get("axis", "0 1 0")
            jrange = joint.get("range", "none")
            joints.append(f"  {name} (axis={axis}, range={jrange})")

    for geom in root.iter("geom"):
        name = geom.get("name")
        if name:
            gtype = geom.get("type", "sphere")
            geoms.append(f"  {name} (type={gtype})")

    for actuator in root.iter("motor"):
        name = actuator.get("name", "unnamed")
        joint = actuator.get("joint", "unknown")
        ctrl = actuator.get("ctrlrange", "none")
        actuators.append(f"  {name} -> {joint} (ctrlrange={ctrl})")

    sections = []
    if bodies:
        sections.append("Bodies:\n" + "\n".join(bodies))
    if joints:
        sections.append("Joints:\n" + "\n".join(joints))
    if geoms:
        sections.append("Named geoms:\n" + "\n".join(geoms))
    if actuators:
        sections.append("Actuators:\n" + "\n".join(actuators))

    return "\n\n".join(sections)


def _build_user_prompt(
    obs_config: dict,
    xml_summary: str,
    k: int,
    current_reward_code: str = None,
    reflection: str = None,
) -> str:
    task = obs_config["task_description"]
    robot_type = obs_config["robot_type"]

    obs_lines = []
    for name, comp in obs_config["obs_components"].items():
        shape = comp["end"] - comp["start"]
        obs_lines.append(f"  {name} -- {comp['description']}, shape ({shape},)")
    obs_block = "\n".join(obs_lines)

    parts = [
        f"Task: {task}",
        f"Robot type: {robot_type}",
        "",
        "Observation dictionary keys:",
        obs_block,
        "",
        "MuJoCo model summary:",
        xml_summary,
    ]

    if current_reward_code:
        parts.extend(["", f"Current reward function:\n{current_reward_code}"])

    if reflection:
        parts.extend([
            "",
            f"Critique to address in the new candidates:\n{reflection}",
        ])

    parts.extend([
        "",
        f"Generate {k} DISTINCT reward function implementations.",
        "Separate each with exactly this delimiter on its own line:",
        "### REWARD_CANDIDATE ###",
        "Vary reward components, scales, and functional forms across candidates.",
    ])

    return "\n".join(parts)


def _strip_markdown_fences(code: str) -> str:
    code = re.sub(r"^```(?:python)?\s*\n?", "", code, flags=re.MULTILINE)
    code = re.sub(r"\n?```\s*$", "", code, flags=re.MULTILINE)
    return code.strip()


def _parse_candidates(response: str) -> list[str]:
    chunks = response.split("### REWARD_CANDIDATE ###")
    candidates = []
    for chunk in chunks:
        cleaned = _strip_markdown_fences(chunk.strip())
        if cleaned and "def compute_reward" in cleaned:
            candidates.append(cleaned)
    return candidates


def generate_candidates(
    obs_config: dict,
    xml_summary: str,
    k: int = 4,
    current_reward_code: str = None,
    reflection: str = None,
    client: NIMClient = None,
) -> list[str]:
    if client is None:
        client = NIMClient()

    user_prompt = _build_user_prompt(
        obs_config, xml_summary, k, current_reward_code, reflection
    )

    response = client.complete_text(SYSTEM_PROMPT, user_prompt)
    candidates = _parse_candidates(response)

    if len(candidates) < k:
        logger.warning(
            f"Requested {k} candidates but only parsed {len(candidates)} from LLM response"
        )

    if not candidates:
        logger.error("No valid candidates parsed from LLM response")
        raise ValueError(
            "LLM did not return any valid reward function candidates. "
            "Check the API response format."
        )

    return candidates
