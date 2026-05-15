import re
import logging
import xml.etree.ElementTree as ET

from .client import NIMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert reward function designer for MuJoCo locomotion RL trained with PPO.
You write Python reward functions for quadruped and biped robots.
Return ONLY the function code. No markdown fences, no explanation.
Each function must match this exact signature:
  def compute_reward(obs, info, data, model) -> tuple[float, dict]
Available in scope: numpy as np, math, mujoco.

obs is a Python dict mapping component name strings to numpy arrays:
  obs["component_name"]     -> np.ndarray for that observation slice
  obs["component_name"][0]  -> first element of that slice

info is a dict containing:
  info["action"]  -> np.ndarray (nu,) the NORMALIZED action in [-1, 1] range
Use info["action"] (NOT data.ctrl) for action/energy penalties. data.ctrl has robot-specific
magnitudes (often [-20, 45]) that make penalties scale incorrectly across robots.

VALID MjData attributes (these are the ONLY ones that exist -- do not invent others):
  data.qpos          -> np.ndarray (nq,)  joint positions; [0:3]=base xyz, [3:7]=base quat, [7:]=joint angles
  data.qvel          -> np.ndarray (nv,)  joint velocities; [0:3]=base lin vel, [3:6]=base ang vel, [6:]=joint vel
  data.ctrl          -> np.ndarray (nu,)  actuator control signals
  data.actuator_force-> np.ndarray (nu,)  actuator forces
  data.cfrc_ext      -> np.ndarray (nbody, 6)  external forces on each body
  data.xpos          -> np.ndarray (nbody, 3)  body positions in world frame
  data.xquat         -> np.ndarray (nbody, 4)  body orientations in world frame
  data.sensordata    -> np.ndarray  sensor readings (if sensors defined in XML)
  data.ncon          -> int  number of active contacts
  data.contact       -> contact array (use data.contact[i].geom1, .geom2)

DO NOT use: data.base_lin_vel, data.base_ang_vel, data.body_xpos, data.body_xquat,
            data.body_vel, data.base_pos, or any other attribute not listed above.

All array attributes must be indexed with integers or slices, NEVER strings:
  data.qvel[0:3]                           -> base linear velocity (correct)
  data.qvel["forward"]                     -> WRONG, crashes
  data.xpos[model.body("trunk").id]        -> body position by name (correct)
  mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "name")  -> joint id by name

The dict return must map component name strings to float values.
Example return: return total_reward, {"forward_vel": float(fwd), "energy": float(energy_pen)}

SURVIVAL BONUS: the environment automatically adds a survival bonus of +1.0 per step unless you
include "survival" in your components dict. If you want to tune it, add it yourself:
  return total + survival, {"forward_vel": float(fwd), "survival": float(survival)}
Do not include "survival" if you are happy with the default +1.0/step.

REWARD DESIGN PRINCIPLES (critical for PPO to learn):
1. GATE velocity reward on stability. The pattern: forward_reward = weight * velocity * exp(-k * (roll^2 + pitch^2))
   ensures the robot only gets velocity credit when upright. Without this, the robot either stands still
   (if standing rewards dominate) or sprints and crashes (if velocity reward dominates).
2. The DOMINANT component must be the task objective (e.g., forward velocity for locomotion).
   Make it the largest contributor by a factor of 5-10x over standing-still rewards at target speed.
3. Use exp(-k * error^2) for soft penalties with temperature k=1..10. This gives smooth gradients.
   Avoid hard thresholds or clamp/clip -- PPO needs differentiable signal.
4. Standing-still rewards (height bonus, survival) must be SMALL relative to velocity at target speed.
   If the robot gets more reward for standing than walking, it will stand still.
5. Action energy penalty: -0.02 * sum(info["action"]^2) prevents jittery motion.
   ALWAYS use info["action"] (normalized [-1,1]), NEVER data.ctrl (robot-specific magnitudes).
6. Base orientation penalty (keep upright): -15.0 * (roll^2 + pitch^2) or exp form.
   Extract roll/pitch from quaternion: w,x,y,z = quat; roll = atan2(2*(w*x+y*z), 1-2*(x^2+y^2))
7. Angular velocity penalty: -0.8 * sum(data.qvel[3:6]^2) prevents spinning/wobbling during gait.
8. Typical per-step reward magnitude should be 1.0-15.0. If reward is < 0.1 per step, learning is too slow.
   If reward > 50 per step, value function training destabilizes.

REFERENCE: a proven quadruped locomotion reward (produces 1 m/s stable walking):
  stability = exp(-8.0 * (roll^2 + pitch^2))
  forward_vel:    10.0 * data.qvel[0] * stability  (gated on uprightness)
  lateral_pen:    -2.0 * data.qvel[1]^2
  orientation:    -15.0 * (roll^2 + pitch^2)
  height_bonus:   2.0 * exp(-30.0 * (height - target_height)^2)
  action_energy:  -0.02 * sum(info["action"]^2)  (uses normalized action, NOT data.ctrl)
  ang_vel_pen:    -0.8 * sum(data.qvel[3:6]^2)
This is a starting point -- vary the weights, functional forms, and components across candidates."""


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
        "",
        "Requirements for EACH candidate:",
        "- Must be syntactically valid Python that runs without error",
        "- Must return (float, dict) where dict maps component names to float values",
        "- The task objective component must be the DOMINANT reward signal",
        "- Vary reward components, weights, and functional forms across candidates",
        "- At least one candidate should use exponential reward shaping (exp(-k*error^2))",
        "- At least one candidate should use linear velocity reward",
        "- Include regularization (action penalty, orientation penalty) but keep them small",
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
