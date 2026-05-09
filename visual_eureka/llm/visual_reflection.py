import base64
import io
import logging

import imageio
from PIL import Image

from .client import NIMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert robotics behavior analyst.
You watch evaluation frames of a MuJoCo robot trained with RL and identify
behavioral problems that indicate reward function issues.
Be specific, concrete, and actionable."""


def generate_visual_critique(
    gif_path: str,
    task_description: str,
    robot_type: str,
    current_reward_code: str,
    client: NIMClient,
) -> str:
    frames = imageio.mimread(gif_path, memtest=False)

    if not frames:
        raise ValueError(f"No frames found in GIF: {gif_path}")

    # NIM vision model accepts at most 1 image per request — use the middle frame
    mid_frame = frames[len(frames) // 2]
    img = Image.fromarray(mid_frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    image_b64_list = [base64.b64encode(buf.getvalue()).decode("utf-8")]

    user_text = f"""\
Task: {task_description}
Robot type: {robot_type}

Current reward function:
{current_reward_code}

The attached frames show the robot's behavior during evaluation rollout.

Identify:
1. What the robot is actually doing
2. What it should be doing for the task
3. Specific reward function changes needed (add/remove/reweight components, change functional forms)

Format your response as:
OBSERVED BEHAVIOR: ...
DESIRED BEHAVIOR: ...
REWARD CHANGES NEEDED: ..."""

    try:
        critique = client.complete_vision(SYSTEM_PROMPT, user_text, image_b64_list)
        return critique
    except Exception as e:
        logger.error(f"Visual critique generation failed: {e}")
        raise
