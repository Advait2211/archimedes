import logging

from .client import NIMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert RL reward function analyst.
Analyze training statistics and identify what the reward function is incentivizing incorrectly.
Be specific. Refer to named reward components in the code."""


def _describe_curve_trend(training_curve: list) -> str:
    if not training_curve or len(training_curve) < 3:
        return "insufficient data"

    rewards = [r for _, r in training_curve]
    first_third = rewards[:len(rewards) // 3]
    last_third = rewards[-(len(rewards) // 3):]

    mean_first = sum(first_third) / len(first_third)
    mean_last = sum(last_third) / len(last_third)

    recent = rewards[-5:]
    recent_std = (sum((r - sum(recent)/len(recent))**2 for r in recent) / len(recent)) ** 0.5

    if recent_std > abs(mean_last) * 0.3 and abs(mean_last) > 0.01:
        return "unstable (high variance in recent rewards)"
    elif mean_last > mean_first * 1.1:
        return "rising (reward improving over time)"
    elif abs(mean_last - mean_first) < abs(mean_first) * 0.1 + 0.01:
        return "plateaued (reward stagnant)"
    elif mean_last < mean_first * 0.9:
        return "declining (reward decreasing over time)"
    else:
        return "mixed (slight upward trend)"


def generate_text_critique(
    training_stats: dict,
    task_description: str,
    current_reward_code: str,
    client: NIMClient,
) -> str:
    mean_reward = training_stats.get("mean_reward", 0.0)
    std_reward = training_stats.get("std_reward", 0.0)
    mean_ep_length = training_stats.get("mean_ep_length", 0.0)
    success_rate = training_stats.get("success_rate")
    component_means = training_stats.get("reward_component_means", {})
    training_curve = training_stats.get("training_curve", [])

    component_block = "\n".join(
        f"    {k}: {v:.6f}" for k, v in sorted(component_means.items())
    ) if component_means else "    (no component data)"

    success_str = f"{success_rate:.2%}" if success_rate is not None else "N/A"
    curve_trend = _describe_curve_trend(training_curve)

    user_prompt = f"""\
Task: {task_description}

Current reward function:
{current_reward_code}

Training statistics:
- Mean reward: {mean_reward:.4f} +/- {std_reward:.4f}
- Mean episode length: {mean_ep_length:.1f}
- Success rate: {success_str}
- Reward component means:
{component_block}
- Training curve shape: {curve_trend}

Diagnose which components are dominating, which are too weak,
and what behaviors are likely being incentivized incorrectly.

Format:
STATISTICAL ANALYSIS: ...
LIKELY ISSUES: ...
SUGGESTED CHANGES: ..."""

    try:
        critique = client.complete_text(SYSTEM_PROMPT, user_prompt)
        return critique
    except Exception as e:
        logger.error(f"Text critique generation failed: {e}")
        raise
