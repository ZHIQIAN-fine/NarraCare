"""Guided input composes explicit user messages, never measurements or tool calls."""

import math

CATEGORIES = ["strength", "flexibility", "balance", "cardio", "rest"]


def rating(value, maximum, label):
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            raise ValueError(
                f"Enter {label} between 0 and {maximum}; no value is inferred."
            ) from None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= maximum
    ):
        raise ValueError(
            f"Enter {label} between 0 and {maximum}; no value is inferred."
        )
    return f"{value:g}"


def compose_checkin(
    pain, fatigue, mood, rom_status, rom, preferred, avoided, pain_prevents_task=False
):
    """Return inspectable English for the normal LLM + guarded-tool route."""
    if pain_prevents_task:
        return "Pain prevents me from doing any task today. I have not performed today's task. Please pause today's activity."
    fields = [
        f"pain {rating(pain, 10, 'pain')}",
        f"fatigue {rating(fatigue, 10, 'fatigue')}",
        f"mood {rating(mood, 10, 'mood')}",
    ]
    if rom_status == "Measured":
        fields.append(f"ROM {rating(rom, 180, 'measured ROM')}")
    elif rom_status == "Unknown / not measured":
        fields.append("ROM unknown")
    else:
        raise ValueError("Select whether ROM is measured or unknown.")
    preferred, avoided = list(preferred or []), list(avoided or [])
    if not set(preferred + avoided) <= set(CATEGORIES):
        raise ValueError("Choose categories from the listed options.")
    if set(preferred) & set(avoided):
        raise ValueError("A category cannot be both preferred and avoided.")
    text = "My current " + ", ".join(fields) + ". "
    text += (
        "I prefer " + ", ".join(preferred) + ". "
        if preferred
        else "I have no preferred categories. "
    )
    text += (
        "Avoid " + ", ".join(avoided) + ". "
        if avoided
        else "I have no category avoidances. "
    )
    return text + "Please recommend today's manageable task."


def compose_feedback(action, pain, fatigue, satisfaction):
    if action not in {"Completed", "Partial"}:
        raise ValueError("Report completed or partial only after performing the task.")
    return (
        f"{action}. After pain {rating(pain, 10, 'after-task pain')}, "
        f"fatigue {rating(fatigue, 10, 'after-task fatigue')}, "
        f"satisfaction {rating(satisfaction, 1, 'satisfaction')}. "
    )
