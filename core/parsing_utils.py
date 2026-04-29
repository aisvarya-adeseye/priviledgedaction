"""
Shared parsing utilities for intent extraction and JSON handling.

This module consolidates duplicate parsing logic used across multiple systems
to ensure consistency and reduce maintenance burden.
"""

import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    """
    Extract JSON object from text with multiple fallback strategies.

    Strategy:
        1. Try full-text JSON parsing
        2. Try the first {...} block (non-greedy to avoid merging multiple objects)
        3. Fall back to empty dict

    Args:
        text: Raw text potentially containing JSON

    Returns:
        Parsed JSON dict, or empty dict if extraction fails
    """
    try:
        return json.loads(text)
    except Exception:
        pass

    # Use non-greedy match to avoid capturing multiple JSON objects
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return {}


def fallback_extract_action(raw_text: str) -> str:
    """
    Deterministic fallback action extraction when model output is incomplete.

    Checks for action keywords in order of priority:
        halt > reroute > request_confirmation > continue

    Args:
        raw_text: Raw user input text

    Returns:
        One of: "halt", "reroute", "request_confirmation", "continue"
    """
    lowered = raw_text.lower()

    if any(word in lowered for word in ["halt", "stop", "pause"]):
        return "halt"

    if any(word in lowered for word in ["reroute", "re-route", "divert", "alternate route"]):
        return "reroute"

    if any(word in lowered for word in [
        "request confirmation",
        "confirm",
        "ask operator",
        "need confirmation",
        "ambiguous",
    ]):
        return "request_confirmation"

    if any(word in lowered for word in ["continue", "proceed", "resume"]):
        return "continue"

    return "request_confirmation"


# Alias for consistency with robot parser terminology
fallback_extract_verb = fallback_extract_action


def normalize_target(value: str) -> str | None:
    """
    Normalize robot target identifiers to standard format.

    Converts various forms (robot_1, robot-1, Robot 1, etc.) to 'robot_<n>'.

    Args:
        value: Raw target string

    Returns:
        Normalized target like 'robot_1', or None if empty
    """
    if not value:
        return None

    lowered = value.strip().lower()
    match = re.search(r"\brobot[_-]?(\d+)\b", lowered)
    if match:
        return f"robot_{match.group(1)}"
    return lowered if lowered else None


def normalize_route_id(value: str) -> str | None:
    """
    Normalize route identifiers to standard format.

    Converts various forms (r1, route-1, Route 1, etc.) to 'r<n>'.

    Args:
        value: Raw route identifier

    Returns:
        Normalized route like 'r1', or None if empty
    """
    if not value:
        return None

    lowered = value.strip().lower()
    match = re.search(r"\br(\d+)\b", lowered)
    if match:
        return f"r{match.group(1)}"
    return lowered if lowered else None


def fallback_extract_target(raw_text: str) -> str | None:
    """
    Extract robot target identifier from raw text.

    Looks for patterns like 'robot_1', 'robot-1', 'Robot 1', etc.

    Args:
        raw_text: Raw user input text

    Returns:
        Normalized target like 'robot_1', or None if not found
    """
    match = re.search(r"\brobot[_-]?(\d+)\b", raw_text.lower())
    if not match:
        return None
    return f"robot_{match.group(1)}"


def fallback_extract_route_id(raw_text: str) -> str | None:
    """
    Extract route identifier from raw text.

    Looks for patterns like 'r1', 'route 1', 'Route-1', etc.

    Args:
        raw_text: Raw user input text

    Returns:
        Normalized route like 'r1', or None if not found
    """
    match = re.search(r"\br(\d+)\b", raw_text.lower())
    if not match:
        return None
    return f"r{match.group(1)}"


def fallback_extract_speed_band(raw_text: str) -> str | None:
    """
    Extract speed band preference from raw text.

    Looks for keywords: 'slow', 'fast', 'normal'.

    Args:
        raw_text: Raw user input text

    Returns:
        One of: 'slow', 'normal', 'fast', or None if not found
    """
    lowered = raw_text.lower()
    if "slow" in lowered:
        return "slow"
    if "fast" in lowered:
        return "fast"
    if "normal speed" in lowered or "normal" in lowered:
        return "normal"
    return None
