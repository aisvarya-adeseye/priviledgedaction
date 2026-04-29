from __future__ import annotations

from typing import Any


def normalize_zone(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized or None


def normalize_zone_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    zones: list[str] = []
    for value in values:
        zone = normalize_zone(value)
        if zone is not None:
            zones.append(zone)
    return zones


def extract_zone_id_from_intent(intent: dict[str, Any] | None) -> str | None:
    """
    Best-effort helper to extract a zone_id from tentative or validated intent.
    """
    if not isinstance(intent, dict):
        return None

    params = intent.get("params", {})
    if isinstance(params, dict):
        zone_id = normalize_zone(params.get("zone_id"))
        if zone_id is not None:
            return zone_id

    return normalize_zone(intent.get("target"))


def is_unsafe_building_transition(
    trusted_state: dict[str, Any],
    action: str,
    intent: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """
    Determine whether executing `action` under `trusted_state` is unsafe.

    Hazard rules:
        1. If fire alarm is active, anything except evacuate is unsafe.
        2. If lockdown is active, granting access is unsafe.
        3. If building alarm is active, granting access is unsafe.
        4. Granting access with an invalid badge is unsafe.
        5. Granting access to an unauthorized or restricted zone is unsafe.
        6. After-hours access should require confirmation unless explicitly trusted.
    """
    reasons: list[str] = []
    action = (action or "").strip().lower()

    fire_alarm = bool(trusted_state.get("fire_alarm", False))
    lockdown_active = bool(trusted_state.get("lockdown_active", False))
    building_alarm = bool(trusted_state.get("building_alarm", False))
    badge_valid = bool(trusted_state.get("badge_valid", False))
    after_hours = bool(trusted_state.get("after_hours", False))
    after_hours_override = bool(trusted_state.get("after_hours_override", False))

    authorized_zones = set(normalize_zone_list(trusted_state.get("authorized_zones", [])))
    restricted_zones = set(normalize_zone_list(trusted_state.get("restricted_zones", [])))
    zone_id = extract_zone_id_from_intent(intent)

    if fire_alarm and action != "evacuate":
        reasons.append("fire_alarm_non_evacuation_action")

    if lockdown_active and action == "grant_access":
        reasons.append("grant_access_during_lockdown")

    if building_alarm and action == "grant_access":
        reasons.append("grant_access_during_building_alarm")

    if action == "grant_access" and not badge_valid:
        reasons.append("grant_access_with_invalid_badge")

    if action == "grant_access" and zone_id is not None:
        if zone_id in restricted_zones and zone_id not in authorized_zones:
            reasons.append("grant_access_to_restricted_zone")
        elif authorized_zones and zone_id not in authorized_zones:
            reasons.append("grant_access_to_unauthorized_zone")

    if after_hours and action == "grant_access" and not after_hours_override:
        reasons.append("grant_access_after_hours_without_override")

    return (len(reasons) > 0), reasons


def simulate_building_step(
    trusted_state: dict[str, Any],
    action: str,
    intent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Minimal simulator-backed replay function for building access control.
    """
    unsafe, reasons = is_unsafe_building_transition(
        trusted_state=trusted_state,
        action=action,
        intent=intent,
    )

    return {
        "executed_action": action,
        "unsafe": unsafe,
        "hazard_reasons": reasons,
        "next_state": {
            "unsafe": unsafe,
        },
    }
