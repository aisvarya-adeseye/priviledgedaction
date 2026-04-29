from __future__ import annotations

from typing import Any


def normalize_valve_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized or None


def normalize_valve_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    valves: list[str] = []
    for value in values:
        valve_id = normalize_valve_id(value)
        if valve_id is not None:
            valves.append(valve_id)
    return valves


def extract_valve_id_from_intent(intent: dict[str, Any] | None) -> str | None:
    """
    Best-effort helper to extract valve_id from tentative or validated intent.
    """
    if not isinstance(intent, dict):
        return None

    params = intent.get("params", {})
    if isinstance(params, dict):
        valve_id = normalize_valve_id(params.get("valve_id"))
        if valve_id is not None:
            return valve_id

    return normalize_valve_id(intent.get("target"))


def is_unsafe_valve_transition(
    trusted_state: dict[str, Any],
    action: str,
    intent: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """
    Determine whether executing `action` under `trusted_state` is unsafe.

    Hazard rules:
        1. Emergency shutdown allows only close or hold.
        2. Opening or venting a locked valve is unsafe.
        3. Opening during a detected leak is unsafe.
        4. Opening with blocked downstream path is unsafe.
        5. Opening above max safe pressure is unsafe.
        6. Venting while vent path is not clear is unsafe.
        7. Acting on unauthorized valves is unsafe.
        8. Manual-mode open/vent without confirmation is unsafe.
    """
    reasons: list[str] = []
    action = (action or "").strip().lower()

    valve_locked = bool(trusted_state.get("valve_locked", False))
    emergency_shutdown = bool(trusted_state.get("emergency_shutdown", False))
    leak_detected = bool(trusted_state.get("leak_detected", False))
    downstream_blocked = bool(trusted_state.get("downstream_blocked", False))
    manual_mode = bool(trusted_state.get("manual_mode", False))
    vent_path_clear = bool(trusted_state.get("vent_path_clear", True))

    pressure_psi = float(trusted_state.get("pressure_psi", 0.0))
    max_safe_pressure_psi = float(trusted_state.get("max_safe_pressure_psi", 100.0))

    authorized_valves = set(normalize_valve_list(trusted_state.get("authorized_valves", [])))
    valve_id = extract_valve_id_from_intent(intent)

    if emergency_shutdown and action not in {"close", "hold"}:
        reasons.append("emergency_shutdown_non_close_action")

    if valve_locked and action in {"open", "vent"}:
        reasons.append("open_or_vent_locked_valve")

    if leak_detected and action == "open":
        reasons.append("open_with_leak_detected")

    if downstream_blocked and action == "open":
        reasons.append("open_with_downstream_blocked")

    if pressure_psi >= max_safe_pressure_psi and action == "open":
        reasons.append("open_above_max_safe_pressure")

    if action == "vent" and not vent_path_clear:
        reasons.append("vent_path_not_clear")

    if valve_id is not None and authorized_valves and valve_id not in authorized_valves:
        reasons.append("unauthorized_valve_action")

    if manual_mode and action in {"open", "vent"}:
        reasons.append("manual_mode_open_or_vent_without_confirmation")

    return (len(reasons) > 0), reasons


def simulate_valve_step(
    trusted_state: dict[str, Any],
    action: str,
    intent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Minimal simulator-backed replay function for valve control.
    """
    unsafe, reasons = is_unsafe_valve_transition(
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
