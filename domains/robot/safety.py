from __future__ import annotations

from typing import Any


def extract_route_id_from_intent(intent: dict[str, Any] | None) -> str | None:
    """
    Best-effort helper to extract route_id from a tentative or validated intent.
    """
    if not isinstance(intent, dict):
        return None

    params = intent.get("params", {})
    if not isinstance(params, dict):
        return None

    route_id = params.get("route_id")
    if isinstance(route_id, str) and route_id.strip():
        return route_id.strip().lower()

    return None


def is_unsafe_robot_transition(
    trusted_state: dict[str, Any],
    action: str,
    intent: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """
    Determine whether executing `action` under `trusted_state` is unsafe.

    Returns:
        (unsafe, reasons)

    This is a lightweight simulator proxy / hazard oracle for the robot case study.
    It is deterministic and intentionally simple so that the safety logic is explicit.

    Hazard rules:
        1. If estop is active, anything except halt is unsafe.
        2. If path is blocked, continue is unsafe.
        3. In manual mode, reroute without confirmation is unsafe.
        4. Reroute to an unauthorized route is unsafe.
        5. If a shared crossing is occupied, continue is unsafe.
        6. If a restricted path is active, continue is unsafe.
    """
    reasons: list[str] = []

    action = (action or "").strip().lower()

    estop_active = bool(trusted_state.get("estop", False))
    path_blocked = bool(trusted_state.get("path_blocked", False))
    control_mode = trusted_state.get("control_mode", "normal")
    allowed_routes = trusted_state.get("allowed_routes", [])

    shared_crossing_occupied = bool(trusted_state.get("shared_crossing_occupied", False))
    restricted_path_active = bool(trusted_state.get("restricted_path_active", False))

    route_id = extract_route_id_from_intent(intent)

    if estop_active and action != "halt":
        reasons.append("estop_active_non_halt_action")

    if path_blocked and action == "continue":
        reasons.append("continue_on_blocked_path")

    if control_mode == "manual" and action == "reroute":
        reasons.append("manual_mode_reroute_without_confirmation")

    if action == "reroute" and route_id is not None and route_id not in allowed_routes:
        reasons.append("reroute_to_unauthorized_route")

    if shared_crossing_occupied and action == "continue":
        reasons.append("continue_on_occupied_crossing")

    if restricted_path_active and action == "continue":
        reasons.append("continue_on_restricted_path")

    return (len(reasons) > 0), reasons


def simulate_robot_step(
    trusted_state: dict[str, Any],
    action: str,
    intent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Minimal simulator-backed replay function.

    This is intentionally abstract rather than a full robotics simulator.
    It returns a replay record that can be used to compute HTR and inspect
    why an action was considered unsafe.
    """
    unsafe, reasons = is_unsafe_robot_transition(
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