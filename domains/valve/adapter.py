from typing import Any

from core.domain_adapter import DomainAdapter
from core.types import ValidationResult, DecisionResult


class ValveAdapter(DomainAdapter):
    """
    Domain adapter for an industrial valve-control case study.

    The adapter validates tentative valve intents, normalizes them into a
    stable representation, and maps trusted process state plus validated intent
    to a final privileged action. Raw untrusted text is never used in decide().

    The conservative safe fallback for this domain is "hold".
    """

    name = "valve"

    ALLOWED_ACTIONS = {
        "open",
        "close",
        "hold",
        "vent",
        "request_confirmation",
    }

    PRIVILEGED_ROLES = {
        "admin",
        "process_engineer",
        "safety_operator",
        "maintenance_lead",
    }

    def validate(
        self,
        trusted_state: dict[str, Any],
        intent: dict[str, Any],
    ) -> ValidationResult:
        """
        Validate and normalize a tentative valve intent.

        Expected intent shape:
            {
                "action": str,
                "target": str | None,
                "params": {
                    "valve_id": str | None,
                    "flow_rate": float | int | None,
                    "duration_seconds": int | None,
                },
                "justification": str | None,
                "evidence": list | None,
            }
        """
        if not isinstance(intent, dict):
            return ValidationResult.reject("intent_not_dict")

        raw_action = intent.get("action")
        raw_target = intent.get("target")
        raw_params = intent.get("params", {})
        raw_justification = intent.get("justification")
        raw_evidence = intent.get("evidence")

        if raw_action is None:
            return ValidationResult.reject("missing_action")
        if not isinstance(raw_action, str):
            return ValidationResult.reject("action_not_string")

        action = raw_action.strip().lower()
        if action not in self.ALLOWED_ACTIONS:
            return ValidationResult.reject("invalid_action")

        if raw_target is not None and not isinstance(raw_target, str):
            return ValidationResult.reject("target_not_string")

        if raw_params is None:
            raw_params = {}
        if not isinstance(raw_params, dict):
            return ValidationResult.reject("params_not_dict")

        if raw_justification is not None and not isinstance(raw_justification, str):
            return ValidationResult.reject("justification_not_string")

        if raw_evidence is not None and not isinstance(raw_evidence, list):
            return ValidationResult.reject("evidence_not_list")

        valve_id = self._normalize_valve_id(raw_params.get("valve_id"))
        if valve_id is None and isinstance(raw_target, str):
            valve_id = self._normalize_valve_id(raw_target)

        flow_rate = raw_params.get("flow_rate")
        if flow_rate is not None:
            if not isinstance(flow_rate, (int, float)):
                return ValidationResult.reject("flow_rate_not_numeric")
            if flow_rate < 0:
                return ValidationResult.reject("flow_rate_negative")

        duration_seconds = raw_params.get("duration_seconds")
        if duration_seconds is not None:
            if not isinstance(duration_seconds, int):
                return ValidationResult.reject("duration_not_int")
            if duration_seconds <= 0 or duration_seconds > 3600:
                return ValidationResult.reject("duration_out_of_range")

        if action in {"open", "close", "vent"} and not valve_id:
            return ValidationResult.reject("missing_valve_id")

        authorized_valves = set(self._normalize_valve_list(trusted_state.get("authorized_valves", [])))
        if valve_id is not None and authorized_valves and valve_id not in authorized_valves:
            return ValidationResult.reject("unauthorized_valve")

        role = str(trusted_state.get("role", "")).strip().lower()
        valve_locked = bool(trusted_state.get("valve_locked", False))
        emergency_shutdown = bool(trusted_state.get("emergency_shutdown", False))
        leak_detected = bool(trusted_state.get("leak_detected", False))
        downstream_blocked = bool(trusted_state.get("downstream_blocked", False))
        maintenance_mode = bool(trusted_state.get("maintenance_mode", False))

        if valve_locked and action in {"open", "vent"}:
            return ValidationResult.reject("valve_locked")

        if emergency_shutdown and action in {"open", "vent"}:
            return ValidationResult.reject("emergency_shutdown_blocks_motion")

        if leak_detected and action == "open":
            return ValidationResult.reject("leak_blocks_open")

        if downstream_blocked and action == "open":
            return ValidationResult.reject("downstream_blocked")

        if maintenance_mode and action == "open" and role not in self.PRIVILEGED_ROLES:
            return ValidationResult.reject("maintenance_mode_blocks_open")

        validated_intent = {
            "action": action,
            "target": valve_id,
            "params": {
                "valve_id": valve_id,
                "flow_rate": float(flow_rate) if flow_rate is not None else None,
                "duration_seconds": duration_seconds,
            },
            "justification": raw_justification,
            "evidence": raw_evidence,
        }

        return ValidationResult.accept(validated_intent)

    def decide(
        self,
        trusted_state: dict[str, Any],
        validated_intent: dict[str, Any],
    ) -> DecisionResult:
        """
        Compute the final valve action from trusted process state.
        """
        action = validated_intent["action"]
        params = validated_intent.get("params", {})
        valve_id = params.get("valve_id")

        emergency_shutdown = bool(trusted_state.get("emergency_shutdown", False))
        leak_detected = bool(trusted_state.get("leak_detected", False))
        manual_mode = bool(trusted_state.get("manual_mode", False))
        vent_path_clear = bool(trusted_state.get("vent_path_clear", True))

        pressure_psi = float(trusted_state.get("pressure_psi", 0.0))
        max_safe_pressure_psi = float(trusted_state.get("max_safe_pressure_psi", 100.0))

        if emergency_shutdown:
            return DecisionResult(
                action="close",
                fallback_used=False,
                audit={
                    "decision_reason": "emergency_shutdown_close",
                    "requested_action": action,
                    "valve_id": valve_id,
                },
            )

        if leak_detected and action != "close":
            return DecisionResult(
                action="close",
                fallback_used=False,
                audit={
                    "decision_reason": "leak_detected_close",
                    "requested_action": action,
                    "valve_id": valve_id,
                },
            )

        if pressure_psi >= max_safe_pressure_psi and action == "open":
            return DecisionResult(
                action="vent" if vent_path_clear else "close",
                fallback_used=False,
                audit={
                    "decision_reason": "high_pressure_remap",
                    "requested_action": action,
                    "valve_id": valve_id,
                    "pressure_psi": pressure_psi,
                },
            )

        if manual_mode and action in {"open", "vent"}:
            return DecisionResult(
                action="request_confirmation",
                fallback_used=False,
                audit={
                    "decision_reason": "manual_mode_confirmation",
                    "requested_action": action,
                    "valve_id": valve_id,
                },
            )

        return DecisionResult(
            action=action,
            fallback_used=False,
            audit={
                "decision_reason": "validated_passthrough",
                "requested_action": action,
                "valve_id": valve_id,
            },
        )

    def safe_fallback(self, trusted_state: dict[str, Any]) -> str:
        """
        Conservative safe action for valve control.
        """
        return "hold"

    def _normalize_valve_id(self, value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        return normalized or None

    def _normalize_valve_list(self, values: Any) -> list[str]:
        if not isinstance(values, list):
            return []
        valves: list[str] = []
        for value in values:
            normalized = self._normalize_valve_id(value)
            if normalized is not None:
                valves.append(normalized)
        return valves
