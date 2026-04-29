from typing import Any

from core.domain_adapter import DomainAdapter
from core.types import ValidationResult, DecisionResult


class RobotAdapter(DomainAdapter):
    """
    Domain adapter for the facility robot case study.

    This adapter is responsible for three things:
      1. validating tentative robot intents,
      2. normalizing accepted intents into a clean representation,
      3. mapping trusted state + validated intent to a final action.

    The safe fallback for the robot domain is always "halt".
    """

    # Human-readable domain name used in logs and audit records.
    name = "robot"

    # Finite privileged action set for the robot case study.
    ALLOWED_VERBS = {
        "continue",
        "halt",
        "reroute",
        "request_confirmation",
    }

    def validate(
        self,
        trusted_state: dict[str, Any],
        intent: dict[str, Any],
    ) -> ValidationResult:
        """
        Validate a tentative robot intent.

        The validator should reject anything malformed, ambiguous,
        or incompatible with trusted robot state.

        Expected intent shape for this first version:
            {
                "verb": str,
                "target": str | None,
                "params": dict | None,
                "justification": str | None,
                "evidence": list | None,
            }

        For now, only a small subset is enforced:
          - "verb" must exist and be allowed
          - "target" is optional
          - "params" is optional but, if present, must be a dict
          - if verb == "reroute", route_id must be present in params
          - route_id, if present, must be allowed by trusted state
        """

        # Intent must be a dictionary.
        if not isinstance(intent, dict):
            return ValidationResult.reject("intent_not_dict")

        # Extract raw fields.
        raw_verb = intent.get("verb")
        raw_target = intent.get("target")
        raw_params = intent.get("params", {})
        raw_justification = intent.get("justification")
        raw_evidence = intent.get("evidence")

        # "verb" is required and must be a string.
        if raw_verb is None:
            return ValidationResult.reject("missing_verb")
        if not isinstance(raw_verb, str):
            return ValidationResult.reject("verb_not_string")

        # Normalize the verb for stability.
        verb = raw_verb.strip().lower()

        # Only the fixed action vocabulary is allowed.
        if verb not in self.ALLOWED_VERBS:
            return ValidationResult.reject("invalid_verb")

        # "target" is optional, but if present it must be a string or None.
        if raw_target is not None and not isinstance(raw_target, str):
            return ValidationResult.reject("target_not_string")

        # "params" is optional, but if present must be a dictionary.
        if raw_params is None:
            raw_params = {}
        if not isinstance(raw_params, dict):
            return ValidationResult.reject("params_not_dict")

        # "justification" is non-authorizing.
        # We keep it only for auditability, not for decision authority.
        if raw_justification is not None and not isinstance(raw_justification, str):
            return ValidationResult.reject("justification_not_string")

        # "evidence" is also non-authorizing.
        # For now we accept either None or a list.
        if raw_evidence is not None and not isinstance(raw_evidence, list):
            return ValidationResult.reject("evidence_not_list")

        # Pull robot-specific parameters if present.
        route_id = raw_params.get("route_id")
        speed_band = raw_params.get("speed_band")

        # If route_id exists, it must be a string.
        if route_id is not None and not isinstance(route_id, str):
            return ValidationResult.reject("route_id_not_string")

        # If speed_band exists, it must be a string in this simple version.
        if speed_band is not None and not isinstance(speed_band, str):
            return ValidationResult.reject("speed_band_not_string")

        # If the robot is being asked to reroute, route_id must be provided.
        if verb == "reroute" and not route_id:
            return ValidationResult.reject("reroute_missing_route_id")

        # If a route is named, it must be one of the trusted allowed routes.
        allowed_routes = trusted_state.get("allowed_routes", [])
        if route_id is not None and route_id not in allowed_routes:
            return ValidationResult.reject("unauthorized_route")

        # Example state-based policy check:
        # if emergency stop is active, only "halt" or "request_confirmation"
        # should survive validation.
        estop_active = trusted_state.get("estop", False)
        if estop_active and verb in {"continue", "reroute"}:
            return ValidationResult.reject("estop_blocks_motion")

        # Build the sanitized, normalized intent that downstream decision
        # logic is allowed to consume.
        validated_intent = {
            "verb": verb,
            "target": raw_target.strip().lower() if isinstance(raw_target, str) else None,
            "params": {
                "route_id": route_id,
                "speed_band": speed_band,
            },
            # These are preserved for logging / audit only.
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
        Compute the final robot action.

        Important design rule:
        This function reads only trusted_state and validated_intent.
        It does NOT read raw untrusted text.

        This first version is intentionally simple and conservative.
        """

        verb = validated_intent["verb"]
        params = validated_intent.get("params", {})
        route_id = params.get("route_id")

        # Trusted state flags used by decision logic.
        estop_active = trusted_state.get("estop", False)
        path_blocked = trusted_state.get("path_blocked", False)
        current_mode = trusted_state.get("control_mode", "normal")

        # Safety takes priority over requested action.
        if estop_active:
            return DecisionResult(
                action="halt",
                fallback_used=False,
                audit={
                    "decision_reason": "estop_active",
                    "requested_verb": verb,
                },
            )

        # If path is blocked, a raw "continue" request should not map
        # directly to continue.
        if path_blocked and verb == "continue":
            return DecisionResult(
                action="reroute",
                fallback_used=False,
                audit={
                    "decision_reason": "blocked_path_remap",
                    "requested_verb": verb,
                    "route_id": route_id,
                },
            )

        # Example mode-based guard:
        # in manual mode, risky motion changes may require confirmation.
        if current_mode == "manual" and verb == "reroute":
            return DecisionResult(
                action="request_confirmation",
                fallback_used=False,
                audit={
                    "decision_reason": "manual_mode_confirmation",
                    "requested_verb": verb,
                    "route_id": route_id,
                },
            )

        # Otherwise, pass through the validated action.
        return DecisionResult(
            action=verb,
            fallback_used=False,
            audit={
                "decision_reason": "validated_passthrough",
                "requested_verb": verb,
                "route_id": route_id,
            },
        )

    def safe_fallback(self, trusted_state: dict[str, Any]) -> str:
        """
        Conservative safe action for the robot domain.

        On validation failure, the system halts rather than continuing motion.
        """
        return "halt"