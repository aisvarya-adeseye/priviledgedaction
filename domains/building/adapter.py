from typing import Any

from core.domain_adapter import DomainAdapter
from core.types import ValidationResult, DecisionResult


class BuildingAdapter(DomainAdapter):
    """
    Domain adapter for the building access-control case study.

    The adapter validates tentative building intents, sanitizes them into a
    stable representation, and maps trusted state plus validated intent to a
    final privileged action. Raw text is never consulted during decision logic.

    The conservative safe fallback for this domain is always "deny".
    """

    name = "building"

    ALLOWED_ACTIONS = {
        "grant_access",
        "deny",
        "lockdown",
        "evacuate",
        "request_confirmation",
    }

    PRIVILEGED_LOCKDOWN_ROLES = {
        "admin",
        "security_operator",
        "facility_manager",
        "emergency_coordinator",
    }

    PRIVILEGED_EVACUATION_ROLES = {
        "admin",
        "security_operator",
        "facility_manager",
        "emergency_coordinator",
        "fire_warden",
    }

    def validate(
        self,
        trusted_state: dict[str, Any],
        intent: dict[str, Any],
    ) -> ValidationResult:
        """
        Validate and normalize a tentative building-control intent.

        Expected intent shape:
            {
                "action": str,
                "target": str | None,
                "params": {
                    "zone_id": str | None,
                    "duration_minutes": int | None,
                    "access_level": str | None,
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

        zone_id = self._normalize_zone(raw_params.get("zone_id"))
        if zone_id is None and isinstance(raw_target, str):
            zone_id = self._normalize_zone(raw_target)

        duration_minutes = raw_params.get("duration_minutes")
        if duration_minutes is not None:
            if not isinstance(duration_minutes, int):
                return ValidationResult.reject("duration_not_int")
            if duration_minutes <= 0 or duration_minutes > 480:
                return ValidationResult.reject("duration_out_of_range")

        access_level = raw_params.get("access_level")
        if access_level is not None:
            if not isinstance(access_level, str):
                return ValidationResult.reject("access_level_not_string")
            access_level = access_level.strip().lower()

        if action == "grant_access" and not zone_id:
            return ValidationResult.reject("grant_missing_zone")

        role = str(trusted_state.get("role", "")).strip().lower()
        badge_valid = bool(trusted_state.get("badge_valid", False))
        fire_alarm = bool(trusted_state.get("fire_alarm", False))
        building_alarm = bool(trusted_state.get("building_alarm", False))
        lockdown_active = bool(trusted_state.get("lockdown_active", False))

        authorized_zones = set(self._normalize_zone_list(trusted_state.get("authorized_zones", [])))
        restricted_zones = set(self._normalize_zone_list(trusted_state.get("restricted_zones", [])))

        if action == "grant_access":
            if not badge_valid:
                return ValidationResult.reject("invalid_badge")
            if lockdown_active:
                return ValidationResult.reject("lockdown_blocks_access")
            if fire_alarm:
                return ValidationResult.reject("fire_alarm_blocks_access")
            if building_alarm:
                return ValidationResult.reject("building_alarm_blocks_access")
            if zone_id in restricted_zones and zone_id not in authorized_zones:
                return ValidationResult.reject("restricted_zone_unauthorized")
            if authorized_zones and zone_id not in authorized_zones:
                return ValidationResult.reject("unauthorized_zone")

        if action == "lockdown" and role not in self.PRIVILEGED_LOCKDOWN_ROLES:
            return ValidationResult.reject("insufficient_role_for_lockdown")

        if action == "evacuate":
            if not fire_alarm and role not in self.PRIVILEGED_EVACUATION_ROLES:
                return ValidationResult.reject("insufficient_role_for_evacuation")

        validated_intent = {
            "action": action,
            "target": self._normalize_target(raw_target),
            "params": {
                "zone_id": zone_id,
                "duration_minutes": duration_minutes,
                "access_level": access_level,
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
        Compute the final building action from trusted state and validated intent.
        """
        action = validated_intent["action"]
        params = validated_intent.get("params", {})
        zone_id = params.get("zone_id")

        fire_alarm = bool(trusted_state.get("fire_alarm", False))
        lockdown_active = bool(trusted_state.get("lockdown_active", False))
        after_hours = bool(trusted_state.get("after_hours", False))
        role = str(trusted_state.get("role", "")).strip().lower()

        if fire_alarm:
            return DecisionResult(
                action="evacuate",
                fallback_used=False,
                audit={
                    "decision_reason": "fire_alarm_evacuation",
                    "requested_action": action,
                    "zone_id": zone_id,
                },
            )

        if lockdown_active and action not in {"lockdown", "deny"}:
            return DecisionResult(
                action="deny",
                fallback_used=False,
                audit={
                    "decision_reason": "lockdown_active_deny",
                    "requested_action": action,
                    "zone_id": zone_id,
                },
            )

        if after_hours and action == "grant_access" and role not in self.PRIVILEGED_LOCKDOWN_ROLES:
            return DecisionResult(
                action="request_confirmation",
                fallback_used=False,
                audit={
                    "decision_reason": "after_hours_confirmation",
                    "requested_action": action,
                    "zone_id": zone_id,
                },
            )

        return DecisionResult(
            action=action,
            fallback_used=False,
            audit={
                "decision_reason": "validated_passthrough",
                "requested_action": action,
                "zone_id": zone_id,
            },
        )

    def safe_fallback(self, trusted_state: dict[str, Any]) -> str:
        """
        Conservative safe action for building access control.
        """
        return "deny"

    def _normalize_target(self, value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        return normalized or None

    def _normalize_zone(self, value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        return normalized or None

    def _normalize_zone_list(self, values: Any) -> list[str]:
        if not isinstance(values, list):
            return []
        zones: list[str] = []
        for value in values:
            normalized = self._normalize_zone(value)
            if normalized is not None:
                zones.append(normalized)
        return zones
